"""Single-prompt causal screen in the native model, with exact edit audit records."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['anchors','screen'],required=True)
    ap.add_argument('--model-dir',type=Path,required=True);args=ap.parse_args()
    out=ROOT/args.phase;out.mkdir(exist_ok=True);(out/'vectors').mkdir(exist_ok=True)
    assert not (out/'results.json').exists()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    for f in expected['files']:
        path=args.model_dir/f['file'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    native_attention=[b.self_attn.forward for b in model.model.layers]
    case=next(c for c in json.loads((ROOT/'cases.json').read_text()) if c['id']==P['primary_case'])
    ids=torch.tensor([case['prompt_token_ids']],device='cuda');mask=torch.ones_like(ids)
    target=2149;runner=11237
    weights={}
    if args.phase=='screen':
        assets=json.loads((ROOT/'weights/manifest.json').read_text());assert assets['revision']==P['molt_revision']
        for a in assets['layers']:
            path=ROOT/'weights'/a['file'];assert sha(path)==a['sha256']
            with np.load(path) as f:weights[a['layer']]={k:torch.from_numpy(f[k].copy()).cuda() for k in f.files}

    def measure(logits):
        logits=logits.float();lp=logits.log_softmax(-1)
        other=logits.clone();other[target]=-torch.inf
        topv,topi=lp.exp().topk(5)
        return {'target_log_probability':float(lp[target]),'target_probability':float(lp[target].exp()),
            'target_log_odds':float(logits[target]-torch.logsumexp(other,0)),
            'CH_minus_CI':float(logits[target]-logits[runner]),'greedy_token_id':int(logits.argmax()),
            'greedy_token':tok.decode([int(logits.argmax())]),
            'top_tokens':[{'id':int(i),'token':tok.decode([int(i)]),'p':float(v)} for i,v in zip(topi,topv)]}

    baseline_logits=model(input_ids=ids,attention_mask=mask,use_cache=True).logits[0,-1].detach().float()
    assert int(baseline_logits.argmax())==target
    baseline=measure(baseline_logits)
    conditions=[{'id':'zero_strength','kind':'native_mlp','layer':3,'site':'source','position':27,'fraction':0}]
    if args.phase=='anchors':
        conditions += [{'id':f'native_L{l}_{site}','kind':'native_mlp','layer':l,'site':site,'position':pos,'fraction':1}
            for l in range(36) for site,pos in [('source',27),('prediction',37)]]
    else:
        candidates=json.loads((ROOT/'survey/candidates.json').read_text())
        assert sha(ROOT/'survey/candidates.json')==assets['candidate_list_sha256']
        conditions += [{**c,'kind':'single_molt','features':[c['transform']],'fraction':1} for c in candidates['rows']]
        groups={}
        for c in candidates['rows']:groups.setdefault((c['layer'],c['site'],c['position']),[]).append(c['transform'])
        conditions += [{'id':f'all_active_L{l}_{site}','kind':'all_active_molts','layer':l,'site':site,'position':pos,
                        'features':sorted(fs),'fraction':1} for (l,site,pos),fs in groups.items()]
        # Known inactive CH feature provides an exact zero-contribution control.
        conditions.append({'id':'inactive_L3T2334_source','kind':'inactive_control','layer':3,'site':'source',
            'position':27,'features':[2334],'fraction':1})
    result={'status':'running','phase':args.phase,'baseline':baseline,'rows':[],
        'candidate_selection_before_outcomes':args.phase=='screen','conditions':conditions,
        'base_hashes_verified':True,'torch':torch.__version__,'transformers':transformers.__version__,
        'source_sha256':sha(Path(__file__))}
    start=time.monotonic()
    for condition in conditions:
        assert time.monotonic()-start<P['max_runtime_minutes']*60
        l,pos=condition['layer'],condition['position'];state={};handles=[]
        def pre(_m,inputs):
            state['input']=inputs[0].detach()[0,pos].float()
        def edit(_m,_i,output):
            native=output[0,pos].float();components=[]
            if condition['kind']=='native_mlp':delta=-condition['fraction']*native
            else:
                w=weights[l];z=(state['input']-w['mean'])/w['std'];delta=torch.zeros_like(native)
                for f in condition['features']:
                    ix=w['ids'].tolist().index(f)
                    raw=F.linear(z,w['encoder'][ix].reshape(1,-1),w['bias'][ix:ix+1])[0]
                    gate=torch.where((raw>w['threshold'][ix])&(raw>0),raw,torch.zeros_like(raw))
                    low=(z@w[f'V_{f}']@w[f'U_{f}'])*w['outstd']
                    contribution=gate*low;delta-=condition['fraction']*contribution
                    components.append({'transform':f,'raw':float(raw),'gate':float(gate),
                        'threshold':float(w['threshold'][ix]),'contribution_norm':float(contribution.norm())})
                state['z']=z
            changed=output.clone();changed[0,pos]=(native+delta).to(output.dtype)
            other=torch.ones(output.shape[1],device='cuda',dtype=torch.bool);other[pos]=False
            assert torch.equal(output[:,other],changed[:,other])
            actual=changed[0,pos].float()-native
            if condition['fraction']==0:assert torch.equal(output,changed)
            filename=f"vectors/{condition['id']}.npz"
            np.savez_compressed(out/filename,input=state['input'].cpu().numpy(),native_mlp=native.cpu().numpy(),
                intended_delta=delta.cpu().numpy(),actual_delta=actual.cpu().numpy(),
                edited_mlp=changed[0,pos].float().cpu().numpy())
            state['audit']={'components':components,'intended_delta_norm':float(delta.norm()),
                'actual_bf16_delta_norm':float(actual.norm()),'native_mlp_norm':float(native.norm()),
                'other_positions_identical':True,'file':filename,'sha256':sha(out/filename)}
            if condition['kind']=='all_active_molts':
                reconstruction=-delta+w['outmean']
                state['audit']['reconstruction_relative_L2_error']=float((reconstruction-native).norm()/native.norm())
                state['audit']['reconstruction_cosine']=float(F.cosine_similarity(reconstruction,native,dim=0))
            return changed
        block=model.model.layers[l]
        handles=[block.post_attention_layernorm.register_forward_pre_hook(pre),block.mlp.register_forward_hook(edit)]
        try:logits=model(input_ids=ids,attention_mask=mask,use_cache=True).logits[0,-1].detach().float()
        finally:
            for h in handles:h.remove()
        metrics=measure(logits)
        if condition['fraction']==0:assert torch.equal(logits,baseline_logits)
        if condition['kind']=='inactive_control':
            assert state['audit']['actual_bf16_delta_norm']==0 and torch.equal(logits,baseline_logits)
        row={'condition':condition,**metrics,'drop_log_odds':baseline['target_log_odds']-metrics['target_log_odds'],
             'drop_CH_minus_CI':baseline['CH_minus_CI']-metrics['CH_minus_CI'],'edit':state['audit']}
        result['rows'].append(row);save(out/'results.json',result)
        print('TESTED',condition['id'],'P_CH',round(metrics['target_probability'],6),'drop_log_odds',round(row['drop_log_odds'],5),
              'greedy',json.dumps(metrics['greedy_token']),flush=True)
    assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,native_attention))
    for b in model.model.layers:assert not b.mlp._forward_hooks and not b.post_attention_layernorm._forward_pre_hooks
    endlogits=model(input_ids=ids,attention_mask=mask,use_cache=True).logits[0,-1].float()
    assert torch.equal(endlogits,baseline_logits)
    result.update(status='complete',elapsed_seconds=time.monotonic()-start,all_native_attention_unmodified=True,
        restored_unhooked_baseline_bit_exact=True)
    save(out/'results.json',result);print('PHASE_COMPLETE',args.phase,len(conditions),flush=True)

if __name__=='__main__':main()
