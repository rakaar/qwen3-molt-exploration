"""Dynamic input-only zero ablations of L3-T1372 and L28-T1123 in native Qwen."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True);(OUT/'vectors').mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(value):
    tmp=OUT/'results.tmp';tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);a=ap.parse_args()
    assert not (OUT/'results.json').exists()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    assert expected['revision']==P['base_revision'] and expected['repo']==P['base_repo']
    for item in expected['files']:
        path=a.model_dir/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    native_attention=[b.self_attn.forward for b in model.model.layers]
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    assets=json.loads((ROOT/'weights/manifest.json').read_text());assert assets['molt_revision']==P['molt_revision']
    weights={}
    for f in assets['features']:
        path=ROOT/'weights'/f['file'];assert sha(path)==f['sha256']
        with np.load(path) as arrays:weights[f['layer']]={k:torch.from_numpy(arrays[k].copy()).float().cuda() for k in arrays.files}
    cases=json.loads((ROOT/'cases.json').read_text());assert len(cases)==2
    conditions=P['conditions'];assert len(conditions)==5
    result={'status':'running','protocol':P,'assets':assets,'rows':[],'base_hashes_verified':True,
        'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','cases.json']},
        'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()}}
    started=time.monotonic()

    def run(case,condition,forced=None):
        chat=tok.apply_chat_template([{'role':'user','content':case['user']}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
        assert chat==case['chat'] and enc['input_ids']==case['prompt_token_ids']
        word_start=chat.index(case['user'])+case['target_span'][0];word_end=chat.index(case['user'])+case['target_span'][1]
        positions=[i for i,(x,y) in enumerate(enc['offset_mapping']) if y>x and y>word_start and x<word_end]
        assert positions==case['target_positions']==[27] and tok.decode([enc['input_ids'][27]])=='child'
        state={'prefill':True};handles=[];observations=[];vectors=[]
        mode='forced' if forced is not None else 'free'
        for layer,w in weights.items():
            fraction=float(condition['removal_fractions'].get(str(layer),0))
            def pre(_m,inputs,layer=layer,w=w):
                if not state['prefill']:return None
                h=inputs[0].detach()[0,27].float();z=(h-w['mean'])/w['std']
                raw=F.linear(z,w['encoder'].reshape(1,-1),w['bias'].reshape(1))[0]
                gate=torch.where((raw>w['threshold'][0])&(raw>0),raw,torch.zeros_like(raw))
                low=(z@w['V']@w['U'])*w['outstd']
                state[layer]={'input':h,'z':z,'raw':raw,'gate':gate,'low':low,'contribution':gate*low}
                return None
            def edit(_m,_i,output,layer=layer,w=w,fraction=fraction):
                if not state['prefill']:return None
                v=state[layer];delta=-fraction*v['contribution'];changed=output.clone()
                changed[0,27]=(output[0,27].float()+delta).to(output.dtype)
                other=torch.ones(output.shape[1],device='cuda',dtype=torch.bool);other[27]=False
                assert torch.equal(output[:,other],changed[:,other])
                actual=changed[0,27].float()-output[0,27].float()
                if fraction==0:assert torch.equal(output,changed)
                name=f"vectors/{case['id']}-{condition['id']}-{mode}-L{layer}.npz"
                np.savez_compressed(OUT/name,input=v['input'].cpu().numpy(),z=v['z'].cpu().numpy(),
                    raw=np.array([float(v['raw'])],dtype=np.float32),gate=np.array([float(v['gate'])],dtype=np.float32),
                    lowrank=v['low'].cpu().numpy(),contribution=v['contribution'].cpu().numpy(),
                    native_mlp=output[0,27].float().cpu().numpy(),edited_mlp=changed[0,27].float().cpu().numpy(),
                    intended_delta=delta.cpu().numpy(),actual_delta=actual.cpu().numpy())
                obs={'layer':layer,'transform':1372 if layer==3 else 1123,'position':27,'segment':'input',
                    'raw':float(v['raw']),'natural_gate':float(v['gate']),'removed_fraction':fraction,
                    'estimated_remaining_gate':(1-fraction)*float(v['gate']),
                    'contribution_norm':float(v['contribution'].norm()),'actual_delta_norm':float(actual.norm()),
                    'native_mlp_norm':float(output[0,27].float().norm()),'other_positions_byte_identical':True,
                    'vector_file':name,'vector_sha256':sha(OUT/name)}
                observations.append(obs);vectors.append(name)
                return changed
            if condition['id']!='native':
                block=model.model.layers[layer]
                handles.extend([block.post_attention_layernorm.register_forward_pre_hook(pre),block.mlp.register_forward_hook(edit)])
        ids=torch.tensor([enc['input_ids']],device='cuda');mask=torch.ones_like(ids);cache=None;output_ids=[];scores=[]
        target_step=0 if case['id']=='fewshot_upper_child' else 2
        target=2149 if target_step==0 else 521
        foil=tok.encode('CI' if target_step==0 else ' ci',add_special_tokens=False);assert len(foil)==1
        try:
            for step in range(len(forced) if forced is not None else 16):
                assert time.monotonic()-started<P['max_runtime_minutes']*60
                state['prefill']=step==0
                pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values
                logits=pred.logits[0,-1].float();lp=logits.log_softmax(-1);greedy=int(logits.argmax())
                chosen=forced[step] if forced is not None else greedy
                others=logits.clone();others[target]=-torch.inf
                values,indices=lp.exp().topk(5)
                scores.append({'step':step,'chosen_token_id':chosen,'chosen_token':tok.decode([chosen]),
                    'greedy_token_id':greedy,'greedy_token':tok.decode([greedy]),
                    'target_token_id':target,'target_token':tok.decode([target]),
                    'target_probability':float(lp[target].exp()),'target_log_probability':float(lp[target]),
                    'target_log_odds':float(logits[target]-torch.logsumexp(others,0)),
                    'target_minus_foil_logit':float(logits[target]-logits[foil[0]]),
                    'foil_token_id':foil[0],'foil_token':tok.decode(foil),
                    'top_tokens':[{'id':int(i),'token':tok.decode([int(i)]),'p':float(v)} for i,v in zip(indices,values)]})
                output_ids.append(chosen)
                if chosen in eos:break
                ids=torch.tensor([[chosen]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],1)
        finally:
            for handle in handles:handle.remove()
        if condition['id']!='native':assert len(observations)==2 and all(o['position']==27 for o in observations)
        for b in model.model.layers:assert not b.mlp._forward_hooks and not b.post_attention_layernorm._forward_pre_hooks
        return {'answer':tok.decode(output_ids,skip_special_tokens=True),'output_token_ids':output_ids,
            'scores':scores,'target_step':target_step,'target_score':scores[target_step] if len(scores)>target_step else None,
            'input_observations':observations,'generated_tokens_edited':False}

    for case in cases:
        native=None
        for condition in conditions:
            free=run(case,condition);forced=run(case,condition,case['output_token_ids'])
            if condition['id']=='native':
                assert free['output_token_ids']==case['output_token_ids'];native={'free':free,'forced':forced}
            if condition['id']=='zero_control':
                assert free['output_token_ids']==native['free']['output_token_ids']
                assert free['scores']==native['free']['scores'] and forced['scores']==native['forced']['scores']
            row={'case_id':case['id'],'condition':condition,'user':case['user'],'free':free,'forced':forced}
            result['rows'].append(row);save(result)
            print('JOINT_TEST',case['id'],condition['id'],json.dumps(free['answer']),
                'P_target',forced['target_score']['target_probability'],
                'source_gates',[(x['layer'],round(x['natural_gate'],6),x['removed_fraction']) for x in free['input_observations']],flush=True)
        restored=run(case,conditions[0]);assert restored['scores']==native['free']['scores']
    assert len(result['rows'])==10 and all(b.self_attn.forward==f for b,f in zip(model.model.layers,native_attention))
    result.update(status='complete',elapsed_seconds=time.monotonic()-started,
        native_attention_unchanged=True,restored_native_scores_bit_exact=True,all_edits_input_only=True)
    save(result);print('JOINT_ABLATION_COMPLETE',len(result['rows']),flush=True)

if __name__=='__main__':main()
