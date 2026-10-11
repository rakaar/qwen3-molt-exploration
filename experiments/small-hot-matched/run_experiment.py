"""Matched original-model transcoder and MOLT contribution interventions."""
import hashlib
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from small_hot_upstream import _make_base_mlp_hook, _make_frozen_attn_forward

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'
P=json.loads((ROOT/'protocol.json').read_text())
sys.path.insert(0,str(ROOT/'source'))
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer, DimensionwiseOutputStandardizer
from crosslayer_transcoder.model.molt import Molt

torch.manual_seed(0)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
assert torch.cuda.is_available()


def save_json(name,value):
    temp=OUT/(name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(OUT/name)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class SelectedTC:
    def __init__(self):
        manifest=json.loads((ROOT/'assets/weights-manifest.json').read_text())
        self.layers={}
        for s in manifest['layers']:
            file=ROOT/'assets'/s['file'];assert sha(file)==s['sha256']
            blob=file.read_bytes();ids=[f['feature'] for f in s['features']]
            rows=[torch.frombuffer(bytearray(blob[i*10242:(i+1)*10242]),dtype=torch.bfloat16) for i in range(len(ids))]
            self.layers[s['layer']]={'ids':ids,'index':{f:i for i,f in enumerate(ids)},
                'enc':torch.stack([r[:2560] for r in rows]).cuda(),
                'dec':torch.stack([r[2560:5120] for r in rows]).cuda(),
                'bias':torch.stack([r[5120] for r in rows]).cuda()}
    def at(self,caps,layer,feature,pos):
        s=self.layers[layer];i=s['index'][feature]
        acts=F.relu(F.linear(caps.mlp_inputs[layer],s['enc'],s['bias']))
        g=acts[pos,i].float();uv=s['dec'][i].float()
        return {'gate':g,'ungated':uv,'contribution':g*uv}


class SelectedMolt:
    def __init__(self,layer,spec):
        self.layer=layer
        file=ROOT/'weights'/spec['file'];assert sha(file)==spec['sha256']
        self.w={k:v.cuda() for k,v in load_file(file).items()}
        self.ids=spec['ids']
        self.input=DimensionwiseInputStandardizer(36,2560).cuda()
        self.output=DimensionwiseOutputStandardizer(36,2560).cuda()
        self.gate=JumpReLU(theta=0.,bandwidth=1.,n_layers=1,d_features=2480).cuda()
        for module,prefix in [(self.input,'input_standardizer'),(self.output,'output_standardizer'),(self.gate,'nonlinearity')]:
            module.load_state_dict({k.split('.',1)[1]:v for k,v in self.w.items() if k.startswith(prefix+'.')},strict=True)
            module.eval().requires_grad_(False)
        self.input.is_initialized=self.output.is_initialized=True
    def at(self,caps,feature,pos):
        x=self.input(caps.prenorm[self.layer].float(),self.layer)
        gates=self.gate(F.linear(x,self.w['e.weight'],self.w['e.bias']))
        uv=(x[pos] @ self.w[f'V_{feature}']) @ self.w[f'U_{feature}']
        uv=uv*self.output.std[self.layer]
        g=gates[pos,feature]
        return {'gate':g,'ungated':uv,'contribution':g*uv}


tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen3-4b',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(ROOT/'models/qwen3-4b',local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
model.config.use_cache=False
tc=SelectedTC()
download=json.loads((OUT/'download-manifest.json').read_text())
molts={s['layer']:SelectedMolt(s['layer'],s) for s in download['selected_molt']}
recipe=json.loads((ROOT/'assets/recipe.json').read_text())
tcgroup={'small':[(f['layer'],f['feature']) for f in recipe['selection']['small (multilingual)']],
         'hot':[(f['layer'],f['feature']) for f in recipe['selection']['hot (multilingual)']]}
GROUPS={'transcoder':tcgroup,**P['molt_groups']}
archive={}
traces={}
R={'protocol':P,'runtime':{'python':platform.python_version(),'torch':torch.__version__,
    'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()},
    'feature_table':[],'prompts':[],'trials':[],'generation':[],'checks':{},
    'source_sha256':{q.name:sha(q) for q in (ROOT/'upstream').glob('*.py')}}


def encoded(prompt,word):
    rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    batch=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
    offsets=batch.pop('offset_mapping')[0].tolist()
    assert prompt.count(word)==1
    start=rendered.index(word,rendered.index(prompt));end=start+len(word)
    positions=[i for i,(a,b) in enumerate(offsets) if b>a and b>start and a<end]
    assert len(positions)==1,positions
    ids=batch['input_ids'].cuda()
    return ids,positions[0],[{'position':i,'id':int(v),'text':tokenizer.decode([int(v)]),'offset':offsets[i]} for i,v in enumerate(ids[0])]


def capture(ids):
    c=SimpleNamespace(attn_weights={},mlp_outputs={},mlp_inputs={},prenorm={})
    handles=[]
    for i,layer in enumerate(model.model.layers):
        def a(_m,_args,out,i=i):c.attn_weights[i]=out[1].detach()
        def m(_m,args,out,i=i):
            c.mlp_outputs[i]=out.detach()
            if i in tc.layers:c.mlp_inputs[i]=args[0][0].detach()
        handles.extend([layer.self_attn.register_forward_hook(a),layer.mlp.register_forward_hook(m)])
        if i in molts:
            def pre(_m,args,i=i):c.prenorm[i]=args[0][0].detach()
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pre))
    try:
        c.logits=model(ids,output_attentions=True,use_cache=False).logits[0].detach()
    finally:
        for h in handles:h.remove()
    return c


def summary(logits):
    logits=logits[-1].float();p=logits.softmax(-1);lp=logits.log_softmax(-1)
    vals,ids=p.topk(8)
    return {'P_large':float(p[16767]),'P_cold':float(p[87072]),
        'logP_large':float(lp[16767]),'logP_cold':float(lp[87072]),
        'cold_minus_large_logprob':float(lp[87072]-lp[16767]),
        'top_token':tokenizer.decode([int(ids[0])]),
        'top_tokens':[{'id':int(i),'text':tokenizer.decode([int(i)]),'probability':float(v)} for i,v in zip(ids,vals)]}


def contributions(caps,pos):
    result={}
    for group,spec in GROUPS.items():
        for role,pairs in spec.items():
            for layer,feature in pairs:
                key=(group,layer,feature)
                result[key]=tc.at(caps,layer,feature,pos) if group=='transcoder' else molts[layer].at(caps,feature,pos)
    return result


def deltas(group,mode,strength,rec,don,pos,seq,component='combined',published=False):
    out={}
    for role,pairs in GROUPS[group].items():
        if component=='suppress_only' and role!='small':continue
        if component=='inject_only' and role!='hot':continue
        for layer,feature in pairs:
            a=rec[(group,layer,feature)];b=don[(group,layer,feature)]
            target=b['gate']
            if published:
                if role=='small':target=torch.zeros_like(target)
                else:
                    f=next(f for f in recipe['selection']['hot (multilingual)'] if f['layer']==layer and f['feature']==feature)
                    target=torch.tensor(f['acts']['hot_en'],device='cuda')
            if mode=='scalar':change=(target-a['gate'])*a['ungated']
            else:
                target_vec=target*b['ungated']
                change=target_vec-a['contribution']
            if component=='suppress_only':change=-a['contribution']
            delta=out.setdefault(layer,torch.zeros(seq,2560,device='cuda',dtype=torch.float32))
            delta[pos]+=strength*change
    return out


def delta_info(ds):
    return {'total_delta_norm':float(torch.stack([d.float().square().sum() for d in ds.values()]).sum().sqrt()) if ds else 0.,
        'by_layer':[{'layer':l,'norm':float(d.norm()),'positions':torch.where(d.abs().sum(-1)>0)[0].tolist()} for l,d in sorted(ds.items())]}


@contextmanager
def edit_context(caps,ds,protocol,key=None):
    handles=[];saved={};written={}
    try:
        if protocol=='constrained':
            for i,layer in enumerate(model.model.layers):
                saved[i]=layer.self_attn.forward
                layer.self_attn.forward=_make_frozen_attn_forward(layer.self_attn,caps.attn_weights[i])
        for i,layer in enumerate(model.model.layers):
            pin=protocol=='constrained' and i<=22
            delta=ds.get(i)
            if not pin and delta is None:continue
            upstream=_make_base_mlp_hook(pin,caps.mlp_outputs.get(i) if pin else None,delta)
            def hook(m,args,out,i=i,pin=pin,delta=delta,upstream=upstream):
                res=upstream(m,args,out)
                if delta is not None:
                    before=caps.mlp_outputs[i] if pin else out
                    expected=before+delta.to(before.dtype)
                    assert torch.equal(res,expected)
                    positions=torch.where(delta.abs().sum(-1)>0)[0]
                    mask=torch.ones(delta.shape[0],device='cuda',dtype=torch.bool);mask[positions]=False
                    assert torch.equal(res[:,mask],before[:,mask])
                    written[i]={'delta':delta[positions].detach().cpu(),'before':before[0,positions].detach().cpu(),'after':res[0,positions].detach().cpu(),
                        'intended_positions':positions.cpu().tolist(),'pin':pin,'outside_positions_exactly_unchanged':True}
                return res
            handles.append(layer.mlp.register_forward_hook(hook))
        yield
    finally:
        for h in handles:h.remove()
        for i,forward in saved.items():model.model.layers[i].self_attn.forward=forward
        if key is not None:traces[key]=written


def forward(ids,caps,ds,protocol,key=None):
    with edit_context(caps,ds,protocol,key):return model(ids,use_cache=False).logits[0].detach()


def run_trial(prompt_id,ids,caps,group,mode,s,protocol,ds,component='combined',note=''):
    key=f'{prompt_id}/{protocol}/{group}/{mode}/{component}/{s}/{len(R["trials"])}'
    logits=forward(ids,caps,ds,protocol,key)
    row={'key':key,'prompt_id':prompt_id,'protocol':protocol,'group':group,'mode':mode,
         'component':component,'strength':s,'note':note,**delta_info(ds),**summary(logits)}
    R['trials'].append(row);save_json('results.json',R)
    print('TRIAL',prompt_id,protocol,group,mode,component,s,row['top_token'],round(row['P_cold'],6),flush=True)
    return row


def generate_answer(prompt_id,ids,ds,protocol,group,mode,strength):
    seq=ids.clone();generated=[]
    for step in range(6):
        caps=capture(seq)
        padded={l:F.pad(d,(0,0,0,seq.shape[1]-d.shape[0])) for l,d in ds.items()}
        logits=forward(seq,caps,padded,protocol)
        token=int(logits[-1].argmax());generated.append(token)
        if token in {tokenizer.eos_token_id,151645}:break
        seq=torch.cat([seq,torch.tensor([[token]],device='cuda')],dim=1)
    row={'prompt_id':prompt_id,'protocol':protocol,'group':group,'mode':mode,'strength':strength,
         'token_ids':generated,'answer':tokenizer.decode(generated,skip_special_tokens=True),
         'max_new_tokens':6,'history':'recipient-generated history; clean captures refreshed each step; no donor decoding'}
    R['generation'].append(row);save_json('results.json',R);print('GENERATION',row,flush=True)


@torch.inference_mode()
def main():
    started=time.monotonic()
    prompt_specs=[('primary',P['recipient'],'small','hot')]
    prompt_specs += [(f'heldout{i+1}',q,'small','hot') for i,q in enumerate(P['heldout_prompts'])]
    prompt_specs += [('reverse',P['donor'],'hot','small'),('synonym','Give one synonym of "small". Reply with only the word, nothing else.','small','hot')]
    primary=None
    for prompt_id,prompt,word,other in prompt_specs:
        donor_prompt=prompt.replace(word,other)
        ids,pos,tokens=encoded(prompt,word);donor_ids,dpos,dtokens=encoded(donor_prompt,other)
        assert ids.shape==donor_ids.shape and pos==dpos
        caps,doncaps=capture(ids),capture(donor_ids)
        rec,don=contributions(caps,pos),contributions(doncaps,dpos)
        R['prompts'].append({'id':prompt_id,'recipient':prompt,'donor':donor_prompt,'position':pos,
            'readout_position':ids.shape[1]-1,'tokens':tokens,'donor_tokens':dtokens,
            'baseline':summary(caps.logits),'donor_baseline':summary(doncaps.logits)})
        archive[prompt_id]={'recipient_prenorm':{l:x.cpu() for l,x in caps.prenorm.items()},
            'donor_prenorm':{l:x.cpu() for l,x in doncaps.prenorm.items()},
            'recipient_tc_inputs':{l:x.cpu() for l,x in caps.mlp_inputs.items()},
            'donor_tc_inputs':{l:x.cpu() for l,x in doncaps.mlp_inputs.items()},
            'recipient_mlp_operand':{l:x[0,pos].cpu() for l,x in caps.mlp_outputs.items()},
            'donor_mlp_operand':{l:x[0,dpos].cpu() for l,x in doncaps.mlp_outputs.items()},
            'selected':{str(k):{'recipient':{n:v.cpu() for n,v in rec[k].items()},
                               'donor':{n:v.cpu() for n,v in don[k].items()}} for k in rec},'position':pos}
        if prompt_id=='primary':
            assert ids.shape==(1,29) and pos==9
            assert summary(caps.logits)['top_token']=='large' and summary(doncaps.logits)['top_token']=='cold'
            primary=(ids,caps,doncaps,rec,don,pos)
            for group,spec in GROUPS.items():
                for role,pairs in spec.items():
                    for l,i in pairs:
                        a,b=rec[(group,l,i)],don[(group,l,i)]
                        label=json.loads((ROOT/'assets'/f'label-{l}-{i}.json').read_text()) if group!='transcoder' else {}
                        R['feature_table'].append({'group':group,'role':role,'layer':l,'feature':i,
                            'label':label.get('title',role+' (published selection)'),
                            'rank':label.get('rank',1),'recipient_gate':float(a['gate']),'donor_gate':float(b['gate']),
                            'recipient_contribution_norm':float(a['contribution'].norm()),'donor_contribution_norm':float(b['contribution'].norm())})
                        if group=='transcoder':
                            f=next(f for f in recipe['selection'][role+' (multilingual)'] if f['layer']==l and f['feature']==i)
                            R['feature_table'][-1]['published_donor_gate']=0. if role=='small' else f['acts']['hot_en']
            for protocol in ['constrained','propagated']:
                z=forward(ids,caps,{},protocol)
                diff=float((z-caps.logits).abs().max());assert diff==0
                R['checks'][f'{protocol}_zero_delta_max_logit_difference']=diff
                for group in GROUPS:
                    ds=deltas(group,'full',1,rec,rec,pos,ids.shape[1]);assert all(torch.count_nonzero(d)==0 for d in ds.values())
                    diff=float((forward(ids,caps,ds,protocol)-caps.logits).abs().max());assert diff==0
                    R['checks'][f'{protocol}_{group}_self_patch_max_logit_difference']=diff
            for s in P['strengths']:
                ds=deltas('transcoder','full',s,rec,don,pos,ids.shape[1],published=True)
                run_trial(prompt_id,ids,caps,'transcoder_published','full',s,'constrained',ds,note='Published stored donor values; replication control')
        protocols=['constrained','propagated']
        strengths=P['strengths'] if prompt_id=='primary' else [1,3]
        groups=list(GROUPS) if prompt_id=='primary' else ['transcoder','pair22','expanded']
        for protocol in protocols:
            for group in groups:
                modes=['full','scalar'] if prompt_id=='primary' and group!='transcoder' else ['full']
                for mode in modes:
                    for s in strengths:
                        ds=deltas(group,mode,s,rec,don,pos,ids.shape[1])
                        row=run_trial(prompt_id,ids,caps,group,mode,s,protocol,ds)
                        if prompt_id=='primary' and s in [1,3,4.5] and group in ['transcoder','pair22','expanded']:
                            generate_answer(prompt_id,ids,ds,protocol,group,mode,s)
            if prompt_id=='primary':
                for group in ['transcoder','pair22','expanded']:
                    for component in ['suppress_only','inject_only']:
                        ds=deltas(group,'full',3,rec,don,pos,ids.shape[1],component=component)
                        run_trial(prompt_id,ids,caps,group,'full',3,protocol,ds,component)
                ds=deltas('pair22','full',3,rec,don,pos,ids.shape[1])
                random={}
                generator=torch.Generator(device='cuda').manual_seed(1729)
                for l,d in ds.items():
                    r=torch.randn(d.shape,device='cuda',generator=generator);r.zero_();v=torch.randn(2560,device='cuda',generator=generator)
                    r[pos]=v/v.norm()*d.norm();random[l]=r
                run_trial(prompt_id,ids,caps,'random_pair22','random',3,protocol,random,note='Seed 1729; per-layer norm matched')
                target=deltas('transcoder','full',3,rec,don,pos,ids.shape[1])
                for group in ['pair22','expanded']:
                    ds=deltas(group,'full',3,rec,don,pos,ids.shape[1]);factor=delta_info(target)['total_delta_norm']/max(delta_info(ds)['total_delta_norm'],1e-20)
                    ds={l:d*factor for l,d in ds.items()}
                    run_trial(prompt_id,ids,caps,group,'norm_matched_full',3,protocol,ds,note=f'Total norm matched to transcoder; factor {factor:.8g}')
                ds={22:torch.zeros(ids.shape[1],2560,device='cuda')}
                ds[22][pos]=doncaps.mlp_outputs[22][0,pos].float()-caps.mlp_outputs[22][0,pos].float()
                run_trial(prompt_id,ids,caps,'whole_mlp22','full_output',1,protocol,ds,note='Complete donor MLP output at operand; site positive control')
        restored=model(ids,use_cache=False).logits[0]
        diff=float((restored-caps.logits).abs().max());assert diff==0
        R['checks'][f'{prompt_id}_restored_max_logit_difference']=diff
        save_json('results.json',R)
    torch.save(archive,OUT/'selected-contributions.pt')
    torch.save(traces,OUT/'intervention-traces.pt')
    # Secondary experiment: genuine full layer-22 MOLT replacement.
    ids,caps,doncaps,rec,don,pos=primary
    cfg=json.loads((ROOT/'models/molt/layer-22/config.json').read_text())
    with torch.device('meta'):
        molt=Molt(d_acts=2560,N=80,ranks=cfg['ranks'],nonlinearity=JumpReLU(theta=0.,bandwidth=1.,n_layers=1,d_features=2480),
            input_standardizer=DimensionwiseInputStandardizer(36,2560),output_standardizer=DimensionwiseOutputStandardizer(36,2560))
    molt.load_state_dict(load_file(ROOT/'models/molt/layer-22/checkpoint.safetensors'),assign=True,strict=True)
    molt=molt.cuda().eval().requires_grad_(False)
    molt.input_standardizer.is_initialized=molt.output_standardizer.is_initialized=True
    for i in [941,2339]:
        block=3 if i<1200 else 4;offset=i-(560 if block==3 else 1200)
        assert torch.equal(molt.Us[block][offset],molts[22].w[f'U_{i}'])
        assert torch.equal(molt.Vs[block][offset],molts[22].w[f'V_{i}'])
    recons=molt(caps.prenorm[22].float(),22)[2]
    actual=caps.mlp_outputs[22][0].float()
    sigma=molt.output_standardizer.std[22]
    error=((recons-actual)/sigma).square().mean()
    null=((actual-molt.output_standardizer.mean[22])/sigma).square().mean()
    R['reconstruction22']={'standardized_mse':float(error),'zero_output_baseline_mse':float(null),
        'error_reduction':float(1-error/null),'operand_relative_l2_error':float((recons[pos]-actual[pos]).norm()/actual[pos].norm()),
        'scope':'one recipient prompt, no held-out corpus','selected_UV_exact_match_full_checkpoint':True,'precision':'FP32 MOLT, BF16 Qwen'}
    state={};handles=[]
    def pre(_m,args):state['input']=args[0]
    def replace(_m,args,out):
        x=state.pop('input');raw=molt(x[0].float(),22)[2]
        delta=state.get('delta')
        if delta is not None:raw=raw+delta
        return raw.unsqueeze(0).to(out.dtype)
    handles=[model.model.layers[22].post_attention_layernorm.register_forward_pre_hook(pre),model.model.layers[22].mlp.register_forward_hook(replace)]
    try:
        state['delta']=None;baseline=model(ids,use_cache=False).logits[0]
        R['full_molt22_baseline']=summary(baseline)
        for mode in ['full','scalar']:
            for s in [0,1,3,4.5]:
                state['delta']=deltas('pair22',mode,s,rec,don,pos,ids.shape[1])[22]
                logits=model(ids,use_cache=False).logits[0]
                if s==0:assert torch.equal(logits,baseline)
                R['trials'].append({'prompt_id':'primary','protocol':'full_molt22_replacement','group':'pair22','mode':mode,
                    'component':'combined','strength':s,**delta_info({22:state['delta']}),**summary(logits),
                    'note':'Full layer-22 MOLT; no reconstruction-error addback; different baseline from original-model comparison'})
                save_json('results.json',R)
                print('FULL_MOLT22',mode,s,summary(logits)['top_token'],summary(logits)['P_cold'],flush=True)
    finally:
        for h in handles:h.remove()
    diff=float((model(ids,use_cache=False).logits[0]-caps.logits).abs().max());assert diff==0
    R['checks']['after_full_replacement_restoration_max_logit_difference']=diff
    R['seconds']=time.monotonic()-started
    R['peak_gpu_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    R['status']='completed'
    save_json('results.json',R)
    print('EXPERIMENT_COMPLETED',R['seconds'],len(R['trials']),flush=True)


if __name__=='__main__':
    main()
