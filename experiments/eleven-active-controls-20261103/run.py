"""Ablate every naturally active L29 MOLT separately at the month decision token."""
import copy
import hashlib
import json
import os
from pathlib import Path

for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:
    os.environ.pop(key,None)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
BASE=Path('/workspace/eleven-activation-20261102')
OUT=ROOT/'artifacts'
OUT.mkdir(exist_ok=True)
assert not (OUT/'results.json').exists(),'Preserve prior results'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


assert torch.__version__=='2.14.0+cu126' and transformers.__version__=='4.57.6'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
protocol=json.loads((ROOT/'protocol.json').read_text())
expected=json.loads((BASE/'expected-base-verification.json').read_text())
for row in expected['files']:
    p=BASE/'model'/row['file']
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
manifest=json.loads((ROOT/'weights/manifest.json').read_text())
assert manifest['revision']=='081ce901b090b64aeecf9bc8709b57c10411e32a'
weights={}
for feature in manifest['features']:
    assert feature['layer']==29
    path=ROOT/'weights'/feature['file'];assert sha(path)==feature['sha256']
    with np.load(path,allow_pickle=False) as a:weights[feature['transform']]={k:torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
assert list(weights)==protocol['selected_transforms']
assert sha(ROOT/'weights/all-gates.npz')==manifest['all_gate_file_sha256']
with np.load(ROOT/'weights/all-gates.npz',allow_pickle=False) as a:all_weights={k:torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
w=weights[2375]
selection=json.loads((ROOT/'selection.json').read_text())
references=json.loads((ROOT/'reference.json').read_text())
tok=AutoTokenizer.from_pretrained(BASE/'model',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(BASE/'model',local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
eos=model.generation_config.eos_token_id
eos=set(eos if isinstance(eos,list) else [eos])
probe_texts=[' '+m for name in ['january','february','march','april','may','june','july','august','september','october','november','december'] for m in [name,name.title()]]
probe_ids={text:tok.encode(text,add_special_tokens=False) for text in probe_texts}
assert all(ids for ids in probe_ids.values())


def hooks():
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


@torch.inference_mode()
def run(name,prompt,observe=False,target=None,alpha=1.):
    assert hooks()==0
    prompt_ids=tok.encode(prompt,add_special_tokens=False)
    pos=len(prompt_ids)-1
    assert pos==protocol['position'] and tok.decode([prompt_ids[pos]])==' is'
    state={'step':0,'edit_calls':0,'generated_passes_without_edit':0}
    handles=[]

    def observer(_module,inputs):
        if state['step']!=0:return
        residual=inputs[0].detach().float()
        z=(residual-w['input_mean'])/w['input_std']
        all_raw=F.linear(z,all_weights['encoder'],all_weights['bias'])
        all_gate=torch.where((all_raw>all_weights['threshold'])&(all_raw>0),all_raw,0.)
        active=torch.nonzero(all_gate[0,pos]>0).reshape(-1).tolist()
        assert active==protocol['selected_transforms']
        assert torch.allclose(all_gate[0,pos],torch.tensor(selection['all_final_gate_values'],device='cuda'),atol=3e-6,rtol=2e-6)
        state.update(all_raw=all_raw,all_gate=all_gate)
        raw=F.linear(z,w['e'].reshape(1,-1),w['bias']).squeeze(-1)
        gate=torch.where((raw>w['theta'][0])&(raw>0),raw,0.)
        state.update(residual=residual,z=z,raw=raw,gate=gate,
                     operator=(z@w['V']@w['U'])*w['output_std'])

    def editor(_module,_inputs,output):
        if state['step']!=0:
            state['generated_passes_without_edit']+=1
            return None
        state['edit_calls']+=1
        delta=torch.zeros_like(output,dtype=torch.float32)
        delta[:,pos]=float(alpha)*(float(target)-state['gate'][:,pos,None])*state['operator'][:,pos]
        changed=output.clone()
        changed[:,pos]=(output[:,pos].float()+delta[:,pos]).to(output.dtype)
        assert torch.isfinite(changed).all() and torch.isfinite(delta).all()
        assert torch.equal(changed[:,:pos],output[:,:pos])
        assert torch.count_nonzero(delta[:,:pos])==0
        if alpha==0 or float(target)==float(state['gate'][0,pos]):assert torch.equal(changed,output)
        state.update(native_mlp=output.detach().float(),edited_mlp=changed.detach().float(),
                     intended_delta=delta,actual_delta=changed.float()-output.float())
        return changed

    if observe or target is not None:
        handles.append(model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observer))
    if target is not None:
        handles.append(model.model.layers[29].mlp.register_forward_hook(editor))
    ids=torch.tensor([prompt_ids],device='cuda')
    mask,cache,generated,steps=torch.ones_like(ids),None,[],[]
    stop='token_limit'
    try:
        for step in range(protocol['max_new_tokens']):
            state['step']=step
            pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True)
            cache=pred.past_key_values
            logits=pred.logits[0,-1].float()
            assert torch.isfinite(logits).all()
            probs=logits.softmax(-1)
            chosen=int(logits.argmax())
            values,indices=probs.topk(8)
            steps.append({'step':step,'token_id':chosen,'token':tok.decode([chosen]),
                          'probability':float(probs[chosen]),
                          'logits_sha256':hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
                          'top_tokens':[{'token_id':int(i),'token':tok.decode([int(i)]),'probability':float(v)} for v,i in zip(values,indices)]})
            if step==0:
                state['first_logits']=logits.detach().cpu().numpy()
                cache0=copy.deepcopy(cache)
                state['first_probabilities']={text:float(probs[tids[0]]) for text,tids in probe_ids.items()}
            generated.append(chosen)
            if chosen in eos:
                stop='eos'
                break
            ids=torch.tensor([[chosen]],device='cuda')
            mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
    finally:
        for handle in handles:handle.remove()
    assert hooks()==0
    if target is not None:
        assert state['edit_calls']==1
        assert state['generated_passes_without_edit']==len(steps)-1
    arrays={'first_logits':state['first_logits']}
    candidate_scores={}
    for candidate_index,(text,tids) in enumerate(probe_ids.items()):
        conditional=[state['first_probabilities'][text]]
        score_cache=copy.deepcopy(cache0)
        score_mask=torch.ones((1,len(prompt_ids)),dtype=torch.long,device='cuda')
        for j,previous in enumerate(tids[:-1],start=1):
            score_ids=torch.tensor([[previous]],device='cuda')
            score_mask=torch.cat([score_mask,torch.ones_like(score_ids)],dim=1)
            scored=model(input_ids=score_ids,attention_mask=score_mask,past_key_values=score_cache,use_cache=True)
            score_cache=scored.past_key_values
            score_logits=scored.logits[0,-1].float()
            arrays[f'candidate_{candidate_index}_step_{j}']=score_logits.cpu().numpy()
            conditional.append(float(score_logits.softmax(-1)[tids[j]]))
        candidate_scores[text]={'token_ids':tids,'conditional_probabilities':conditional,
                                'probability':float(np.prod(conditional,dtype=np.float64))}


    if observe or target is not None:
        for key in ['residual','z','raw','gate','all_raw','all_gate','operator','native_mlp','edited_mlp','intended_delta','actual_delta']:
            if key in state:arrays[key]=state[key][0].cpu().numpy()
    path=OUT/(name+'-arrays.npz')
    np.savez_compressed(path,**arrays)
    row={'name':name,'prompt':prompt,'prompt_token_ids':prompt_ids,'prompt_tokens':[tok.decode([i]) for i in prompt_ids],
         'transform':selected_transform,'position':pos,'token':' is','target_gate':target,'alpha':alpha,'output_token_ids':generated,
         'response':tok.decode(generated,skip_special_tokens=True),'stop_reason':stop,'steps':steps,
         'first_token_probabilities':state['first_probabilities'],'candidate_scores':candidate_scores,
         'selected_input_positions':[pos] if target is not None else [],'generated_positions_edited':[],
         'edit_calls':state['edit_calls'],'generated_passes_without_edit':state['generated_passes_without_edit'],
         'arrays':{'file':path.name,'sha256':sha(path)}}
    if observe or target is not None:
        row.update(natural_gate=float(state['gate'][0,pos]),operator_l2=float(state['operator'][0,pos].norm()))
    return row


selected_transform=2375
result={'status':'running','protocol':protocol,'selection':selection,'controls':{},'runs':[],'repeats':[],
        'probe_token_ids':probe_ids,'probability_definition':'Conditional probabilities of exact leading-space lowercase/title-case month-name prefixes, summed by name in analysis. Exact multi-token prefixes are scored from the edited prefill cache without further edits. Other renderings, following punctuation and EOS excluded.',
        'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'dtype':'BF16 native model; FP32 gate/operator edit, one BF16 cast',
        'attention':'eager','tf32':False,'base_revision':expected['revision'],'molt_revision':manifest['revision'],
        'weights_manifest_sha256':sha(ROOT/'weights/manifest.json'),'source_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(ROOT/'protocol.json'),'selection_sha256':sha(ROOT/'selection.json'),'reference_sha256':sha(ROOT/'reference.json')}}
native=run('native',protocol['prompt'])
observed=run('observed',protocol['prompt'],observe=True)
assert native['steps']==observed['steps'] and native['output_token_ids']==observed['output_token_ids']
assert native['candidate_scores']==observed['candidate_scores']
assert native['output_token_ids']==references['native']['output_token_ids']
assert [s['logits_sha256'] for s in native['steps']]==references['native']['logits_sha256']
zero=run('zero-strength',protocol['prompt'],target=0.,alpha=0.)
assert zero['steps']==native['steps'] and zero['candidate_scores']==native['candidate_scores']
result['controls'].update(native=native,observed=observed,zero_strength=zero)
save(OUT/'results.json',result)
for selected_transform,w in weights.items():
    row=run(f'off-T{selected_transform}',protocol['prompt'],target=0.)
    assert row['natural_gate']>0
    if selected_transform==2375:assert row['output_token_ids']==references['gate_off']['output_token_ids'][:len(row['output_token_ids'])]
    result['runs'].append(row);save(OUT/'results.json',result)
    print(json.dumps({'transform':selected_transform,'gate':row['natural_gate'],'operator_l2':row['operator_l2'],'response':row['response'],'month_probabilities':row['first_token_probabilities']},ensure_ascii=False),flush=True)
for selected_transform,w in weights.items():
    row=run(f'repeat-off-T{selected_transform}',protocol['prompt'],target=0.)
    original=next(r for r in result['runs'] if r['transform']==selected_transform)
    assert row['steps']==original['steps'] and row['output_token_ids']==original['output_token_ids']
    assert row['candidate_scores']==original['candidate_scores']
    result['repeats'].append(row);save(OUT/'results.json',result)
result.update(status='complete',hooks_remaining=hooks(),native_prior_reference_exact=True,T2375_prior_reference_exact=True)
save(OUT/'results.json',result)
print('ACTIVE_CONTROL_ABLATIONS_COMPLETE',flush=True)
