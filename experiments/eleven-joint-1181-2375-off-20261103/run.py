"""Joint and single L29-T1181/T2375 removals at the eleventh-month final input."""
import copy
import re
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
WEIGHTS=Path('/workspace/eleven-active-controls-20261103-retry/weights')
assert sha(WEIGHTS/'manifest.json')==protocol['weights_manifest_sha256']
manifest=json.loads((WEIGHTS/'manifest.json').read_text())
weights={}
for index in [1181,2375]:
    feature=next(f for f in manifest['features'] if f['transform']==index)
    assert feature['layer']==29 and sha(WEIGHTS/feature['file'])==feature['sha256']==protocol['feature_sha256'][str(index)]
    with np.load(WEIGHTS/feature['file'],allow_pickle=False) as a:weights[index]={k:torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
w=weights[1181]
assert all(torch.equal(w[k],weights[2375][k]) for k in ['input_mean','input_std','output_std'])
tok=AutoTokenizer.from_pretrained(BASE/'model',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(BASE/'model',local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
eos=model.generation_config.eos_token_id
eos=set(eos if isinstance(eos,list) else [eos])
probe_texts=protocol['probe_texts']
probe_ids={text:tok.encode(text,add_special_tokens=False) for text in probe_texts}
assert all(ids for ids in probe_ids.values())


def hooks():
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


@torch.inference_mode()
def run(name,prompt,observe=False,off_ids=None,alpha=1.):
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
        state.update(residual=residual,z=z)
        for index,feature_weights in weights.items():
            raw=F.linear(z,feature_weights['e'].reshape(1,-1),feature_weights['bias']).squeeze(-1)
            gate=torch.where((raw>feature_weights['theta'][0])&(raw>0),raw,0.)
            operator=(z@feature_weights['V']@feature_weights['U'])*feature_weights['output_std']
            state.update({f'raw_T{index}':raw,f'gate_T{index}':gate,f'operator_T{index}':operator})


    def editor(_module,_inputs,output):
        if state['step']!=0:
            state['generated_passes_without_edit']+=1
            return None
        state['edit_calls']+=1
        delta=torch.zeros_like(output,dtype=torch.float32)
        contribution=torch.zeros_like(state['operator_T1181'][:,pos])
        for index in off_ids:contribution=contribution+state[f'gate_T{index}'][:,pos,None]*state[f'operator_T{index}'][:,pos]
        delta[:,pos]=-float(alpha)*contribution
        changed=output.clone()
        changed[:,pos]=(output[:,pos].float()+delta[:,pos]).to(output.dtype)
        assert torch.isfinite(changed).all() and torch.isfinite(delta).all()
        assert torch.equal(changed[:,:pos],output[:,:pos])
        assert torch.count_nonzero(delta[:,:pos])==0
        if alpha==0 or not off_ids:assert torch.equal(changed,output)
        state.update(native_mlp=output.detach().float(),edited_mlp=changed.detach().float(),
                     intended_delta=delta,actual_delta=changed.float()-output.float())
        return changed

    if observe or off_ids is not None:
        handles.append(model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observer))
    if off_ids is not None:
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
    if off_ids is not None:
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

    if observe or off_ids is not None:
        for key in ['residual','z','raw_T1181','gate_T1181','operator_T1181','raw_T2375','gate_T2375','operator_T2375','native_mlp','edited_mlp','intended_delta','actual_delta']:
            if key in state:arrays[key]=state[key][0].cpu().numpy()
    path=OUT/(name+'-arrays.npz')
    np.savez_compressed(path,**arrays)
    row={'name':name,'prompt':prompt,'prompt_token_ids':prompt_ids,'prompt_tokens':[tok.decode([i]) for i in prompt_ids],
         'position':pos,'token':' is','off_transforms':off_ids if off_ids is not None else [],'alpha':alpha,'output_token_ids':generated,
         'response':tok.decode(generated,skip_special_tokens=True),'stop_reason':stop,'steps':steps,
         'first_token_probabilities':state['first_probabilities'],'candidate_scores':candidate_scores,
         'selected_input_positions':[pos] if off_ids is not None else [],'generated_positions_edited':[],
         'edit_calls':state['edit_calls'],'generated_passes_without_edit':state['generated_passes_without_edit'],
         'arrays':{'file':path.name,'sha256':sha(path)}}
    if observe or off_ids is not None:
        row.update(natural_gates={str(index):float(state[f'gate_T{index}'][0,pos]) for index in weights},operator_l2={str(index):float(state[f'operator_T{index}'][0,pos].norm()) for index in weights})
    return row


result={'status':'running','features':['L29-T1181','L29-T2375'],'protocol':protocol,'controls':{},'runs':[],'repeats':[],
        'probability_definition':'Conditional probability of exact leading-space month-name prefixes; lowercase/title-case variants summed in analysis, complete token-sequence probabilities multiplied. Punctuation, EOS and other formats excluded.',
        'probe_token_ids':probe_ids,'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'dtype':'BF16 native model; FP32 contributions summed before one BF16 output cast',
        'attention':'eager','tf32':False,'base_revision':expected['revision'],'molt_revision':manifest['revision'],
        'feature_sha256':protocol['feature_sha256'],'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json')}}
prompt=protocol['prompt']
native=run('native',prompt)
observed=run('observed',prompt,observe=True)
zero=run('zero-strength',prompt,off_ids=[1181,2375],alpha=0.)
assert native['steps']==observed['steps']==zero['steps']
assert native['candidate_scores']==observed['candidate_scores']==zero['candidate_scores']
assert native['output_token_ids']==protocol['references']['native']['output_token_ids']
assert [s['logits_sha256'] for s in native['steps']]==protocol['references']['native']['logit_hashes']
assert native['response'].lstrip().startswith('november')
result['controls'].update(native=native,observed=observed,zero_strength=zero)
save(OUT/'results.json',result)
for ids in [[1181],[2375],[1181,2375]]:
    name='off-'+'-'.join(map(str,ids));row=run(name,prompt,off_ids=ids)
    if len(ids)==1:
        ref=protocol['references'][str(ids[0])]
        assert row['output_token_ids']==ref['output_token_ids']
        assert [s['logits_sha256'] for s in row['steps']]==ref['logit_hashes']
    result['runs'].append(row);save(OUT/'results.json',result)
    print(json.dumps({'off':ids,'response':row['response'],'p_december':sum(row['candidate_scores'][s]['probability'] for s in [' december',' December']),'p_november':sum(row['candidate_scores'][s]['probability'] for s in [' november',' November'])}),flush=True)
row=run('repeat-joint',prompt,off_ids=[1181,2375])
assert row['steps']==result['runs'][-1]['steps'] and row['candidate_scores']==result['runs'][-1]['candidate_scores']
result['repeats'].append(row)
result.update(status='complete',hooks_remaining=hooks(),native_and_single_ablation_references_exact=True)
save(OUT/'results.json',result)
print('JOINT_ABLATION_COMPLETE',flush=True)
