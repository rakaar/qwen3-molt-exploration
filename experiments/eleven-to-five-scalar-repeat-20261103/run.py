"""Copy/sweep only L29-T2375's scalar at the fifth-month final input token."""
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
manifest=json.loads((BASE/'weights/manifest.json').read_text())
feature=manifest['features'][0]
assert (feature['layer'],feature['transform'])==(29,2375)
assert sha(BASE/'weights'/feature['file'])==feature['sha256']
with np.load(BASE/'weights'/feature['file'],allow_pickle=False) as a:
    w={k:torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
tok=AutoTokenizer.from_pretrained(BASE/'model',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(BASE/'model',local_files_only=True,
    dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
eos=model.generation_config.eos_token_id
eos=set(eos if isinstance(eos,list) else [eos])
probe_texts=[' may',' May',' november',' November']
probe_ids={text:tok.encode(text,add_special_tokens=False) for text in probe_texts}
assert all(len(ids)==1 for ids in probe_ids.values())
probe_ids={text:ids[0] for text,ids in probe_ids.items()}


def hooks():
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


@torch.inference_mode()
def run(name,prompt,observe=False,target=None,alpha=1.):
    assert hooks()==0
    prompt_ids=tok.encode(prompt,add_special_tokens=False)
    pos=len(prompt_ids)-1
    assert tok.decode([prompt_ids[pos]])==' is'
    state={'step':0,'edit_calls':0,'generated_passes_without_edit':0}
    handles=[]

    def observer(_module,inputs):
        if state['step']!=0:return
        residual=inputs[0].detach().float()
        z=(residual-w['input_mean'])/w['input_std']
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
                state['first_probabilities']={text:float(probs[tid]) for text,tid in probe_ids.items()}
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
    if observe or target is not None:
        for key in ['residual','z','raw','gate','operator','native_mlp','edited_mlp','intended_delta','actual_delta']:
            if key in state:arrays[key]=state[key][0].cpu().numpy()
    path=OUT/(name+'-arrays.npz')
    np.savez_compressed(path,**arrays)
    row={'name':name,'prompt':prompt,'prompt_token_ids':prompt_ids,'prompt_tokens':[tok.decode([i]) for i in prompt_ids],
         'position':pos,'token':' is','target_gate':target,'alpha':alpha,'output_token_ids':generated,
         'response':tok.decode(generated,skip_special_tokens=True),'stop_reason':stop,'steps':steps,
         'first_token_probabilities':state['first_probabilities'],
         'selected_input_positions':[pos] if target is not None else [],'generated_positions_edited':[],
         'edit_calls':state['edit_calls'],'generated_passes_without_edit':state['generated_passes_without_edit'],
         'arrays':{'file':path.name,'sha256':sha(path)}}
    if observe or target is not None:
        row.update(natural_gate=float(state['gate'][0,pos]),operator_l2=float(state['operator'][0,pos].norm()))
    return row


result={'status':'running','feature':'L29-T2375','protocol':protocol,'controls':{},'runs':[],'repeats':[],
        'probability_definition':'Unconditional next-token probabilities at the original raw prompt. Each exact case-sensitive string, including leading space, is one token; no following punctuation or EOS is included.',
        'probe_token_ids':probe_ids,'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'dtype':'BF16 native model; FP32 MOLT calculation and edit; one BF16 cast',
        'attention':'eager','tf32':False,'base_revision':expected['revision'],'molt_revision':manifest['revision'],
        'feature_sha256':feature['sha256'],'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'protocol.json')}}
for label,prompt in [('donor',protocol['donor_prompt']),('recipient',protocol['recipient_prompt'])]:
    native=run(label+'-native',prompt)
    observed=run(label+'-observed',prompt,observe=True)
    assert native['steps']==observed['steps'] and native['output_token_ids']==observed['output_token_ids']
    assert native['output_token_ids'][:12]==protocol['reference_prefixes'][label]
    result['controls'][label]={'native':native,'observed':observed,'observer_logits_exact':True}
    save(OUT/'results.json',result)
donor_gate=result['controls']['donor']['observed']['natural_gate']
assert donor_gate==protocol['donor_gate'] and donor_gate>0
assert result['controls']['recipient']['observed']['natural_gate']==0
assert result['controls']['recipient']['native']['response'].startswith(' may')
zero=run('recipient-zero-strength',protocol['recipient_prompt'],target=donor_gate,alpha=0.)
assert zero['steps']==result['controls']['recipient']['native']['steps']
result['controls']['zero_strength']=zero
for i,target in enumerate(protocol['fixed_gates']):
    row=run(f'fixed-{i:02d}',protocol['recipient_prompt'],target=target)
    result['runs'].append(row)
    if target==0:assert row['steps']==result['controls']['recipient']['native']['steps']
    save(OUT/'results.json',result)
    print(json.dumps({'target_gate':target,'response':row['response'],'first_probabilities':row['first_token_probabilities']},ensure_ascii=False),flush=True)
for target in [donor_gate,protocol['fixed_gates'][-1]]:
    baseline=next(r for r in result['runs'] if r['target_gate']==target)
    repeated=run('repeat-'+baseline['name'],protocol['recipient_prompt'],target=target)
    assert repeated['steps']==baseline['steps'] and repeated['output_token_ids']==baseline['output_token_ids']
    result['repeats'].append(repeated)
result.update(status='complete',hooks_remaining=hooks(),donor_gate_reproduced_exactly=True)
save(OUT/'results.json',result)
print('SCALAR_SWEEP_COMPLETE',flush=True)
