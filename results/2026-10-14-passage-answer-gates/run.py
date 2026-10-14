"""Observe gates on saved greedy trajectories, with no intervention hooks."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'
OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'protocol.json').read_text())
START=time.monotonic()

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(obj):
    temp=OUT/'answer-gates.json.tmp'
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(OUT/'answer-gates.json')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--base-verification',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    args=parser.parse_args()
    assert sha(args.weights)==P['asset_sha256']
    baseline=json.loads((ROOT/'baseline-generations.json').read_text())
    assessment=json.loads((ROOT/'baseline-assessment.json').read_text())
    grades={r['id']:r for r in assessment['rows']}
    cases=[r for r in baseline['rows'] if not r['is_answerable']]
    assert len(cases)==10 and sum(grades[r['id']]['hallucination'] for r in cases)==4
    verification=json.loads(args.base_verification.read_text())
    assert verification['repo']==P['base_repo'] and verification['revision']==P['base_revision']
    for r in verification['files']:
        path=args.model_dir/r['file']
        assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],str(path)
    (OUT/'base-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    torch.manual_seed(0);torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
         dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    weights={k:v.cuda() for k,v in load_file(args.weights).items()}
    layer=P['layer'];feature=P['transform']
    threshold=weights['nonlinearity.theta'].reshape(-1)[feature]
    assert weights['e.weight'].shape==(2480,2560)
    assert weights[f'V_{feature}'].shape==(2560,P['rank'])
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    attention_functions=[b.self_attn.forward for b in model.model.layers]
    result={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
       'protocol':P,'threshold':float(threshold),
       'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','baseline-generations.json','baseline-assessment.json']},
       'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),
                  'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},'rows':[]}
    state={}
    def observe_input(_module,inputs):
        native=inputs[0].detach()
        x=native[:,-1:,:].float()
        z=(x-weights['input_standardizer.mean'][layer])/weights['input_standardizer.std'][layer]
        # The full released gate matrix also verifies the selected projection.
        full_raw=F.linear(z,weights['e.weight'],weights['e.bias'])
        raw=F.linear(z,weights['e.weight'][feature:feature+1],weights['e.bias'][feature:feature+1])
        assert torch.allclose(raw,full_raw[...,feature:feature+1],atol=2e-5,rtol=2e-5)
        gate=raw*((raw>threshold)&(raw>0))
        lowrank=(z@weights[f'V_{feature}']@weights[f'U_{feature}'])*weights['output_standardizer.std'][layer]
        contribution=gate*lowrank
        state['gate']=float(gate.item());state['raw']=float(raw.item())
        state['lowrank_norm']=float(lowrank.norm())
        state['contribution_norm']=float(contribution.norm())
    def observe_mlp(_module,_inputs,output):
        state['native_mlp_norm']=float(output[:,-1:,:].float().norm())
        # Returning None preserves the native MLP output exactly.
    block=model.model.layers[layer]
    handles=[block.post_attention_layernorm.register_forward_pre_hook(observe_input),
             block.mlp.register_forward_hook(observe_mlp)]
    try:
        with torch.inference_mode():
            for case in cases:
                assert time.monotonic()-START<P['max_runtime_minutes']*60
                encoded=tokenizer(case['chat'],add_special_tokens=False)['input_ids']
                assert encoded==case['prompt_token_ids']
                assert hashlib.sha256(case['answer'].encode()).hexdigest()==grades[case['id']]['answer_sha256']
                prompt=torch.tensor([encoded],device='cuda')
                attention_mask=torch.ones_like(prompt)
                prediction=model(input_ids=prompt,attention_mask=attention_mask,use_cache=True)
                cache=prediction.past_key_values
                ids=case['output_token_ids']
                assert ids[-1] in eos
                assert int(prediction.logits[0,-1].argmax())==ids[0],case['id']+' first prediction differs'
                row={'id':case['id'],'source_id':case['source_id'],'passage_id':case['passage_id'],
                     'question':case['question'],'answer':case['answer'],
                     'group':'unsupported_answer' if grades[case['id']]['hallucination'] else 'abstention',
                     'ancillary_explanation_error':grades[case['id']].get('ancillary_explanation_error',False),
                     'final_prompt_token':{'full_chat_position':len(encoded)-1,'token_id':encoded[-1],
                         'token':tokenizer.decode([encoded[-1]]),**state},'answer_tokens':[]}
                for j,token in enumerate(ids):
                    attention_mask=torch.cat([attention_mask,torch.ones((1,1),device='cuda',dtype=attention_mask.dtype)],dim=1)
                    prediction=model(input_ids=torch.tensor([[token]],device='cuda'),attention_mask=attention_mask,
                                     past_key_values=cache,use_cache=True)
                    cache=prediction.past_key_values
                    if j+1<len(ids):
                        assert int(prediction.logits[0,-1].argmax())==ids[j+1],f'{case["id"]} prediction differs at {j+1}'
                    row['answer_tokens'].append({'answer_position':j,'full_chat_position':len(encoded)+j,
                       'token_id':token,'token':tokenizer.decode([token]),'special':token in tokenizer.all_special_ids,
                       'eos':token in eos,**state,
                       'contribution_over_native_mlp_norm':state['contribution_norm']/state['native_mlp_norm'] if state['native_mlp_norm'] else 0})
                visible=[r for r in row['answer_tokens'] if not r['special']]
                assert tokenizer.decode(ids,skip_special_tokens=True)==case['answer']
                row['summary']={'visible_tokens':len(visible),'active_tokens':sum(r['gate']>0 for r in visible),
                   'max_gate':max(r['gate'] for r in visible),'mean_gate':sum(r['gate'] for r in visible)/len(visible),
                   'max_contribution_norm':max(r['contribution_norm'] for r in visible),
                   'max_contribution_over_native_mlp_norm':max(r['contribution_over_native_mlp_norm'] for r in visible),
                   'strongest_tokens':sorted(visible,key=lambda r:r['gate'],reverse=True)[:5]}
                row['saved_greedy_predictions_verified']=True
                result['rows'].append(row);save(result)
                print(case['id'],row['group'],json.dumps({k:v for k,v in row['summary'].items() if k!='strongest_tokens'}),flush=True)
                assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,attention_functions))
    finally:
        for handle in handles:handle.remove()
    result['status']='complete';result['elapsed_seconds']=time.monotonic()-START
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat();save(result)
    print('Complete: all 10 cached trajectories reproduce saved greedy predictions; no intervention.',flush=True)

if __name__=='__main__':main()
