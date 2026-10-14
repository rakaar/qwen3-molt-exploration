"""Add a positive scalar times current MOLT low-rank output; natural contribution retained."""
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
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'protocol.json').read_text())
START=time.monotonic()

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save(data):
    temp=OUT/'added-gate-results.json.tmp'
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(OUT/'added-gate-results.json')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--base-verification',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads((ROOT/'cases.json').read_text())
    reference={r['case_id']:r for r in json.loads((ROOT/'reference-baseline.json').read_text())['rows']}
    anchors={(r['case_id'],r['added_gate']):r for r in json.loads((ROOT/'previous-anchors.json').read_text())['rows']}
    cases=data['cases']
    assert [r['id'] for r in cases]==P['cases'] and len(cases)==10
    assert all(not r['is_answerable'] for r in cases)
    assert sha(args.weights)==P['asset_sha256']
    verification=json.loads(args.base_verification.read_text())
    assert verification['repo']==P['base_repo'] and verification['revision']==P['base_revision']
    for record in verification['files']:
        path=args.model_dir/record['file']
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],str(path)
    (OUT/'base-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    torch.manual_seed(0);torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
         dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    weights={k:t.cuda() for k,t in load_file(args.weights).items()}
    layer=P['layer'];feature=P['transform'];threshold=weights['nonlinearity.theta'].reshape(-1)[feature]
    assert weights['e.weight'].shape==(2480,2560) and weights[f'V_{feature}'].shape==(2560,128)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    native_attention=[b.self_attn.forward for b in model.model.layers]
    passages={p['id']:p['text'] for p in data['passages']}
    for case in cases:
        case['prompt']=P['prompt_template'].format(context=passages[case['passage_id']],question=case['question'])
        case['chat']=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        case['prompt_token_ids']=tokenizer(case['chat'],add_special_tokens=False)['input_ids']
        old=reference[case['id']]
        assert case['chat']==old['chat'] and case['prompt_token_ids']==old['prompt_token_ids']
        case['output_token_ids']=old['output_token_ids']
    result={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':P,
       'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','cases.json','reference-baseline.json','baseline-groups.json','previous-anchors.json']},
       'gate_threshold':float(threshold),'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
          'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},'rows':[]}

    @torch.inference_mode()
    def generate(case,added):
        assert time.monotonic()-START<P['max_runtime_minutes']*60
        encoded=tokenizer(case['chat'],add_special_tokens=False)['input_ids']
        assert encoded==case['prompt_token_ids']
        ids=torch.tensor([encoded],device='cuda');mask=torch.ones_like(ids)
        state={};traces=[];step=0
        def input_hook(_module,inputs):
            z=(inputs[0][:,-1:,:].detach().float()-weights['input_standardizer.mean'][layer])/weights['input_standardizer.std'][layer]
            full_raw=F.linear(z,weights['e.weight'],weights['e.bias'])
            raw=F.linear(z,weights['e.weight'][feature:feature+1],weights['e.bias'][feature:feature+1])
            assert torch.allclose(raw,full_raw[...,feature:feature+1],atol=2e-5,rtol=2e-5)
            gate=raw*((raw>threshold)&(raw>0))
            lowrank=(z@weights[f'V_{feature}']@weights[f'U_{feature}'])*weights['output_standardizer.std'][layer]
            state.update(raw=float(raw.item()),gate=float(gate.item()),lowrank=lowrank,
                         lowrank_norm=float(lowrank.norm()))
        def output_hook(_module,_inputs,output):
            eligible=step>0
            natural=state['gate'];target=natural if added is None else natural+added
            delta_gate=added if eligible and added is not None else 0.0
            delta=delta_gate*state['lowrank']
            original=output[:,-1:,:].detach().float()
            changed=None
            if eligible and added is not None:
                assert output.shape[1]==1
                changed=(output.float()+delta).to(output.dtype)
                assert torch.isfinite(changed).all()
                if delta_gate==0:assert torch.equal(output,changed)
            actual_delta=0 if changed is None else float((changed.float()-output.float()).norm())
            token=int(ids[0,-1])
            entry={'segment':'prompt_final' if step==0 else 'response',
                'answer_position':None if step==0 else step-1,
                'full_chat_position':len(encoded)-1 if step==0 else len(encoded)+step-1,
                'token_id':token,'token':tokenizer.decode([token]),'natural_raw':state['raw'],
                'natural_gate':natural,'added_gate':added if eligible else None,
                'effective_estimated_gate':target if eligible else natural,'eligible':eligible,
                'gate_delta':delta_gate,'lowrank_norm':state['lowrank_norm'],
                'natural_contribution_norm':abs(natural)*state['lowrank_norm'],
                'effective_contribution_norm':abs(target)*state['lowrank_norm'] if eligible else abs(natural)*state['lowrank_norm'],
                'added_contribution_norm':abs(delta_gate)*state['lowrank_norm'],
                'intended_delta_norm':float(delta.norm()),'actual_bf16_delta_norm':actual_delta,
                'native_mlp_norm':float(original.norm()),
                'intended_delta_over_native_mlp_norm':float(delta.norm()/original.norm())}
            traces.append(entry)
            return changed
        block=model.model.layers[layer]
        handles=[block.post_attention_layernorm.register_forward_pre_hook(input_hook),block.mlp.register_forward_hook(output_hook)]
        outputs=[];cache=None;stop='token_limit'
        try:
            for step in range(P['decoding']['max_new_tokens']):
                assert time.monotonic()-START<P['max_runtime_minutes']*60
                prediction=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True)
                cache=prediction.past_key_values
                token=int(prediction.logits[0,-1].argmax());outputs.append(token)
                if step==0 and 'output_token_ids' in case:assert token==case['output_token_ids'][0],'Prompt computation changed'
                if token in eos:stop='eos';break
                ids=torch.tensor([[token]],device='cuda')
                mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
            assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,native_attention))
        finally:
            for handle in handles:handle.remove()
        response=[t for t in traces if t['eligible']]
        if added is None or added==0:
            if 'output_token_ids' in case:assert outputs==case['output_token_ids'],case['id']+' clean/zero-addition response changed'
            assert all(t['actual_bf16_delta_norm']==0 for t in traces)
        else:
            assert all(t['effective_estimated_gate']==t['natural_gate']+added for t in response)
            assert all(t['gate_delta']==added for t in response)
        return {'trial_id':case['id']+('_native' if added is None else '_add'+str(added)),
            'case_id':case['id'],'source_id':case['source_id'],'passage_id':case['passage_id'],
            'question':case['question'],'prompt':case['prompt'],'chat':case['chat'],
            'condition':'native' if added is None else 'added_gate','added_gate':added,
            'answer':tokenizer.decode(outputs,skip_special_tokens=True),'output_token_ids':outputs,
            'stop_reason':stop,'first_output_token_unchanged':True,
            'native_or_zero_matches_baseline':outputs==case['output_token_ids'] if 'output_token_ids' in case and (added is None or added==0) else None,
            'processed_response_tokens':len(response),'final_generated_token_not_processed':True,
            'traces':traces,'summary':{'edited_positions':sum(t['actual_bf16_delta_norm']>0 for t in response),
               'natural_active_positions':sum(t['natural_gate']>0 for t in response),
               'max_natural_gate':max((t['natural_gate'] for t in response),default=0),
               'max_intended_delta_norm':max((t['intended_delta_norm'] for t in response),default=0),
               'max_delta_over_native_mlp_norm':max((t['intended_delta_over_native_mlp_norm'] for t in response),default=0)}}
    # Collect every clean response before beginning any edited condition.
    for case in cases:
        row=generate(case,None);case['output_token_ids']=row['output_token_ids']
        row['is_reference_baseline']=True;result['rows'].append(row);save(result)
        print(row['trial_id'],'STOP='+row['stop_reason'],'TEXT='+json.dumps(row['answer'],ensure_ascii=False),flush=True)
    baseline_record={'status':'complete','protocol':P,'rows':result['rows'].copy()}
    (OUT/'clean-baseline.json').write_text(json.dumps(baseline_record,ensure_ascii=False,indent=2)+'\n')
    repeat=generate(cases[0],None)
    assert repeat['output_token_ids']==cases[0]['output_token_ids']
    result['native_repeat']={'case_id':cases[0]['id'],'identical_output_token_ids':True}
    for case in cases:
        for added in [0]+P['added_gate_values']:
            row=generate(case,added)
            if (case['id'],added) in anchors:
                assert row['output_token_ids']==anchors[(case['id'],added)]['output_token_ids'],'Prior positive anchor changed'
                row['prior_anchor_matches']=True
            result['rows'].append(row);save(result)
            print(row['trial_id'],'STOP='+row['stop_reason'],'TEXT='+json.dumps(row['answer'],ensure_ascii=False),flush=True)
    assert len(result['rows'])==len(cases)*(2+len(P['added_gate_values']))
    result['status']='complete';result['elapsed_seconds']=time.monotonic()-START
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat();save(result)
    print('Complete:',len(cases),'clean baselines;',len(cases),'zero controls;',len(cases)*len(P['added_gate_values']),'positive edits; one native repeat.',flush=True)

if __name__=='__main__':main()
