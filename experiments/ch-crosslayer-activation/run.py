"""Read-only scalar gates at each selected MOLT's own native input location."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
(OUT/'vectors').mkdir(exist_ok=True)
START=time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(name,value):
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def deadline():assert time.monotonic()-START<P['max_runtime_minutes']*60


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);args=ap.parse_args()
    assert not (OUT/'measurements.json').exists()
    base=json.loads((ROOT/'expected-base-verification.json').read_text())
    assert base['repo']==P['base_repo'] and base['revision']==P['base_revision']
    for f in base['files']:
        path=args.model_dir/f['file']
        assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256'],f['file']
    save('base-verification.json',base)
    torch.manual_seed(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
    original_attention=[b.self_attn.forward for b in model.model.layers]
    original_mlp=[b.mlp.forward for b in model.model.layers]
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    cases=json.loads((ROOT/'cases.json').read_text());assert len(cases)==10
    baselines=[]
    # Independently regenerate with no hooks before observing any feature gates.
    for case in cases:
        deadline()
        chat=tok.apply_chat_template([{'role':'user','content':case['user']}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
        assert chat==case['chat'] and enc['input_ids']==case['prompt_token_ids']
        a,b=[chat.index(case['user'])+v for v in case['target_span']]
        positions=[i for i,(x,y) in enumerate(enc['offset_mapping']) if y>x and y>a and x<b]
        assert positions==case['target_positions'] and chat[a:b]==case['target']
        ids=torch.tensor([enc['input_ids']],device='cuda');mask=torch.ones_like(ids);cache=None;generated=[]
        for step in range(P['decoding']['max_new_tokens']):
            deadline();pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True)
            cache=pred.past_key_values;token=int(pred.logits[0,-1].argmax());generated.append(token)
            if token in eos:break
            ids=torch.tensor([[token]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
        assert generated==case['output_token_ids'],case['id']
        baselines.append({**case,'answer':tok.decode(generated,skip_special_tokens=True),
            'output_token_ids':generated,'native_generation_unhooked':True})
        print('NATIVE',case['id'],json.dumps(baselines[-1]['answer']),flush=True)
        del pred,cache
    save('baseline.json',{'rows':baselines,'all_prior_baselines_reproduced':True})
    assets=json.loads((ROOT/'weights/manifest.json').read_text())
    assert assets['revision']==P['molt_revision'] and len(assets['features'])==7
    weights={};by_layer={}
    for f in assets['features']:
        name=f"L{f['layer']}-T{f['transform']}";path=ROOT/'weights'/f['file']
        assert sha(path)==f['sha256']
        with np.load(path) as a:weights[name]={k:torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
        by_layer.setdefault(f['layer'],[]).append(name)
    state={};result={'status':'running','protocol':P,'assets':assets,'rows':[],
        'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
            'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},
        'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','cases.json','prepare.py']}}

    def hook_for(layer,names):
        def observe(_module,inputs):
            h=inputs[0].detach().float()
            for name in names:
                w=weights[name];z=(h-w['input_mean'])/w['input_std']
                raw=F.linear(z,w['e'].reshape(1,-1),w['bias']).squeeze(-1)
                independent=(z*w['e']).sum(-1)+w['bias'][0]
                assert torch.allclose(raw,independent,atol=3e-5,rtol=3e-5)
                gate=torch.where((raw>w['theta'][0])&(raw>0),raw,torch.zeros_like(raw))
                state['features'][name]={'raw':raw[0].cpu().tolist(),'gate':gate[0].cpu().tolist()}
                if state['segment']=='prompt':
                    positions=state['case']['target_positions']
                    filename=f"vectors/{state['case']['id']}-{name}.npz"
                    assert not (OUT/filename).exists()
                    np.savez_compressed(OUT/filename,native_input=h[0,positions].cpu().numpy(),
                        standardized_input=z[0,positions].cpu().numpy(),
                        raw=raw[0,positions].cpu().numpy(),gate=gate[0,positions].cpu().numpy(),
                        positions=np.array(positions,dtype=np.int64))
                    state['vectors'].append({'feature':name,'file':filename,'sha256':sha(OUT/filename)})
            return None
        return observe

    def entries(tokens,segment,offset,targets):
        assert set(state['features'])==set(weights)
        return [{'segment':segment,'position':offset+j,'token_id':t,'token':tok.decode([t]),
            'target_token':offset+j in targets,
            'features':{name:{k:v[j] for k,v in state['features'][name].items()} for name in weights}}
            for j,t in enumerate(tokens)]

    handles=[model.model.layers[layer].post_attention_layernorm.register_forward_pre_hook(hook_for(layer,names))
        for layer,names in by_layer.items()]
    try:
        for case in baselines:
            deadline();state.update(case=case,segment='prompt',features={},vectors=[])
            ids=torch.tensor([case['prompt_token_ids']],device='cuda');mask=torch.ones_like(ids)
            pred=model(input_ids=ids,attention_mask=mask,use_cache=True);cache=pred.past_key_values
            assert int(pred.logits[0,-1].argmax())==case['output_token_ids'][0]
            row={**case,'prompt_tokens':entries(case['prompt_token_ids'],'prompt',0,case['target_positions']),
                'answer_tokens':[],'vector_files':state['vectors'].copy()}
            for j,token in enumerate(case['output_token_ids']):
                if token in eos:break
                deadline();state.update(segment='answer',features={})
                ids=torch.tensor([[token]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
                pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values
                row['answer_tokens']+=entries([token],'answer',len(case['prompt_token_ids'])+j,[])
                if j+1<len(case['output_token_ids']):assert int(pred.logits[0,-1].argmax())==case['output_token_ids'][j+1]
            row['read_only_replay_matches_native_tokens']=True
            assert len(row['vector_files'])==7
            result['rows'].append(row);save('measurements.json',result)
            print('OBSERVED',case['id'],json.dumps([{'token':t['token'],'features':t['features']} for t in row['prompt_tokens'] if t['target_token']]),flush=True)
            del pred,cache
    finally:
        for handle in handles:handle.remove()
    assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,original_attention))
    assert all(b.mlp.forward==f for b,f in zip(model.model.layers,original_mlp))
    assert all(not model.model.layers[layer].post_attention_layernorm._forward_pre_hooks for layer in by_layer)
    assert len(result['rows'])==10
    result.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-START,native_attention_and_mlp_unmodified=True,
        all_hooks_return_none=True,all_prior_baselines_reproduced=True,all_replays_token_identical=True)
    save('measurements.json',result);print('COMPLETE: ten prompts, seven natural gate traces, no interventions.',flush=True)


if __name__=='__main__':main()
