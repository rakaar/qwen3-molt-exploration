"""Resolve every changed first-token outcome into a complete greedy answer."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'changed-answers';OUT.mkdir(exist_ok=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);a=ap.parse_args()
    assert not (OUT/'results.json').exists()
    for f in json.loads((ROOT/'expected-base-verification.json').read_text())['files']:assert sha(a.model_dir/f['file'])==f['sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
    c=next(x for x in json.loads((ROOT/'cases.json').read_text()) if x['id']=='fewshot_upper_child')
    screen=json.loads((ROOT/'screen/results.json').read_text())
    trials=[r for r in screen['rows'] if r['condition']['kind']=='single_molt' and r['greedy_token_id']!=2149]
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);rows=[]
    for trial in trials:
        cond=trial['condition'];pos=cond['position'];path=ROOT/'screen'/trial['edit']['file'];assert sha(path)==trial['edit']['sha256']
        with np.load(path) as v:delta=torch.from_numpy(v['intended_delta']).cuda();native=torch.from_numpy(v['native_mlp']).cuda()
        state={'prefill':True}
        def hook(_m,_i,output):
            if not state['prefill']:return None
            assert torch.equal(output[0,pos].float(),native)
            changed=output.clone();changed[0,pos]=(output[0,pos].float()+delta).to(output.dtype);return changed
        handle=model.model.layers[cond['layer']].mlp.register_forward_hook(hook)
        try:
            ids=torch.tensor([c['prompt_token_ids']],device='cuda');mask=torch.ones_like(ids);cache=None;generated=[]
            for step in range(16):
                state['prefill']=step==0
                o=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=o.past_key_values
                logits=o.logits[0,-1].float();t=int(logits.argmax())
                if step==0:
                    p=float(logits.log_softmax(-1)[2149].exp())
                    assert p==trial['target_probability'] and t==trial['greedy_token_id']
                generated.append(t)
                if t in eos:break
                ids=torch.tensor([[t]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],1)
        finally:handle.remove()
        row={'condition':cond,'P_first_CH':p,'answer':tok.decode(generated,skip_special_tokens=True),'output_token_ids':generated}
        rows.append(row);(OUT/'results.json').write_text(json.dumps({'status':'running','rows':rows},indent=2)+'\n')
        print('FULL_CHANGED_ANSWER',cond['id'],json.dumps(row['answer']),flush=True)
    (OUT/'results.json').write_text(json.dumps({'status':'complete','rows':rows,'all_first_step_scores_reproduced':True},indent=2)+'\n')
    print('ALL_CHANGED_ANSWERS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
