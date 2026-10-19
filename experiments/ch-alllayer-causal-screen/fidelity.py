"""Check each selected effect within a single-site MOLT reconstruction baseline."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fidelity';OUT.mkdir(exist_ok=True)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);a=ap.parse_args()
    assert not (OUT/'results.json').exists()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    for item in expected['files']:assert sha(a.model_dir/item['file'])==item['sha256']
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    c=next(x for x in json.loads((ROOT/'cases.json').read_text()) if x['id']=='fewshot_upper_child')
    selected=json.loads((ROOT/'confirmation-selection.json').read_text())['features']
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    rows=[]
    for feature in selected:
        l=feature['layer'];pos=27 if feature['site']=='source' else 37
        with np.load(ROOT/f'screen/vectors/all_active_L{l}_{feature["site"]}.npz') as v:
            native=torch.from_numpy(v['native_mlp']).cuda();sum_contribution=-torch.from_numpy(v['intended_delta']).cuda()
        with np.load(ROOT/f'weights/layer-{l}.npz') as w:mean=torch.from_numpy(w['outmean']).cuda()
        with np.load(ROOT/f'screen/vectors/{feature["id"]}.npz') as v:single_delta=torch.from_numpy(v['intended_delta']).cuda()
        reconstructed=sum_contribution+mean
        for remove in [False,True]:
            replacement=reconstructed+(single_delta if remove else 0)
            state={'prefill':True};edits=[]
            def hook(_m,_i,output):
                if not state['prefill']:return None
                assert torch.equal(output[0,pos].float(),native)
                changed=output.clone();changed[0,pos]=replacement.to(output.dtype)
                other=torch.ones(output.shape[1],device='cuda',dtype=torch.bool);other[pos]=False
                assert torch.equal(changed[:,other],output[:,other])
                edits.append(float((changed[0,pos].float()-output[0,pos].float()).norm()))
                return changed
            handle=model.model.layers[l].mlp.register_forward_hook(hook)
            try:
                ids=torch.tensor([c['prompt_token_ids']],device='cuda');mask=torch.ones_like(ids);cache=None;generated=[]
                for step in range(16):
                    state['prefill']=step==0
                    o=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=o.past_key_values
                    logits=o.logits[0,-1].float();t=int(logits.argmax())
                    if step==0:
                        lp=logits.log_softmax(-1);other=logits.clone();other[2149]=-torch.inf
                        score={'P_CH':float(lp[2149].exp()),'CH_log_odds':float(logits[2149]-torch.logsumexp(other,0))}
                    generated.append(t)
                    if t in eos:break
                    ids=torch.tensor([[t]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],1)
            finally:handle.remove()
            row={'feature':feature,'remove_feature':remove,'source_MOLT_relative_L2_error':float((reconstructed-native).norm()/native.norm()),
                **score,'answer':tok.decode(generated,skip_special_tokens=True),'output_token_ids':generated,
                'actual_delta_norm':edits[0],'single_site_reconstruction_only':True}
            rows.append(row)
            (OUT/'results.json').write_text(json.dumps({'status':'running','rows':rows},indent=2)+'\n')
            print('RECON_CHECK',feature['id'],'remove',remove,'P_CH',score['P_CH'],json.dumps(row['answer']),flush=True)
    (OUT/'results.json').write_text(json.dumps({'status':'complete','rows':rows,'source_sha256':sha(Path(__file__))},indent=2)+'\n')
    print('RECON_CHECK_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
