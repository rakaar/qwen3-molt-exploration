"""Exact native cached generations and read-only all-layer residual input capture."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'capture';OUT.mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True,type=Path);a=ap.parse_args()
    assert not (OUT/'measurements.json').exists()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    for f in expected['files']:
        path=a.model_dir/f['file'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
    (OUT/'base-verification.json').write_text(json.dumps(expected,indent=2)+'\n')
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    cases=json.loads((ROOT/'cases.json').read_text())
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    state={};rows=[];start=time.monotonic()
    def hook(layer):
        def read(_m,inputs):
            state[layer]=inputs[0][0].detach().contiguous().view(torch.uint16).cpu().numpy().copy()
            return None
        return read
    for c in cases:
        chat=tok.apply_chat_template([{'role':'user','content':c['user']}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        assert chat==c['chat'] and tok.encode(chat,add_special_tokens=False)==c['prompt_token_ids']
        # Unhooked native trajectory first.
        def replay(observe=False):
            ids=torch.tensor([c['prompt_token_ids']],device='cuda');mask=torch.ones_like(ids);cache=None
            generated=[];scores=[];segments=[];handles=[]
            if observe:handles=[b.post_attention_layernorm.register_forward_pre_hook(hook(l)) for l,b in enumerate(model.model.layers)]
            try:
                for step in range(16):
                    state.clear()
                    o=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True)
                    cache=o.past_key_values;logits=o.logits[0,-1].float();token=int(logits.argmax())
                    scores.append({'step':step,'predicted_token':token,'log_probability':float(logits.log_softmax(-1)[token])})
                    if observe:
                        assert len(state)==36
                        segments.append(np.stack([state[l] for l in range(36)],axis=0))
                    generated.append(token)
                    if token in eos:break
                    ids=torch.tensor([[token]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],1)
            finally:
                for h in handles:h.remove()
            return generated,scores,segments
        native,nscore,_=replay(False);assert native==c['output_token_ids'],c['id']
        observed,oscore,segments=replay(True)
        assert native==observed and nscore==oscore
        inputs=np.concatenate(segments,axis=1)
        processed=c['prompt_token_ids']+native[:-1]
        assert inputs.shape==(36,len(processed),2560)
        path=OUT/(c['id']+'.npz');np.savez_compressed(path,input_bf16=inputs)
        row={**c,'processed_token_ids':processed,'prediction_scores':nscore,
             'input_file':path.name,'input_sha256':sha(path),'read_only_bit_exact_logits':True}
        rows.append(row)
        (OUT/'measurements.json').write_text(json.dumps({'status':'running','rows':rows},indent=2)+'\n')
        print('CAPTURED',c['id'],json.dumps(tok.decode(native,skip_special_tokens=True)),inputs.shape,flush=True)
    for b in model.model.layers:assert not b.post_attention_layernorm._forward_pre_hooks
    (OUT/'measurements.json').write_text(json.dumps({'status':'complete','rows':rows,
        'elapsed_seconds':time.monotonic()-start,'all_unhooked_generations_reproduced':True,
        'all_readonly_prediction_scores_identical':True,'source_sha256':{f:sha(ROOT/f) for f in ['capture.py','cases.json','protocol.json']}},indent=2)+'\n')
    print('CAPTURE_COMPLETE',flush=True)

if __name__=='__main__':main()
