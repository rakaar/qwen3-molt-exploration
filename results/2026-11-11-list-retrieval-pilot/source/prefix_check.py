"""Post-run numerical check: equal-length right padding of unchanged inputs."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);a=ap.parse_args()
r=json.loads((ROOT/'artifacts/results.json').read_text());assert r['status']=='complete'
torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
with np.load(ROOT/'weights/L29-T2375.npz',allow_pickle=False) as f:w={k:torch.from_numpy(f[k].copy()).float().cuda() for k in ['input_mean','input_std','e','bias','theta']}
out=[]
with torch.inference_mode():
 for listid in sorted({x['list_id'] for x in r['cases']}):
  for fmt in ['numbered','bullets']:
   rows=[x for x in r['cases'] if x['list_id']==listid and x['format']==fmt];length=max(len(x['input_token_ids']) for x in rows);prefix=0
   while prefix<min(len(x['input_token_ids']) for x in rows) and len({x['input_token_ids'][prefix] for x in rows})==1:prefix+=1
   values=[]
   for row in rows:
    n=len(row['input_token_ids']);ids=torch.tensor([row['input_token_ids']+[tok.pad_token_id]*(length-n)],device='cuda');mask=torch.tensor([[1]*n+[0]*(length-n)],device='cuda');record={}
    def observe(module,inputs):
     z=(inputs[0].detach().float()-w['input_mean'])/w['input_std'];record['raw']=F.linear(z,w['e'].reshape(1,-1),w['bias'])[0,:,0].cpu().numpy().copy()
    h=m.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observe)
    try:m(input_ids=ids,attention_mask=mask,position_ids=torch.arange(length,device='cuda')[None,:],use_cache=True)
    finally:h.remove()
    values.append(record['raw'])
   diff=max(float(np.max(np.abs(values[0][:prefix]-x[:prefix]))) for x in values[1:]);out.append(dict(list_id=listid,format=fmt,padded_length=length,shared_prefix_tokens=prefix,max_shared_prefix_raw_difference=diff))
result=dict(posthoc_numerical_diagnostic=True,purpose='Check shared-prefix numerical invariance at equal GEMM/prefill lengths; no generated answers or gate edits.',checks=out,maximum_raw_difference=max(x['max_shared_prefix_raw_difference'] for x in out),all_equal=all(x['max_shared_prefix_raw_difference']==0 for x in out),generated_tokens=0,interventions=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'artifacts/prefix-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
