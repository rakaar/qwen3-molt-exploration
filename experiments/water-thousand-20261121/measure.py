"""Natural gates from prefixes ending on the labeled token. No text generation."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
import numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM
def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha('L29-T2375.npz')=='8ae225a36fcfd920adf9cd06aacfbcc53011343b0fbb4e8880cc7ff97111e57c'
lock=load('label-freeze.json')
assert sha('sample.json')==lock['sample_sha256'] and sha('labels.json')==lock['labels_sha256']
assert load('model-verified.json')['revision']=='1cfa9a7208912126459214e8b04321603b3df60c'
state=np.load('L29-T2375.npz')
w={k:torch.tensor(state[k],device='cuda',dtype=torch.float32) for k in ['e','bias','theta','input_mean','input_std']}
torch.backends.cuda.matmul.allow_tf32=False
model=AutoModelForCausalLM.from_pretrained('model',dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').eval().cuda().requires_grad_(False)
result=[]
def observe(module,args):
 x=args[0][0,-1].float()
 z=(x-w['input_mean'])/w['input_std']
 raw=torch.dot(z,w['e'])+w['bias'][0]
 val=raw if raw>w['theta'][0] and raw>0 else raw.new_zeros(())
 result.append((float(raw),float(val)))
model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observe)
@torch.inference_mode()
def run(ids,ident):
 assert ids[-1]==ident and not result
 model.model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False)
 assert len(result)==1
 return result.pop()
checks=[]
for r in load('calibration-input.json'):
 _,v=run(r.pop('ids'),r['token_id']);checks.append(dict(**r,new_gate=v))
err=max(abs(r['gate']-r['new_gate']) for r in checks)
agree=sum((r['gate']>0)==(r['new_gate']>0) for r in checks)
Path('calibration.json').write_text(json.dumps(dict(rows=checks,agreement=agree,max_error=err),indent=2)+'\n')
print('calibration',agree,len(checks),err,flush=True)
assert agree==len(checks) and err<0.15
inputs=load('input_ids.json'); rows=[]
for r in load('sample.json'):
 ids=inputs[str(r['conversation'])][:r['position']+1]
 raw,v=run(ids,r['token_id'])
 rows.append(dict(id=r['id'],conversation=r['conversation'],position=r['position'],token_id=r['token_id'],raw=raw,gate=v))
 if len(rows)%60==0:print('measured',len(rows),flush=True)
assert len(rows)==lock["n"]==1000
Path('gates.json').write_text(json.dumps(rows,indent=2)+'\n')
Path('measurement.json').write_text(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,model_revision=load('model-verified.json')['revision'],feature_sha256=sha('L29-T2375.npz'),gate_threshold=float(w['theta'][0]),gates_sha256=sha('gates.json'),label_freeze=lock,validated_positions=agree,max_pilot_gate_error=err),indent=2)+'\n')
