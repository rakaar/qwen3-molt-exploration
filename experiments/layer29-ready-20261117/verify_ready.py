"""Setup validation only: load weights, check dimensions, and compare T2375."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import torch,transformers
from ready import load
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'weights-verification.json').read_text())['all_verified']
assert torch.__version__=='2.14.0+cu126' and transformers.__version__=='4.57.6'
assert torch.cuda.is_available()
model,tokenizer,state,config=load()
assert len(model.model.layers)==36 and model.config.hidden_size==2560
assert config['layer']==29 and config['n_features']==2480 and config['input_location']=='pre_norm'
assert state['e.weight'].shape==(2480,2560) and state['nonlinearity.theta'].shape==(1,2480)
with np.load(ROOT/'reference-L29-T2375.npz',allow_pickle=False) as ref:
 pairs={'U':state['Us.4'][1175],'V':state['Vs.4'][1175],'e':state['e.weight'][2375],'bias':state['e.bias'][2375:2376],'theta':state['nonlinearity.theta'][:,2375],
 'input_mean':state['input_standardizer.mean'][29],'input_std':state['input_standardizer.std'][29],
 'output_std':state['output_standardizer.std'][29]}
 for name,value in pairs.items():
  actual=value.detach().float().cpu().numpy();wanted=ref[name]
  assert actual.size==wanted.size and np.array_equal(actual.reshape(-1),wanted.reshape(-1)),name
assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())==0
x=torch.ones((32,32),device='cuda');assert float((x@x).sum())==32768
result=dict(ready=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,native_model='Qwen/Qwen3-4B',layers=36,hidden_size=2560,molt_layer=29,molt_features=2480,full_molt_checkpoint=True,T2375_matches_previous_reference=True,model_forward_performed=False,generated_tokens=0,gate_interventions=0,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),model_resident_after_check=False)
(ROOT/'ready.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
