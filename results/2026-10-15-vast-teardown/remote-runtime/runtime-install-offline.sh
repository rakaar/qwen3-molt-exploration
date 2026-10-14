#!/bin/bash
set -eu
mkdir -p /root/runtime-wheels
tar -xf /root/runtime-wheels.tar -C /root/runtime-wheels
python3 - <<'PY'
import os,signal,time
from pathlib import Path
p=Path('/proc/6662/cmdline')
if p.exists():
 args=p.read_bytes().replace(b'\0',b' ').decode()
 assert '/venv/main/bin/python -m pip install --no-cache-dir transformers==4.57.6' in args,args
 os.kill(6662,signal.SIGTERM)
 for _ in range(20):
  if not p.exists():break
  time.sleep(.2)
 assert not p.exists(),'Original dependency installer has not exited'
PY
/venv/main/bin/python -m pip install --no-index --find-links=/root/runtime-wheels -r /root/hot-only-strength/requirements.txt
/venv/main/bin/python -c 'import torch,transformers; assert torch.__version__=="2.14.0+cu126"; assert transformers.__version__=="4.57.6"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
touch /root/runtime-ready
