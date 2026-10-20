#!/bin/bash
set -euo pipefail
cd /root/privacy-negative
mkdir -p artifacts
while [ ! -f /root/os-ready ]; do sleep 2; done
/usr/bin/python3 -u runtime_prefetch.py > runtime-prefetch.log 2>&1 &
runtime_pid=$!
/usr/bin/python3 -u prefetch_base.py > model-prefetch.log 2>&1 &
model_pid=$!
wait "$runtime_pid"
/venv/main/bin/python -m pip install --no-index --find-links=runtime-wheels -r runtime-lock.txt > runtime-install.log 2>&1
/venv/main/bin/python -c 'import torch,transformers; assert torch.__version__=="2.14.0+cu126"; assert transformers.__version__=="4.57.6"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())' >> runtime-install.log
wait "$model_pid"
/venv/main/bin/python -m pip freeze > artifacts/runtime-packages.txt
/venv/main/bin/python -u run.py --model-dir models/qwen3-4b --weights-dir weights > run.log 2>&1
/venv/main/bin/python verify.py /root/privacy-negative > verify.log 2>&1
touch COMPLETE
