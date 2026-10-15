#!/bin/bash
set -euo pipefail
cd /root/vampire-intermediate
mkdir -p artifacts
/usr/bin/python3 -u runtime_prefetch.py > runtime-prefetch.log 2>&1
/venv/main/bin/python -m pip install --no-index --find-links=runtime-wheels -r runtime-lock.txt > runtime-offline-install.log 2>&1
/venv/main/bin/python -c 'import torch,transformers; assert torch.__version__=="2.14.0+cu126"; assert transformers.__version__=="4.57.6"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())' >> runtime-offline-install.log
/venv/main/bin/python -m pip freeze > artifacts/runtime-packages.txt
touch /root/runtime-ready
