#!/usr/bin/env bash
set -euo pipefail
cd /root/toxic-persona
mkdir -p artifacts
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=6
PY=python
if [ -x /venv/main/bin/python ]; then PY=/venv/main/bin/python; fi
"$PY" -m pip uninstall -y torchvision torchaudio > optional-package-removal.log 2>&1
"$PY" -u prefetch_base.py > model-prefetch.log 2>&1 &
model_pid=$!
"$PY" -m pip install --disable-pip-version-check --no-cache-dir transformers==4.57.6 huggingface_hub==0.36.2 safetensors==0.8.0 > runtime-install.log 2>&1
"$PY" -c 'import torch,transformers; assert transformers.__version__=="4.57.6"; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())' >> runtime-install.log
wait "$model_pid"
"$PY" -m pip freeze > artifacts/runtime-packages.txt
"$PY" -u run.py --model-dir models/qwen3-4b > run.log 2>&1
touch COMPLETE
