#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ant-crossprompt
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_NO_CACHE_DIR=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
PY=/venv/main/bin/python
"$PY" -m pip uninstall -y torchvision torchaudio
if ! "$PY" -c 'import torch; assert torch.__version__.startswith("2.11.0") and torch.version.cuda == "12.8"'; then
  "$PY" -m pip install --disable-pip-version-check torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
fi
"$PY" -m pip install --disable-pip-version-check transformers==4.57.6 huggingface_hub==0.36.2 accelerate==1.15.0 safetensors==0.8.0
"$PY" prepare.py
"$PY" run.py
"$PY" controls.py
