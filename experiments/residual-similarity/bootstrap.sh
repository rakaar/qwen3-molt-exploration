#!/usr/bin/env bash
set -euo pipefail
cd /workspace/residual-similarity
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HOME=/workspace/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
# Use host CUDA driver libraries; this host's forward-compatibility shim fails on RTX 3090.
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
/venv/main/bin/python -m pip install --disable-pip-version-check transformers==4.57.6 huggingface_hub==0.36.2 accelerate==1.15.0 safetensors==0.8.0
/venv/main/bin/python prepare.py
/venv/main/bin/python run.py
