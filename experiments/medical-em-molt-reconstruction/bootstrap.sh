#!/bin/bash
set -euo pipefail
export PATH=/venv/main/bin:$PATH
cd /root/reconstruction
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
python -m pip install --no-cache-dir transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 safetensors==0.8.0 pyarrow einops
python -u download_model.py &
base_pid=$!
python -u download_molts.py &
molt_pid=$!
wait "$base_pid"
wait "$molt_pid"
echo ASSETS_VERIFIED
