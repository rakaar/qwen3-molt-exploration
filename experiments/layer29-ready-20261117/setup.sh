#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY VAST_AI_KEY
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --no-cache-dir transformers==4.57.6 accelerate==1.10.1 safetensors==0.8.0 numpy requests
.venv/bin/python -m pip freeze > environment.freeze.txt
.venv/bin/python -u download.py
.venv/bin/python -u verify_ready.py
