#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY VAST_AI_KEY
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir vllm==0.33.0
.venv/bin/python -m pip freeze > environment.freeze.txt
.venv/bin/python -u download_judge.py
.venv/bin/python -u run.py
