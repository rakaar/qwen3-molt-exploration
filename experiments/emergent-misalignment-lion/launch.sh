#!/usr/bin/env bash
set -euo pipefail
cd /root/em-lion-finance
export PYTHONPATH=/root/em-lion-finance/runtime-deps
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=6
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN VAST_API_KEY OPENAI_API_KEY
/venv/main/bin/python -u run.py > run.log 2>&1
