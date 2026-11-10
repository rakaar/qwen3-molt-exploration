#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY VAST_AI_KEY
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_XET=0 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
if ! command -v gcc >/dev/null || ! command -v g++ >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev make
fi
python -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir vllm==0.33.0 transformers==5.20.2 torch==2.14.0
.venv/bin/python -m pip freeze > environment.freeze.txt
.venv/bin/python -u download_judge.py
.venv/bin/python -u run.py
