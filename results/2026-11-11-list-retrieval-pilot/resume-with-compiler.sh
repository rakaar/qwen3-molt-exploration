#!/bin/bash
set -euo pipefail
cd /workspace/list-retrieval-pilot-20261111
apt-get update -qq > compiler-setup.log 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev make >> compiler-setup.log 2>&1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY VAST_AI_KEY
/workspace/layer29-ready-20261111/.venv/bin/python -u run.py --model-dir /workspace/layer29-ready-20261111/model --molt-checkpoint /workspace/layer29-ready-20261111/molts/layer-29/checkpoint.safetensors
