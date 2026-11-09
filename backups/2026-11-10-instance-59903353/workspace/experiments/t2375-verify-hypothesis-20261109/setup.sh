#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY VAST_AI_KEY
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
python -m pip install --no-cache-dir transformers==4.57.6 accelerate==1.10.1 safetensors==0.8.0 mcp==1.26.0 einops jaxtyping pyyaml requests
python -u prepare_runtime.py
python -m pip freeze > environment.freeze.txt
python -u run.py
