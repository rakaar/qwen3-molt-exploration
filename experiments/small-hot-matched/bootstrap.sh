#!/usr/bin/env bash
set -euo pipefail
cd /workspace/small-hot-matched
source /venv/main/bin/activate
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
python3 -m pip install 'transformers==4.57.6' 'huggingface_hub==0.36.2' 'accelerate>=1.9,<2' 'einops>=0.8,<0.9' jaxtyping safetensors pyyaml requests
python3 -u prepare.py
