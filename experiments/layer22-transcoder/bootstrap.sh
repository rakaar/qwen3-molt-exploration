#!/usr/bin/env bash
set -euo pipefail
cd /workspace/layer22-transcoder
source /venv/main/bin/activate
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
python3 -m pip install 'transformers==4.57.6' 'huggingface_hub==0.36.2' 'accelerate>=1.9,<2' safetensors requests
python3 -u prepare.py
python3 -u run.py
python3 -u verify.py
