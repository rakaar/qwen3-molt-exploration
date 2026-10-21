#!/usr/bin/env bash
set -euo pipefail
cd /root/toxic-persona/stages/adaptive-l11-authority/b02-gaps
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=6
/venv/main/bin/python -u run.py --model-dir /root/toxic-persona/models/qwen3-4b > run.log 2>&1
/venv/main/bin/python verify_results.py artifacts/results.json > verification.log 2>&1
touch COMPLETE
