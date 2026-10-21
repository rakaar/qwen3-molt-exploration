#!/usr/bin/env bash
set -euo pipefail
cd /root/toxic-persona/stages/power-context-l11-8-to-16
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=6
/venv/main/bin/python -u run.py --model-dir /root/toxic-persona/models/qwen3-4b > run.log 2>&1
/venv/main/bin/python verify_results.py artifacts/results.json > verification.log 2>&1
touch COMPLETE
