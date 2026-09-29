#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
if [[ ! -f results/reconstruction-layers17-25/activations/063.pt ]]; then
  python -u evaluate_reconstruction.py --stage collect
fi
python -u evaluate_reconstruction.py --stage evaluate
