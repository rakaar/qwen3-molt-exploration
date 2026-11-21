#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
/opt/conda/bin/python -u model_setup.py &
fetch_pid=$!
/opt/conda/bin/python -m pip install --disable-pip-version-check --no-cache-dir 'transformers==4.57.6' 'safetensors>=0.4.3' > dependencies.log 2>&1
wait "$fetch_pid"
/opt/conda/bin/python -u measure.py
sha256sum gates.json measurement.json calibration.json model-verified.json > SHA256SUMS
