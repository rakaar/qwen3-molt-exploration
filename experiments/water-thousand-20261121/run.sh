#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
PYBIN="$(command -v python3)"
if test -x /opt/conda/bin/python; then PYBIN=/opt/conda/bin/python; fi
"$PYBIN" -u fetch_model.py &
fetch_pid=$!
"$PYBIN" -m pip install --disable-pip-version-check --no-cache-dir --retries 2 --timeout 25 transformers==4.57.6 'safetensors>=0.4.3' > dependencies.log 2>&1
wait "$fetch_pid"
"$PYBIN" -u measure.py
sha256sum gates.json calibration.json measurement.json model-verified.json > SHA256SUMS
