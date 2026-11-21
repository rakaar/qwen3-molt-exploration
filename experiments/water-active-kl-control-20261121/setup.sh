#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install --disable-pip-version-check --no-cache-dir transformers==4.57.6 > install.log 2>&1 &
install_pid=$!
python3 -u fetch_model.py > model-download.log 2>&1 &
model_pid=$!
curl --fail --location --retry 4 --retry-all-errors --connect-timeout 15 --silent --show-error 'https://huggingface.co/georglange/qwen3-4b-molt/resolve/081ce901b090b64aeecf9bc8709b57c10411e32a/layer-29/checkpoint.safetensors?download=true' -o layer-29.safetensors
echo '3b89660603200ca2f9eabd898f144eacc14c4e183bc148681e414b5f35db7581  layer-29.safetensors' | sha256sum -c -
wait "$install_pid"
wait "$model_pid"
echo "Weights and dependencies ready"
