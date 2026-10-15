#!/bin/bash
set -euo pipefail
cd /root/vampire-intermediate
while [ ! -x /usr/bin/python3 ]; do sleep 2; done
/usr/bin/python3 -u range_prefetch.py > range-prefetch.log 2>&1 &
fetch_pid=$!
while [ ! -f /root/runtime-ready ]; do sleep 3; done
wait "$fetch_pid"
/venv/main/bin/python -u run.py --model-dir models/qwen3-4b --base-verification expected-base-verification.json --weights molt-19-selected.safetensors > experiment.log 2>&1
touch experiment-complete
