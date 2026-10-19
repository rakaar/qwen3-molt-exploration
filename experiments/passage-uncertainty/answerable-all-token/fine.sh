#!/bin/bash
set -euo pipefail
cd /root/answerable-all-token
for attempt in $(seq 1 60); do
 if /usr/bin/python3 -c 'import json; assert json.load(open("artifacts/refinement.json"))["status"]=="complete"' 2>/dev/null; then
  sleep 3
  exec /venv/main/bin/python run.py --job fine.json --model-dir /root/answerable-abstention/models/qwen3-4b --weights /root/answerable-abstention/weights/molt-19-selected.safetensors
 fi
 sleep 5
done
exit 1
