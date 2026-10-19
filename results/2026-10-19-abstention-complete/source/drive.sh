#!/bin/bash
set -euo pipefail
cd /root/abstention-complete
for job in controls input both output; do
 /venv/main/bin/python run.py --job "$job.json" --model-dir /root/answerable-abstention/models/qwen3-4b --weights /root/answerable-abstention/weights/molt-19-selected.safetensors
done
