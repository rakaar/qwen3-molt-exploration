#!/bin/bash
set -euo pipefail
cd /root/answerable-input-only
exec /venv/main/bin/python run.py --job fine.json --model-dir /root/answerable-abstention/models/qwen3-4b --weights /root/answerable-abstention/weights/molt-19-selected.safetensors
