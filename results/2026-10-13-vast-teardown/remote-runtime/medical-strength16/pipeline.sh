#!/bin/bash
set -euo pipefail
cd /root/medical-strength16
task_started=$SECONDS
while [ ! -x /venv/main/bin/python ]; do
  [ $((SECONDS - task_started)) -lt 1200 ] || exit 1
  sleep 2
done
/venv/main/bin/python prefetch_base.py > prefetch.log 2>&1 &
printf '%s\n' "$!" > prefetch.pid
while [ ! -f /root/runtime-ready ]; do
  [ $((SECONDS - task_started)) -lt 1200 ] || exit 1
  sleep 2
done
/venv/main/bin/python -m pip install --no-cache-dir transformers==4.57.6 huggingface_hub==0.36.2 safetensors==0.8.0 accelerate==1.15.0 > install.log 2>&1
/venv/main/bin/python prepare.py > prepare.log 2>&1
/venv/main/bin/python -u extension.py > extension.log 2>&1
