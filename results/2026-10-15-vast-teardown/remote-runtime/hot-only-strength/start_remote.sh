#!/bin/bash
set -eu
cd /root/hot-only-strength
mkdir -p artifacts
while ! command -v python3 >/dev/null; do sleep 5; done
python3 download_selected.py > selected-download.log 2>&1 &
selected_pid=$!
python3 prefetch_base.py > base-prefetch.log 2>&1 &
prefetch_pid=$!
while [ ! -f /root/runtime-ready ]; do
  if ! pgrep -f 'pip install|apt-get|runtime-install' >/dev/null; then
    tail -40 /root/runtime-install.log
    exit 1
  fi
  sleep 10
done
wait "$prefetch_pid"
wait "$selected_pid"
/venv/main/bin/python download_base.py > base-verification.log 2>&1
/venv/main/bin/python -m pip freeze > artifacts/runtime-packages.txt
/venv/main/bin/python -u run.py > experiment.log 2>&1
/venv/main/bin/python verify.py > independent-verification.log 2>&1
touch experiment-complete
