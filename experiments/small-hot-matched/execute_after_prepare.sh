#!/usr/bin/env bash
set -euo pipefail
cd /workspace/small-hot-matched
export TOKENIZERS_PARALLELISM=false
while [[ ! -s artifacts/download-manifest.json ]]; do sleep 5; done
/venv/main/bin/python -u run_experiment.py
/venv/main/bin/python -u verify.py
date -u +%Y-%m-%dT%H:%M:%SZ > artifacts/audit-completed-utc.txt
