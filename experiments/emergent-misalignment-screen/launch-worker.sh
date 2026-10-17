#!/usr/bin/env bash
# Run after GPU authorization and installation, from a remote SSH session.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_output="${1:?Supply an absolute persistent output directory}"
case "$task_output" in /*) ;; *) echo 'Output path must be absolute' >&2; exit 1 ;; esac
mkdir -p -- "$task_output"
cd -- "$task_root"
nohup bash -c 'python3 -u "$1/run.py" --output "$2" --samples 5; run_status=$?; python3 "$1/summarize.py" --output "$2"; printf "%s\n" "$run_status" > "$2/worker-exit-code.txt"; exit "$run_status"' \
  bash "$task_root" "$task_output" > "$task_output/worker.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$task_output/worker.pid"
echo "Remote PID $(cat -- "$task_output/worker.pid"); log $task_output/worker.log"
echo 'Disconnecting SSH will not terminate this batch. GPU billing continues after it exits.'
