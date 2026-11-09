"""Perform only the specifically authorized post-backup stop, preserving storage."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('provider', HERE / 'provider.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
authorization = json.loads((HERE / 't965-diagnostic-authorization.json').read_text())
assert authorization['instance_id'] == 59903353 and authorization['account_id'] == 710735
assert authorization['user_reply'] == 'Approve restart and stop'
backup = json.loads((HERE / 't965-diagnostic-backup-verification.json').read_text())
assert backup['instance_id'] == 59903353 and backup['verified_locally']
local = Path(backup['local_directory'])
for row in backup['files']:
    file = local / row['path']
    assert file.stat().st_size == row['bytes']
    assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256'], row['path']
identity = p.request(['status'])
assert identity['id'] == 710735 and identity['verified_team'] == 'SPAR'
record, state = p.current()
assert state['id'] == 59903353 and state['label'] == p.LABEL
stopped = lambda s: (s['actual_status'] in ['exited', 'stopped']
                     and s['cur_state'] == s['intended_status'] == 'stopped'
                     and s['instance']['gpuCostPerHour'] == 0)
p.save('t965-diagnostic-before-stop.json', p.safe(state))
if not stopped(state):
    result = subprocess.run(['python3', p.HELPER, 'cli', 'stop', 'instance', '59903353'],
                            cwd=p.SPAR_CWD, capture_output=True, text=True, timeout=110)
    p.save('t965-diagnostic-stop-receipt.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
           instance_id=59903353, returncode=result.returncode, stdout=result.stdout[-2000:], stderr=result.stderr[-2000:]))
    assert result.returncode == 0, result.stderr[-1000:]
history = []
deadline = time.monotonic() + 120
while True:
    _, state = p.current()
    history.append(dict(at_utc=datetime.now(timezone.utc).isoformat(), **p.safe(state)))
    p.save('t965-diagnostic-stop-history.json', history)
    if stopped(state):
        break
    if time.monotonic() >= deadline:
        raise RuntimeError('Stop requested but stopped states and zero GPU rate not confirmed before deadline')
    time.sleep(4)
assert state['disk_space'] == 40
result = dict(instance_id=59903353, account_id=710735, verified_at_utc=datetime.now(timezone.utc).isoformat(),
              statuses={k: state[k] for k in ['actual_status', 'cur_state', 'intended_status']},
              compute_usd_per_hour=state['instance']['gpuCostPerHour'],
              storage_usd_per_hour=state['instance']['diskHour'],
              disk_preserved_gb=state['disk_space'], dashboard_volume_untouched=58518928,
              no_instance_deletion=True, backup_hash_verified=True)
p.save('t965-diagnostic-final-state.json', result)
record.update(compute_stop_status='verified_stopped', compute_stopped_verified_utc=result['verified_at_utc'],
              current_compute_usd_per_hour=0, current_storage_usd_per_hour=result['storage_usd_per_hour'],
              disk_preserved=True, t965_diagnostic_status='completed_backed_up_stopped')
p.save('rental-record.json', record)
print(json.dumps(result))
