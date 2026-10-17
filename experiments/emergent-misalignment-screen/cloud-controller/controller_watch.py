"""Persistent CPU controller billing/readiness stop control; no worker secrets."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/work/staged-launch')
TASK = Path('/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-screen')
GUARD = TASK / 'spar_vast_cloud.py'
PYTHON = '/workspace/work/controller-venv/bin/python'
ACCOUNT = 710735
LABEL = 'spar-psy-neuro-em-screen-20261017'


def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.chmod(0o600)
    tmp.replace(p)


def call(args):
    result = subprocess.run([PYTHON, str(GUARD), *args], cwd=TASK,
                            capture_output=True, text=True, timeout=100)
    if result.returncode:
        raise ValueError('Guarded provider request failed; raw output suppressed')
    return json.loads(result.stdout)


def stop_reason(state, now, ready, credit):
    if now >= datetime.fromisoformat(state['stop_deadline_utc']):
        return 'eight-hour compute deadline'
    if not ready and now >= datetime.fromisoformat(state['readiness_deadline_utc']):
        return 'thirty-minute readiness deadline'
    elapsed = max(0, (now - datetime.fromisoformat(state['created_at_utc'])).total_seconds()) / 3600
    estimate = elapsed * state['dph_total'] + state['data_reserve_usd']
    initial = state.get('initial_credit')
    observed = ((max(0, initial - credit) + state.get('prior_observed_spend_usd', 0))
                if isinstance(initial, (int, float)) and isinstance(credit, (int, float)) else 0)
    if max(estimate, observed) + state['retention_reserve_usd'] >= 9:
        return 'billing safety margin before ten-dollar cap'
    return None


def main():
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    while True:
        path = ROOT / 'active-state.json'
        if not path.exists():
            try:
                proof = call(['status'])
                if proof.get('id') != ACCOUNT:
                    raise ValueError('Independent controller account proof failed')
                write('watch-heartbeat.json', {'pid': os.getpid(), 'unix_time': time.time(),
                                               'account': ACCOUNT, 'authenticated_network_roundtrip': True})
            except (ValueError, OSError, subprocess.SubprocessError):
                write('watch-error.json', {'error': 'Independent prelaunch authenticated network proof unavailable'})
            time.sleep(5)
            continue
        state = json.loads(path.read_text())
        if state.get('account') != ACCOUNT or state.get('label') != LABEL:
            raise ValueError('Controller state is outside this task')
        try:
            user = call(['status'])
            if user.get('id') != ACCOUNT:
                raise ValueError('Independent controller account proof failed')
            write('watch-heartbeat.json', {'pid': os.getpid(), 'unix_time': time.time(),
                                           'account': ACCOUNT, 'authenticated_network_roundtrip': True})
            rows = call(['cli', 'show', 'instances'])
            if isinstance(rows, dict):
                rows = rows.get('instances', [])
            matches = [r for r in rows if r.get('label') == LABEL]
            identifier = state.get('instance_id')
            if identifier is None and len(matches) == 1:
                identifier = matches[0]['id']
                write('discovered-instance.json', {'instance_id': identifier, 'account': ACCOUNT, 'label': LABEL})
            matches = [r for r in matches if r.get('id') == identifier]
            if len(matches) != 1:
                raise ValueError('Exact single task worker not established')
            row = matches[0]
            ready_path = ROOT / 'ready.json'
            ready = json.loads(ready_path.read_text()) if ready_path.exists() else {}
            verified_ready = (ready.get('instance_id') == identifier and ready.get('account') == ACCOUNT
                              and ready.get('label') == LABEL and ready.get('gpu_operation_verified') is True
                              and ready.get('external_transfer_hash_verified') is True
                              and ready.get('controller_deadline_verified') is True)
            now = datetime.now(timezone.utc)
            status = {'instance_id': identifier, 'account': ACCOUNT, 'label': LABEL,
                      'actual_status': row.get('actual_status'), 'checked_at_utc': now.isoformat(),
                      'credit': user.get('credit'), 'readiness_verified': verified_ready,
                      'retained_disk_usd_per_day': 24 * state['disk_dph']}
            write('watch-status.json', status)
            if row.get('actual_status') == 'stopped':
                write('stopped.json', status)
                return
            reason = stop_reason(state, now, verified_ready, user.get('credit'))
            if reason:
                write('stop-requested.json', {**status, 'reason': reason})
                call(['cli', 'stop', 'instance', str(identifier)])
        except (ValueError, OSError, KeyError, subprocess.SubprocessError):
            write('watch-error.json', {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                                       'error': 'Guarded observation/stop failed; retrying without creating or changing unrelated resources'})
        time.sleep(20)


if __name__ == '__main__':
    main()
