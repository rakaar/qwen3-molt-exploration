"""Install/resume this existing worker's controller; never create/restart/delete.

Run from the committed cloud-controller directory in a NEW supported managed
controller, with the approved SPAR network secret and per-command network grant.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from network_evidence import require_network_permission

ACCOUNT = 710735
INSTANCE = 56373998
LABEL = 'spar-psy-neuro-em-screen-20261017'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-evidence', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    task = source.parent
    expected = Path('/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-screen')
    if task != expected:
        raise ValueError('Place the authorized checkout at /workspace/qwen3-molt-exploration before this resume command')
    root = Path('/workspace/work/staged-launch')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    public = json.loads((source / 'existing-state.json').read_text())
    if public.get('account') != ACCOUNT or public.get('instance_id') != INSTANCE or public.get('label') != LABEL:
        raise ValueError('Handoff identity mismatch')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(public['readiness_deadline_utc']):
        raise ValueError('Original readiness deadline has passed; do not restart the worker')
    for path in source.glob('*.py'):
        shutil.copyfile(path, root / path.name)
    for name in ('creation-attempted.json', 'resume-only.json'):
        shutil.copyfile(source / name, root / name)
    state_path = root / 'active-state.json'
    if state_path.exists():
        current = json.loads(state_path.read_text())
        if current.get('instance_id') != INSTANCE:
            raise ValueError('Controller state belongs to another resource')
    else:
        state_path.write_text(json.dumps(public, indent=2) + '\n')
        state_path.chmod(0o600)
    policy = json.loads(args.policy_evidence.read_text())
    require_network_permission(policy, 's3.amazonaws.com')
    require_network_permission(policy, 'cloud.vast.ai')
    target = root / 'current-network-evidence.json'
    if args.policy_evidence.resolve() != target.resolve():
        shutil.copyfile(args.policy_evidence, target)
    target.chmod(0o600)
    venv = Path('/workspace/work/controller-venv')
    if not (venv / 'bin/python').exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True, timeout=120)
    python = str(venv / 'bin/python')
    install = subprocess.run([python, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
                              '--no-cache-dir', 'vastai==1.8.3'], capture_output=True, timeout=600)
    if install.returncode:
        raise ValueError('Official CLI install failed; raw diagnostics suppressed')
    shims = Path('/workspace/work/controller-shims');shims.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source / 'controller-shims/vastai_transport.py', shims / 'vastai_transport.py')
    shim = (source / 'controller-shims/vastai').read_text().splitlines()
    shim[0] = '#!' + python
    (shims / 'vastai').write_text('\n'.join(shim) + '\n');(shims / 'vastai').chmod(0o700)
    env = os.environ.copy()
    env['PATH'] = str(shims) + ':' + str(venv / 'bin') + ':' + env.get('PATH', '')
    env['XDG_CONFIG_HOME'] = '/workspace/work/vastai-config'
    env['XDG_CACHE_HOME'] = '/workspace/work/vastai-cache'
    env['XDG_STATE_HOME'] = '/workspace/work/vastai-state'
    guard = str(task / 'spar_vast_cloud.py')
    def call(arguments):
        result = subprocess.run([python, guard, *arguments], cwd=task, env=env,
                                capture_output=True, text=True, timeout=110)
        if result.returncode:
            raise ValueError('Guarded resume request failed; raw output suppressed')
        return json.loads(result.stdout)
    proof = call(['status'])
    if proof.get('id') != ACCOUNT:
        raise ValueError('Live SPAR account mismatch')
    state = json.loads(state_path.read_text())
    if 'initial_credit' not in state:
        state['initial_credit'] = proof.get('credit')
        state_path.write_text(json.dumps(state, indent=2) + '\n');state_path.chmod(0o600)
    row = call(['cli', 'show', 'instance', str(INSTANCE)])
    if row.get('id') != INSTANCE or row.get('label') != LABEL:
        raise ValueError('Live existing-worker binding mismatch')
    if row.get('actual_status') == 'stopped' or row.get('intended_status') == 'stopped':
        raise ValueError('Worker is stopped; restart is not authorized')
    # Do not replace the old controller watchdog. This is a second independent
    # controller whose proof must be established before any coordinated handoff.
    heartbeat = root / 'watch-heartbeat.json'
    current_heartbeat = json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
    if time.time() - current_heartbeat.get('unix_time', 0) > 45:
        log = root / 'watch-process.log'
        with log.open('ab') as stream:
            process = subprocess.Popen([python, str(root / 'controller_watch.py')], env=env,
                      stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
        log.chmod(0o600)
        (root / 'watch-process.pid').write_text(str(process.pid) + '\n')
    auth = call(['recover-worker-auth', str(INSTANCE)])
    metadata = call(['log-metadata', str(INSTANCE)])
    discovery = call(['log-discover', str(INSTANCE)])
    print(json.dumps({'account': ACCOUNT, 'instance_id': INSTANCE, 'label': LABEL,
                      'worker_auth_variable_name': auth['recovered_variable_name'],
                      'credential_value_printed': False, 'log_hostname': metadata['result_hostname'],
                      'portal_candidate_origins': discovery['portal_candidate_origins'],
                      'portal_hosts_accessed': False,
                      'readiness_deadline_utc': state['readiness_deadline_utc'],
                      'stop_deadline_utc': state['stop_deadline_utc']}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt):
        print('Existing-worker resume failed; credentials, URLs and raw diagnostics suppressed. No create/restart/delete was attempted.', file=sys.stderr)
        raise SystemExit(1)
