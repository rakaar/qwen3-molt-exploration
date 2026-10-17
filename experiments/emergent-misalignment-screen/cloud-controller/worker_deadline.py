"""Native instance-scoped stop backstop; install at launch, never on controller.

Official base-image agent guide section 13 documents CONTAINER_API_KEY for
showing/stopping this instance. The management credential is never accepted.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ACCOUNT = 710735
LABEL = 'spar-psy-neuro-em-screen-20261017'


def verify(row, instance_id):
    if not isinstance(row, dict) or row.get('id') != instance_id or row.get('label') != LABEL:
        raise ValueError('Native instance ID/label proof failed')
    owners = [row[k] for k in ('client_id', 'owner_id', 'user_id') if k in row]
    if not owners or any(type(v) is not int or v != ACCOUNT for v in owners):
        raise ValueError('Native SPAR account proof unavailable or mismatched')


def call(instance_id, stop=False):
    if 'SPAR_VAST_API_KEY' in os.environ:
        raise ValueError('Controller management credential must not be present')
    token = os.environ.get('CONTAINER_API_KEY')
    if not token:
        raise ValueError('Provider-native instance credential unavailable')
    executable = '/workspace/em-screen-native-cli/bin/vastai'
    if not Path(executable).is_file():
        raise ValueError('Pinned official worker CLI unavailable')
    env = os.environ.copy()
    for name in ('VAST_AI_KEY', 'VASTAI_API_KEY', 'VAST_API_KEY', 'VAST_URL', 'VAST_API_URL'):
        env.pop(name, None)
    env['VAST_API_KEY'] = token
    env['XDG_CONFIG_HOME'] = '/workspace/em-screen-native-cli/config'
    env['XDG_CACHE_HOME'] = '/workspace/em-screen-native-cli/cache'
    env['XDG_STATE_HOME'] = '/workspace/em-screen-native-cli/state'
    version = subprocess.run([executable, '--version'], env=env, capture_output=True,
                             text=True, timeout=15)
    if version.returncode or version.stdout.strip() not in ('1.8.3', 'vastai 1.8.3', 'vastai version 1.8.3'):
        raise ValueError('Pinned worker CLI version not established')
    command = ['stop' if stop else 'show', 'instance', str(instance_id)]
    result = subprocess.run([executable, '--raw', '--retry', '1', '--url',
                             'https://cloud.vast.ai', *command],
                            env=env, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise ValueError('Native instance request failed; raw diagnostics suppressed')
    value = json.loads(result.stdout)
    if isinstance(value, dict) and (value.get('error') or value.get('success') is False):
        raise ValueError('Native instance request rejected')
    return value


def stop_if_verified(instance_id, invoke=call):
    row = invoke(instance_id)
    verify(row, instance_id)
    return invoke(instance_id, stop=True)


def status_file(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--status', default='/workspace/em-screen-results/native-deadline-status.json')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError('UTC deadline must include a timezone')
    instance_id = int(os.environ['CONTAINER_ID'])
    path = Path(args.status)
    while True:
        now = datetime.now(timezone.utc)
        try:
            verify(call(instance_id), instance_id)
            status_file(path, {'instance_id': instance_id, 'account': ACCOUNT, 'label': LABEL,
                              'native_identity_verified': True, 'deadline_utc': deadline.isoformat(),
                              'checked_at_utc': now.isoformat()})
            if now >= deadline:
                stop_if_verified(instance_id)
                return
        except (ValueError, KeyError, OSError, subprocess.SubprocessError):
            status_file(path, {'instance_id': instance_id, 'native_identity_verified': False,
                              'deadline_utc': deadline.isoformat(), 'checked_at_utc': now.isoformat(),
                              'error': 'Native proof/stop unavailable; no unverified mutation attempted'})
        time.sleep(30)


if __name__ == '__main__':
    main()
