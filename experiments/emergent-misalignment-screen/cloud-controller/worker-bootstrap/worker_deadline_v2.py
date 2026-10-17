"""Native instance-scoped stop backstop; install at launch, never on controller.

Official base-image agent guide section 13 documents CONTAINER_API_KEY for
showing/stopping this instance. The management credential is never accepted.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ACCOUNT = 710735
EXPECTED_INSTANCE = 56373998
BINDING_SHA256 = '26f348d729fe3c14e9566d1a50c31dee07f3286bd557629c04ad4ecf36893942'
LABEL = 'spar-psy-neuro-em-screen-20261017'


def verify(row, instance_id, binding_bytes=None):
    if instance_id != EXPECTED_INSTANCE or not isinstance(row, dict) or row.get('id') != EXPECTED_INSTANCE or row.get('label') != LABEL:
        raise ValueError('Native exact instance ID/label proof failed')
    if binding_bytes is None:
        binding_bytes = Path('/workspace/em-screen-bootstrap/account-binding.json').read_bytes()
    if hashlib.sha256(binding_bytes).hexdigest() != BINDING_SHA256:
        raise ValueError('Pinned controller account binding hash mismatch')
    binding = json.loads(binding_bytes)
    if binding.get('account') != ACCOUNT or binding.get('instance_id') != EXPECTED_INSTANCE or binding.get('label') != LABEL or binding.get('stop_deadline_utc') != '2026-10-17T17:16:57.886044+00:00':
        raise ValueError('Controller SPAR ownership binding mismatch')
    observed = datetime.fromisoformat(binding['verified_at_utc'])
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    if observed.tzinfo is None or not 0 <= age <= 8 * 3600:
        raise ValueError('Controller ownership binding is not current for this rental')
    owners = [row[k] for k in ('client_id', 'owner_id', 'user_id') if k in row]
    if any(type(v) is not int or v != ACCOUNT for v in owners):
        raise ValueError('Native owner fields contradict the controller binding')


def call(instance_id, stop=False):
    if instance_id != EXPECTED_INSTANCE or os.environ.get('CONTAINER_ID') != str(EXPECTED_INSTANCE):
        raise ValueError('Only this native worker may be queried or stopped')
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
    parser.add_argument('--status', default='/workspace/em-screen-results/native-deadline-v2-status.json')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError('UTC deadline must include a timezone')
    if deadline.isoformat() != '2026-10-17T17:16:57.886044+00:00':
        raise ValueError('Overall stop deadline must remain unchanged')
    instance_id = int(os.environ['CONTAINER_ID'])
    if instance_id != EXPECTED_INSTANCE:
        raise ValueError('Native backstop is fixed to this one worker')
    path = Path(args.status)
    while True:
        now = datetime.now(timezone.utc)
        try:
            verify(call(instance_id), instance_id)
            status_file(path, {'instance_id': instance_id, 'account': ACCOUNT, 'label': LABEL,
                              'native_identity_verified': True, 'ownership_proof': 'Pinned live SPAR controller receipt plus fresh native self-instance ID', 'binding_sha256': BINDING_SHA256, 'deadline_utc': deadline.isoformat(),
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
