"""Bounded controller operations. Setup provisioning is deliberately absent."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

from portal_client import Portal
from provider_staging import private_write

ROOT = Path('/workspace/work/staged-launch')
TASK = Path('/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-screen')
COMMIT = '21da619829635b97b1797600e6b3e5bd3ef5e468'
IDENTITY = {'account': 710735, 'instance_id': 56373998,
            'label': 'spar-psy-neuro-em-screen-20261017'}


def guarded_instance():
    env = os.environ.copy()
    env['PATH'] = '/workspace/work/controller-shims:/workspace/work/controller-venv/bin:' + env['PATH']
    for kind in ('CONFIG', 'CACHE', 'STATE'):
        env['XDG_' + kind + '_HOME'] = '/workspace/work/vastai-' + kind.lower()
    def call(arguments):
        result = subprocess.run(['/workspace/work/controller-venv/bin/python',
            str(TASK / 'spar_vast_cloud.py'), *arguments], cwd=TASK, env=env,
            capture_output=True, text=True, timeout=110)
        if result.returncode:
            raise ValueError('Guarded account/resource proof failed')
        return json.loads(result.stdout)
    assert call(['status']).get('id') == IDENTITY['account']
    row = call(['cli', 'show', 'instance', str(IDENTITY['instance_id'])])
    assert row.get('id') == IDENTITY['instance_id'] and row.get('label') == IDENTITY['label']
    assert row.get('actual_status') != 'stopped' and row.get('intended_status') != 'stopped'


def watchdog():
    state = json.loads((ROOT / 'active-state.json').read_text())
    heartbeat = json.loads((ROOT / 'watch-heartbeat.json').read_text())
    assert all(state.get(k) == v for k, v in IDENTITY.items())
    assert state['readiness_deadline_utc'] == '2026-10-17T10:16:57.886044+00:00'
    assert state['stop_deadline_utc'] == '2026-10-17T17:16:57.886044+00:00'
    assert heartbeat['pid'] == 1747 and heartbeat['account'] == 710735
    assert heartbeat['authenticated_network_roundtrip'] is True
    assert 0 <= time.time() - heartbeat['unix_time'] <= 45
    os.kill(heartbeat['pid'], 0)
    return {'watch_pid': heartbeat['pid'], 'heartbeat_unix_time': heartbeat['unix_time'],
            'readiness_deadline_utc': state['readiness_deadline_utc'],
            'stop_deadline_utc': state['stop_deadline_utc']}


def smoke_proof(destination, transfer):
    smoke = json.loads((destination / 'smoke.json').read_text())
    assert smoke['instance_id'] == 56373998 and smoke['source_commit'] == COMMIT
    assert smoke['cuda_operation_verified'] is True and smoke['cuda_tensor_value'] == 6.0
    assert smoke['model_inference_performed'] is False
    expected = {'torch': '2.8.0', 'transformers': '4.57.1', 'accelerate': '1.10.1',
                'huggingface-hub': '0.35.3', 'safetensors': '0.6.2'}
    assert all(smoke['packages'][name].split('+')[0] == version for name, version in expected.items())
    assert tuple(map(int, smoke['torch_cuda'].split('.')[:2])) >= (12, 8)
    for name in ('models.json', 'protocol.json', 'questions.json', 'judge-prompts.json'):
        original = subprocess.check_output(['git', '-C', str(TASK), 'show',
            COMMIT + ':experiments/emergent-misalignment-screen/' + name])
        assert (destination / 'inputs' / name).read_bytes() == original
    proof = {**IDENTITY, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
             'gpu_operation_verified': True, 'external_transfer_hash_verified': True,
             'controller_deadline_verified': True, 'source_commit': COMMIT,
             'transfer': transfer, 'controller_watch': watchdog()}
    private_write(ROOT / 'smoke-transfer-proof.json', json.dumps(proof, indent=2) + '\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guard', 'capabilities', 'retrieve', 'stage1'))
    parser.add_argument('--origin')
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    evidence_path = ROOT / 'current-network-evidence.json'
    evidence = json.loads(evidence_path.read_text())
    guarded_instance()
    if args.origin:
        origin = urlsplit(args.origin)
        assert args.mode == 'guard' and origin.scheme == 'https' and origin.hostname
        assert origin.port in (None, 443) and origin.path in ('', '/')
        assert not origin.username and not origin.password and not origin.query and not origin.fragment
        evidence['approved_worker_origin'] = args.origin.rstrip('/')
    evidence.update(IDENTITY)
    evidence['guarded_instance_verified'] = True
    evidence['guarded_instance_checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    private_write(evidence_path, json.dumps(evidence, indent=2) + '\n')
    if args.mode == 'guard':
        print(json.dumps({**IDENTITY, 'guarded_instance_verified': True,
                          'origin_bound': 'approved_worker_origin' in evidence}))
        return
    portal = Portal(evidence['approved_worker_origin'], ROOT / 'private/worker-bearer',
                    evidence_path, ROOT / 'active-state.json')
    if args.mode == 'capabilities':
        capabilities = json.loads(portal.request('/capabilities', limit=1024 * 1024))
        api = json.loads(portal.request('/openapi.json', limit=2 * 1024 * 1024))
        supported = '/capabilities/provision' in api.get('paths', {})
        assert supported
        print(json.dumps({**IDENTITY, 'capabilities_received': isinstance(capabilities, dict),
                          'provision_route_present': supported}))
    elif args.mode == 'retrieve':
        assert args.destination is not None
        transfer = portal.retrieve(args.destination)
        proof = smoke_proof(args.destination, transfer)
        print(json.dumps(proof))
    else:
        ready = json.loads((ROOT / 'ready.json').read_text())
        assert all(ready.get(k) == v for k, v in IDENTITY.items())
        assert all(ready.get(k) is True for k in ('gpu_operation_verified',
            'external_transfer_hash_verified', 'controller_deadline_verified'))
        watchdog()
        manifest = json.loads((ROOT / 'stage1-manifest.json').read_text())
        command = '/workspace/em-screen-venv/bin/python /workspace/em-screen-task/worker_task.py run --samples 5'
        assert manifest['post_commands'] == [command]
        marker = ROOT / 'stage1-requested.json'
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump({**IDENTITY, 'requested_at_utc': datetime.now(timezone.utc).isoformat(),
                       'manifest_sha256': hashlib.sha256((ROOT / 'stage1-manifest.json').read_bytes()).hexdigest()}, stream)
        print(json.dumps({'stage1': portal.provision(manifest), 'target_answers': 120}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('Controller portal operation failed (' + type(error).__name__ + '); raw diagnostics suppressed')
