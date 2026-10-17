"""Retrieve only bounded EMSCREEN2 task frames via guarded official Vast/S3."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import subprocess
import time
from urllib.parse import urlsplit
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

from log_channel import (INSTANCE, RUN_ID, MAX_LOG_BYTES, assemble, loads,
                         records_from_log, save_archive)
from network_evidence import require_network_permission
from portal_client import NoRedirect
from provider_staging import private_write

ROOT = Path('/workspace/work/staged-launch')
TASK = Path('/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-screen')
COMMIT = '21da619829635b97b1797600e6b3e5bd3ef5e468'


def fetch():
    evidence = json.loads((ROOT / 'current-network-evidence.json').read_text())
    require_network_permission(evidence, 'cloud.vast.ai')
    require_network_permission(evidence, 's3.amazonaws.com')
    env = os.environ.copy()
    env['PATH'] = '/workspace/work/controller-shims:/workspace/work/controller-venv/bin:' + env['PATH']
    for kind in ('CONFIG', 'CACHE', 'STATE'):
        env['XDG_' + kind + '_HOME'] = '/workspace/work/vastai-' + kind.lower()
    result = subprocess.run(['/workspace/work/controller-venv/bin/python',
        str(TASK / 'spar_vast_cloud.py'), 'frame-log-metadata', str(INSTANCE)],
        env=env, cwd=TASK, capture_output=True, text=True, timeout=110)
    if result.returncode:
        raise ValueError('Guarded account/resource/log proof failed')
    metadata = json.loads(result.stdout)
    if metadata.get('instance_id') != INSTANCE or metadata.get('result_hostname') != 's3.amazonaws.com':
        raise ValueError('Unexpected guarded log resource or host')
    url = (ROOT / 'private/log-result-url').read_text().strip()
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.hostname != 's3.amazonaws.com'
            or parts.port not in (None, 443) or parts.username or parts.password):
        raise ValueError('Exact presigned S3 HTTPS origin required')
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()), NoRedirect())
    # No management/worker credential header is sent to this presigned URL.
    for attempt in range(30):
        try:
            with opener.open(urllib.request.Request(url, method='GET'), timeout=20) as response:
                body = response.read(MAX_LOG_BYTES + 1)
            break
        except urllib.error.HTTPError as error:
            # Official log creation is asynchronous. Retry only not-yet-created
            # Missing private S3 objects may return AccessDenied while the
            # asynchronous export is pending. Classify only a bounded XML Code;
            # never print/store the body, capability, or message. No auth retry.
            transient = error.code == 404
            if error.code == 403:
                data = error.read(4097)
                if len(data) <= 4096:
                    try:transient = ET.fromstring(data).findtext('Code') in {'AccessDenied', 'NoSuchKey'}
                    except ET.ParseError:pass
            if not transient or attempt == 29:
                raise
            time.sleep(2)
    if len(body) > MAX_LOG_BYTES:
        raise ValueError('Provider log exceeds the 8 MiB fetch bound')
    return body


def verify_inputs(destination):
    for name in ('models.json', 'protocol.json', 'questions.json', 'judge-prompts.json'):
        original = subprocess.check_output(['git', '-C', str(TASK), 'show',
            COMMIT + ':experiments/emergent-misalignment-screen/' + name])
        if (destination / 'inputs' / name).read_bytes() != original:
            raise ValueError('Transferred input differs from the pinned original experiment')


def select(records, wanted, kind):
    if wanted:
        candidates = [wanted]
    else:
        candidates = []
        for row in reversed(records):
            if row['t'] == 'h' and row['k'] == kind and row['s'] not in candidates:
                candidates.append(row['s'])
    for candidate in candidates:
        selected = [row for row in records if row['s'] == candidate]
        try:
            result = assemble(selected, candidate)
        except ValueError:
            if wanted:
                raise
            continue
        if result[0]['kind'] != kind:
            raise ValueError('Unexpected snapshot kind')
        return result
    raise ValueError('No complete verified snapshot of the requested kind')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('probe', 'archive'), required=True)
    parser.add_argument('--snapshot-id')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--nonce')
    parser.add_argument('--expect-compressed-sha256')
    parser.add_argument('--expect-raw-sha256')
    args = parser.parse_args()
    body = fetch()
    records, scan = records_from_log(body, args.snapshot_id)
    metadata, compressed, raw = select(records, args.snapshot_id, args.kind)
    if ((args.expect_compressed_sha256 and metadata['sha256'] != args.expect_compressed_sha256)
            or (args.expect_raw_sha256 and metadata['raw_sha256'] != args.expect_raw_sha256)):
        raise ValueError('Snapshot differs from the independently reported hash')
    result = {'account': 710735, 'instance_id': INSTANCE, 'run_id': RUN_ID,
              'snapshot': metadata, 'scan': scan, 'compressed_and_raw_hashes_verified': True,
              'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
    if args.kind == 'probe':
        payload = loads(raw)
        if not args.nonce or payload != {'probe': args.nonce}:
            raise ValueError('Probe nonce mismatch')
        result.update({'exact_probe_received': True, 'nonce': args.nonce})
        private_write(ROOT / 'log-channel-probe-proof.json', json.dumps(result, indent=2) + '\n')
    else:
        if args.destination is None:
            raise ValueError('Private external backup destination required')
        result['transfer'] = save_archive(metadata, compressed, raw, args.destination)
        verify_inputs(args.destination)
        result.update({'pinned_inputs_verified': True,
                       'smoke_present': (args.destination / 'smoke.json').is_file(),
                       'readiness_marked': False})
        private_write(args.destination / 'log-channel-transfer-proof.json', json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('Bounded log retrieval failed (' + type(error).__name__ + '); raw diagnostics suppressed')
