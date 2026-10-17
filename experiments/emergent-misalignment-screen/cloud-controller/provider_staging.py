"""Narrow task launch and non-following log metadata, only via SPAR guard."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
import re
import ssl
from urllib.parse import urlsplit
import urllib.request

from build_bootstrap import build
from network_evidence import require_network_permission

ROOT = Path('/workspace/work/staged-launch')
PRIVATE = ROOT / 'private'
IMAGE = 'vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311'
QUERY = 'num_gpus=1 gpu_ram>=24 compute_cap>=800 cuda_max_good>=12.8 verified=true rentable=true reliability>=0.98 dph_total<=0.4'


def private_write(path, data):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.is_symlink():
        raise ValueError('Private task state may not be a symlink')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(data)
    path.chmod(0o600)


def check_offer(offer):
    if (offer.get('num_gpus') != 1 or offer.get('gpu_ram', 0) < 24000
            or offer.get('compute_cap', 0) < 800 or offer.get('cuda_max_good', 0) < 12.8
            or offer.get('disk_space', 0) < 50 or not 0 < offer.get('dph_total', 0) <= .4):
        raise ValueError('Offer violates the one-worker GPU/driver/disk/rate limits')
    disk = offer.get('storage_total_cost')
    if not isinstance(disk, (int, float)) or disk < 0:
        raise ValueError('Retained-disk rate unavailable')
    estimate = (8 * offer['dph_total'] + 60 * offer['inet_down_cost']
                + 5 * offer['inet_up_cost'] + 24 * disk)
    if not 0 < estimate <= 9:
        raise ValueError('Eight-hour cost plus data/storage reserve exceeds safe budget')
    return estimate


def run(mode, args, key, initial_user, guard):
    def call(command):
        guard.check_identity(guard.call_cli(key, ['show', 'user']))
        return guard.call_cli(key, command)

    if mode == 'staged-launch':
        if (ROOT / 'resume-only.json').exists():
            raise guard.GuardError('This handoff is resume-only; worker creation is prohibited')
        if len(args) != 1 or not (args[0].isdigit() or args[0] == 'best-stable'):
            raise guard.GuardError('Staged launch requires one offer ID or bounded stable-offer selection')
        watch = json.loads((ROOT / 'watch-heartbeat.json').read_text())
        if (time.time() - watch['unix_time'] > 45 or watch.get('account') != guard.ACCOUNT
                or watch.get('authenticated_network_roundtrip') is not True):
            raise guard.GuardError('Independent controller watch is not current')
        os.kill(watch['pid'], 0)
        marker = ROOT / 'creation-attempted.json'
        if marker.exists():
            raise guard.GuardError('One creation was already attempted; inspect guarded inventory instead')
        offers = call(['search', 'offers', QUERY, '--limit', '100', '--storage', '50', '-o', 'dph_total'])
        if args[0] == 'best-stable':
            stable = set(json.loads((ROOT / 'stable-offer-ids.json').read_text()))
            candidates = []
            for candidate in offers:
                if candidate.get('id') not in stable or candidate.get('compute_cap') != 860:
                    continue
                try:
                    price = check_offer(candidate)
                except ValueError:
                    continue
                candidates.append((price, candidate))
            candidates.sort(key=lambda item: item[0])
            matched = [candidates[0][1]] if candidates else []
        else:
            matched = [o for o in offers if o.get('id') == int(args[0])]
        if len(matched) != 1:
            raise guard.GuardError('Selected offer not present in refreshed results')
        offer = matched[0]
        estimate = check_offer(offer)
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=8)
        readiness = now + timedelta(minutes=30)
        bootstrap = build(deadline.isoformat())
        bootstrap_path = ROOT / 'active-bootstrap.sh'
        private_write(bootstrap_path, bootstrap)
        bootstrap_path.chmod(0o700)
        token = secrets.token_urlsafe(48)
        private_write(PRIVATE / 'worker-bearer', token + '\n')
        state = {'account': guard.ACCOUNT, 'label': guard.LABEL, 'offer_id': offer['id'],
                 'instance_id': None, 'created_at_utc': now.isoformat(),
                 'readiness_deadline_utc': readiness.isoformat(), 'stop_deadline_utc': deadline.isoformat(),
                 'dph_total': offer['dph_total'], 'disk_dph': offer['storage_total_cost'],
                 'data_reserve_usd': 60 * offer['inet_down_cost'] + 5 * offer['inet_up_cost'],
                 'retention_reserve_usd': 24 * offer['storage_total_cost'],
                 'estimate_8h_plus_data_24h_retention_usd': estimate,
                 'initial_credit': initial_user.get('credit'), 'image': IMAGE,
                 'bootstrap_sha256': hashlib.sha256(bootstrap.encode()).hexdigest()}
        command = ['create', 'instance', str(offer['id']), '--label', guard.LABEL, '--disk', '50',
                   '--image', IMAGE, '--ssh', '--direct', '--onstart', str(bootstrap_path),
                   '--env', '-p 1111:1111 -e OPEN_BUTTON_PORT=1111 '
                   '-e PORTAL_CONFIG=localhost:1111:11111:/:Instance_Portal '
                   '-e WEB_PASSWORD=' + token + ' -e ENABLE_AUTH=true -e ENABLE_HTTPS=true -e SERVERLESS=true']
        # Preserve the original guard's live inventory, offer, CUDA and budget checks.
        guard.guard_mutation(command, call)
        private_write(ROOT / 'active-state.json', json.dumps(state, indent=2) + '\n')
        private_write(marker, json.dumps({'created_at_utc': now.isoformat(), 'offer_id': offer['id']}) + '\n')
        try:
            result = call(command)
            identifier = result.get('new_contract') if isinstance(result, dict) else None
            if type(identifier) is not int:
                raise guard.GuardError('Creation response uncertain; inspect inventory, never create again')
            state['instance_id'] = identifier
            private_write(ROOT / 'active-state.json', json.dumps(state, indent=2) + '\n')
        except Exception:
            print(json.dumps({'creation_response_uncertain': True, 'retry_creation': False}))
            raise
        print(json.dumps(state, indent=2))
        return

    if mode == 'recover-worker-auth':
        if args != ['56373998']:
            raise guard.GuardError('Worker auth recovery is restricted to existing instance 56373998')
        row = call(['show', 'instance', '56373998'])
        if not isinstance(row, dict) or row.get('id') != 56373998 or row.get('label') != guard.LABEL:
            raise guard.GuardError('Exact existing-worker identity proof failed')
        environment = row.get('extra_env')
        if isinstance(environment, dict):
            token = environment.get('WEB_PASSWORD')
        elif isinstance(environment, list):
            selected = [item[1] for item in environment
                        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == 'WEB_PASSWORD']
            token = selected[0] if len(selected) == 1 else None
        else:
            token = None
        if not isinstance(token, str) or len(token) < 32 or '\n' in token:
            raise guard.GuardError('Generated WEB_PASSWORD unavailable; no other auth field was selected')
        private_write(PRIVATE / 'worker-bearer', token + '\n')
        print(json.dumps({'instance_id': 56373998, 'account': guard.ACCOUNT, 'label': guard.LABEL,
                          'recovered_variable_name': 'WEB_PASSWORD', 'saved_mode': '0600',
                          'value_printed': False}, indent=2))
        return
    if mode in ('log-metadata', 'log-discover'):
        if len(args) != 1 or not args[0].isdigit():
            raise guard.GuardError('Log metadata requires one explicit instance ID')
        identifier = int(args[0])
        rows = guard.instance_rows(call(['show', 'instances']))
        matches = [r for r in rows if r.get('id') == identifier and r.get('label') == guard.LABEL]
        if len(matches) != 1:
            raise guard.GuardError('Log metadata limited to the exact screening worker')
        if mode == 'log-discover':
            evidence = json.loads((ROOT / 'current-network-evidence.json').read_text())
            observed = datetime.fromisoformat(evidence['observed_at_utc'])
            age = (datetime.now(timezone.utc) - observed).total_seconds()
            policy = evidence.get('network_policy', {})
            url = (PRIVATE / 'log-result-url').read_text().strip()
            parts = urlsplit(url)
            if (not 0 <= age <= 300 or parts.scheme != 'https'
                    or parts.hostname != 's3.amazonaws.com' or parts.port not in (None, 443)
                    or parts.username or parts.password):
                raise guard.GuardError('Fresh exact S3 log permission is required')
            require_network_permission(evidence, parts.hostname)
            proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            snapshot = json.loads(Path('/etc/codex/network-policy.json').read_text())
            if not proxy or proxy != snapshot.get('http_proxy') or urllib.request.proxy_bypass(parts.hostname):
                raise guard.GuardError('Configured managed proxy route required for log retrieval')
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({'https': proxy, 'http': proxy}),
                urllib.request.HTTPSHandler(context=ssl.create_default_context()), NoRedirect())
            # This presigned GET carries no Authorization header, worker bearer
            # or management credential. Never print/store the raw startup body.
            with opener.open(urllib.request.Request(url, method='GET'), timeout=20) as response:
                body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise guard.GuardError('Provider startup logs exceed the bounded discovery limit')
            hosts = sorted(set(re.findall(rb'https://([a-z0-9-]+\.trycloudflare\.com)', body)))
            output = {'instance_id': identifier, 'account': guard.ACCOUNT, 'label': guard.LABEL,
                      'portal_candidate_origins': ['https://' + host.decode('ascii') for host in hosts],
                      'provider_log_host': parts.hostname, 'bytes_scanned_privately': len(body),
                      'raw_logs_saved_or_printed': False, 'candidate_hosts_accessed': False}
            private_write(ROOT / 'portal-discovery.json', json.dumps(output, indent=2) + '\n')
            print(json.dumps(output, indent=2))
            return
        # The official CLI logs command follows result_url. Use its documented
        # request endpoint only, without the following GET or printing its token.
        from vastai.api.client import VastClient, VastSession
        import sys
        sys.path.insert(0, '/workspace/work/controller-shims')
        from vastai_transport import configure
        configure(VastSession)
        guard.check_identity(guard.call_cli(key, ['show', 'user']))
        client = VastClient(api_key=key, server_url=guard.HOST, retry=1, timeout=30)
        response = client.put(f'/instances/request_logs/{identifier}/', json_data={'tail': '2000'})
        response.raise_for_status()
        result = response.json()
        url = result.get('result_url')
        if not isinstance(url, str):
            raise guard.GuardError('Provider did not return log result metadata')
        parts = urlsplit(url)
        if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
            raise guard.GuardError('Provider log result is not an approved-style HTTPS URL')
        private_write(PRIVATE / 'log-result-url', url + '\n')
        print(json.dumps({'instance_id': identifier, 'label': guard.LABEL,
                          'result_scheme': parts.scheme, 'result_hostname': parts.hostname,
                          'result_port': parts.port or 443, 'result_fetched': False,
                          'query_and_capability_omitted': True}, indent=2))
        return
    raise guard.GuardError('Unsupported staged provider mode')
