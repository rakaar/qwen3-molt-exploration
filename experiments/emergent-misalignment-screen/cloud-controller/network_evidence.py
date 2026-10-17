"""Honest, bounded proxy evidence when platform enforcement attestation is unknown.

Never changes platform observation fields. Functional proof is separate, fresh,
credential-free, and bound to the current managed policy file and exact hosts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
import urllib.request

SNAPSHOT = Path('/etc/codex/network-policy.json')
PROXY = 'http://proxy:8080'


def fresh(value, seconds=300):
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(value)).total_seconds()
    if not 0 <= age <= seconds:
        raise ValueError('Network evidence is stale or future-dated')


def context(evidence, hostname):
    if evidence.get('observations_current') is not True:
        raise ValueError('Current platform observations are required')
    fresh(evidence['observed_at_utc'])
    raw = SNAPSHOT.read_bytes()
    snapshot = json.loads(raw)
    hosts = {r['host'] for r in snapshot.get('http_network_policy', {}).get('egress_rules', [])}
    live = evidence.get('network_policy', {})
    proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if (snapshot.get('version') != 1 or snapshot.get('http_proxy') != PROXY
            or proxy != PROXY or urllib.request.proxy_bypass(hostname)
            or snapshot.get('http_network_policy', {}).get('type') != 'restricted'
            or hostname not in hosts or hostname not in live.get('effective_allowed_hosts', [])
            or 'example.org' in hosts):
        raise ValueError('Exact host permission and fixed restricted managed proxy required')
    return hashlib.sha256(raw).hexdigest()


def require_network_permission(evidence, hostname):
    snapshot_hash = context(evidence, hostname)
    state = evidence.get('network_policy', {}).get('state')
    if state == 'enforced':
        return
    if state != 'unknown':
        raise ValueError('Unsupported platform network state')
    proof = evidence.get('functional_proxy_proof', {})
    fresh(proof['observed_at_utc'])
    allow = proof.get('allowed', {}).get(hostname, {})
    deny = proof.get('deny_control', {})
    if (proof.get('proxy') != PROXY or proof.get('network_snapshot_sha256') != snapshot_hash
            or proof.get('credential_free') is not True
            or allow.get('proxy_connect_status') != 200 or allow.get('tls_verify_result') != 0
            or allow.get('redirects_followed') != 0 or allow.get('exit_code') != 0
            or not 100 <= allow.get('http_status', 0) <= 599
            or deny.get('hostname') != 'example.org' or deny.get('proxy_connect_status') != 403
            or deny.get('http_status') != 0 or deny.get('exit_code') != 56
            or deny.get('redirects_followed') != 0):
        raise ValueError('Fresh exact-host verified-TLS allow and proxy-deny controls required')


def probe(hostname):
    result = subprocess.run([
        'curl', '-q', '--silent', '--head', '--proxy', PROXY, '--noproxy', '',
        '--proto', '=https', '--max-redirs', '0', '--connect-timeout', '10', '--max-time', '20',
        '--output', '/dev/null', '--write-out',
        '{"proxy_connect_status":%{http_connect},"http_status":"%{http_code}",'
        '"tls_verify_result":%{ssl_verify_result},"redirects_followed":%{num_redirects}}',
        'https://' + hostname + '/'], capture_output=True, text=True, timeout=25)
    value = json.loads(result.stdout)
    value['http_status'] = int(value['http_status'])
    value.update({'hostname': hostname, 'exit_code': result.returncode})
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--probe-host', action='append', required=True)
    args = parser.parse_args()
    if args.evidence.is_symlink():
        raise ValueError('Evidence file may not be a symlink')
    evidence = json.loads(args.evidence.read_text())
    permitted = {'cloud.vast.ai', 'console.vast.ai', 's3.amazonaws.com'}
    if evidence.get('guarded_instance_verified') is True:
        if (evidence.get('account') != 710735 or evidence.get('instance_id') != 56373998
                or evidence.get('label') != 'spar-psy-neuro-em-screen-20261017'):
            raise ValueError('Portal proof is outside the exact task')
        origin = urlsplit(evidence.get('approved_worker_origin', ''))
        if origin.scheme == 'https' and origin.hostname and origin.port in (None, 443):
            permitted.add(origin.hostname)
    hashes = []
    allowed = {}
    for hostname in args.probe_host:
        if hostname not in permitted:
            raise ValueError('Probe destination is outside this task')
        hashes.append(context(evidence, hostname))
        allowed[hostname] = probe(hostname)
    deny = probe('example.org')
    after = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    if any(h != after for h in hashes):
        raise ValueError('Managed policy changed during probes')
    evidence['functional_proxy_proof'] = {
        'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'proxy': PROXY,
        'network_snapshot_sha256': after, 'credential_free': True,
        'allowed': allowed, 'deny_control': deny}
    for hostname in args.probe_host:
        # Check the functional branch even when platform attestation is ready.
        checked = {**evidence, 'network_policy': {**evidence['network_policy'], 'state': 'unknown'}}
        require_network_permission(checked, hostname)
    temporary = args.evidence.with_suffix('.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(evidence, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(args.evidence)
    print(json.dumps({'platform_policy_state': evidence['network_policy']['state'],
                      'functional_proxy_proof': evidence['functional_proxy_proof']}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise SystemExit('Network proof failed; raw diagnostics suppressed')
