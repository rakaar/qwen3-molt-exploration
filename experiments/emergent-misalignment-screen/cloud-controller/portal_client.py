"""Task-only client for an explicitly approved provider-created portal origin.

No management credential is read here. Requests preserve managed proxy and CA
trust and refuse every redirect. Output archives are never extracted wholesale.
"""
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import ssl
import tarfile
from urllib.parse import urlsplit
import urllib.request
import zipfile

MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_TASK_BYTES = 16 * 1024 * 1024
MAX_TASK_MEMBERS = 64
PREFIX = 'log/em-screen-artifacts/'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Portal:
    def __init__(self, approved_origin, bearer_path, current_policy_evidence, instance_state):
        parts = urlsplit(approved_origin)
        if (parts.scheme != 'https' or not parts.hostname or parts.port not in (None, 443)
                or parts.path not in ('', '/') or parts.query or parts.fragment
                or parts.username or parts.password):
            raise ValueError('An exact approved HTTPS origin is required')
        self.origin = approved_origin.rstrip('/')
        path = Path(bearer_path)
        if path.stat().st_mode & 0o077:
            raise ValueError('Worker bearer file must be private to its owner')
        self._bearer = path.read_text().strip()
        if len(self._bearer) < 32 or '\n' in self._bearer:
            raise ValueError('Invalid worker bearer file')
        policy = json.loads(Path('/etc/codex/network-policy.json').read_text())
        proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        if not proxy or proxy != policy.get('http_proxy'):
            raise ValueError('Inherited managed proxy is required')
        evidence = json.loads(Path(current_policy_evidence).read_text())
        state = json.loads(Path(instance_state).read_text())
        if (state.get('account') != 710735
                or state.get('label') != 'spar-psy-neuro-em-screen-20261017'
                or type(state.get('instance_id')) is not int
                or evidence.get('instance_id') != state['instance_id']
                or evidence.get('account') != 710735
                or evidence.get('label') != state['label']
                or evidence.get('guarded_instance_verified') is not True):
            raise ValueError('Portal origin must bind to the guarded SPAR task instance')
        live = evidence.get('network_policy', {})
        observed = datetime.fromisoformat(evidence['observed_at_utc'])
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        if (not evidence.get('observations_current') or live.get('state') != 'enforced'
                or evidence.get('approved_worker_origin') != self.origin
                or not 0 <= age <= 300):
            raise ValueError('Fresh enforced policy and exact-origin approval evidence required')
        hosts = set(live.get('effective_allowed_hosts', []))
        if parts.hostname not in hosts:
            raise ValueError('Exact worker hostname absent from current enforced policy')
        if urllib.request.proxy_bypass(parts.hostname):
            raise ValueError('Inherited proxy bypass would apply; refusing request')
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'http': proxy, 'https': proxy}),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()), NoRedirect())

    def request(self, path, payload=None, limit=MAX_ZIP_BYTES):
        if path not in ('/capabilities', '/openapi.json', '/download-logs', '/capabilities/provision'):
            raise ValueError('Unsupported task portal route')
        data = None if payload is None else json.dumps(payload).encode()
        headers = {'Authorization': 'Bearer ' + self._bearer}
        if data is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(self.origin + path, data=data, headers=headers)
        with self.opener.open(req, timeout=30) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError('Portal response exceeds the task transfer limit')
        return data

    def provision(self, manifest):
        # JSON is a YAML subset; no extra YAML parser is needed on the controller.
        result = json.loads(self.request('/capabilities/provision',
                                        {'inline_yaml': json.dumps(manifest)}, 1024 * 1024))
        if result.get('status') != 'started':
            raise ValueError('Portal did not acknowledge detached provisioning')
        return {'status': 'started'}

    def retrieve(self, destination):
        return extract_task_snapshot(self.request('/download-logs'), destination)


def allowed_result(name):
    if name in ('responses.jsonl', 'generation-status.json', 'summary.json', 'summary.md',
                'worker-exit-code.txt', 'worker.log', 'smoke.json', 'provenance.json',
                'task-status.json', 'native-deadline-status.json', 'SHA256SUMS'):
        return True
    if name.startswith('runtime-') and name.endswith('.json'):
        return name[8:-5].isdigit()
    return name in ('inputs/models.json', 'inputs/protocol.json', 'inputs/questions.json',
                    'inputs/judge-prompts.json')


def extract_task_snapshot(data, destination):
    """Read only exact task ZIP members; never persist other /var/log contents."""
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError('Oversized portal ZIP')
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        wanted = [PREFIX + 'results.tar.gz', PREFIX + 'results.sha256']
        for name in wanted:
            matches = [i for i in outer.infolist() if i.filename == name]
            if len(matches) != 1 or matches[0].file_size > MAX_TASK_BYTES:
                raise ValueError('Missing, duplicate, or oversized task artifact')
        archive = outer.read(wanted[0])
        expected = outer.read(wanted[1]).decode('ascii').strip()
    digest = hashlib.sha256(archive).hexdigest()
    if len(expected) != 64 or digest != expected:
        raise ValueError('Task archive hash mismatch')
    extracted = {}
    expanded_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as task:
        for member in task:
            p = PurePosixPath(member.name)
            if (not member.isfile() or p.is_absolute() or '..' in p.parts
                    or not allowed_result(member.name) or member.name in extracted
                    or member.size > MAX_TASK_BYTES or len(extracted) >= MAX_TASK_MEMBERS
                    or expanded_bytes + member.size > MAX_TASK_BYTES):
                raise ValueError('Unsafe or unapproved task archive member')
            extracted[member.name] = task.extractfile(member).read(member.size + 1)
            expanded_bytes += len(extracted[member.name])
    if 'SHA256SUMS' not in extracted:
        raise ValueError('Task file checksum manifest missing')
    checks = {}
    for line in extracted['SHA256SUMS'].decode('ascii').splitlines():
        if len(checks) >= MAX_TASK_MEMBERS or len(line) > 300:
            raise ValueError('Oversized task checksum manifest')
        checksum, name = line.split('  ', 1)
        if name in checks or name not in extracted or name == 'SHA256SUMS':
            raise ValueError('Invalid task checksum manifest')
        checks[name] = checksum
    if set(checks) != set(extracted) - {'SHA256SUMS'}:
        raise ValueError('Task checksum coverage incomplete')
    for name, checksum in checks.items():
        if hashlib.sha256(extracted[name]).hexdigest() != checksum:
            raise ValueError('Task file hash mismatch')
    destination = Path(destination)
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError('External backup destination may not contain symlinks')
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.chmod(0o700)
    for name, contents in extracted.items():
        path = destination / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('External backup path may not contain symlinks')
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.write_bytes(contents)
        path.chmod(0o600)
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(contents).hexdigest():
            raise ValueError('External backup verification failed')
    (destination / 'retrieved-archive.sha256').write_text(digest + '\n')
    (destination / 'retrieved-archive.sha256').chmod(0o600)
    return {'archive_sha256': digest, 'files_verified': len(checks),
            'external_backup': str(destination)}
