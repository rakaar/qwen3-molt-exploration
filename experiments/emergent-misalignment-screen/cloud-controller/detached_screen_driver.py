"""One existing GPU only: wait for cached setup, smoke, pilot, reviewed extension.

No installer, provider lifecycle call, Portal, executable download, or management
credential is used. Install this reviewed control code via the existing pinned
SSH session, then launch detached with system Python. Model processes use only
the already prepared GPU venv and unchanged original experiment files.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from driver_core import (COMMIT, DECISION_URL, INSTANCE, MODELS, OVERALL, READINESS,
                         RUN_ID, run_command, validate_decision, verify_runtime, verify_sources)
from log_channel import MAX_RAW, task_member, validate_task_content
from portal_client import NoRedirect
from export_artifacts import export

ROOT = Path('/workspace/em-screen-task')
EXPERIMENT = ROOT / 'experiment'
OUTPUT = Path('/workspace/em-screen-results')
EXPORT = Path('/var/log/em-screen-artifacts')
PRIVATE = ROOT / 'private'
PYTHON = '/workspace/em-screen-venv/bin/python'
OWNS_LOCK = False
BASELINE_LOCK = Path('/workspace/em-screen-bootstrap/baseline-model-smoke.lock')


def now():return datetime.now(timezone.utc)


def write_json(path, value):
    if path.is_symlink():raise ValueError('Task path may not be a symlink')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def phase(name, **fields):
    write_json(OUTPUT / 'task-status.json', {'phase': name, 'instance_id': INSTANCE,
        'pid': os.getpid(), 'updated_at_utc': now().isoformat(), **fields})


def source_manifest():
    manifest = json.loads((ROOT / 'driver-pinned-sources.json').read_text())
    verify_sources(EXPERIMENT, manifest)
    return manifest


def update_provenance(**fields):
    path = OUTPUT / 'control-provenance.json'
    value = json.loads(path.read_text())
    if value.get('instance_id') != INSTANCE or value.get('experiment_sources_changed') is not False:
        raise ValueError('Control provenance does not bind the unchanged existing experiment')
    for name in ('detached_screen_driver.py', 'driver_core.py', 'log_channel.py', 'driver-pinned-sources.json'):
        value.setdefault('helper_sha256', {})[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    value.setdefault('detached_driver', {}).update({'run_id': RUN_ID, 'pid': os.getpid(), **fields})
    validate_task_content('control-provenance.json', json.dumps(value).encode())
    write_json(path, value)


def snapshot():
    # Never export the private model process logs. Use the existing exporter on
    # an immutable, validated task-only staging tree; original inputs are added
    # by that exporter. The installed sender watches only its existing archive.
    with tempfile.TemporaryDirectory(prefix='safe-export-', dir=PRIVATE) as folder:
        staged = Path(folder)
        total = 0
        for path in OUTPUT.iterdir():
            if path.name == 'SHA256SUMS' or not task_member(path.name):continue
            if path.is_symlink() or not path.is_file():raise ValueError('Unsafe task output')
            size = path.stat().st_size
            if total + size > MAX_RAW:raise ValueError('Task export exceeds 16 MiB')
            with path.open('rb') as stream:data = stream.read(size + 1)
            if len(data) > size:raise ValueError('Task output changed during staging')
            if path.name == 'responses.jsonl' and data and not data.endswith(b'\n'):
                data = data[:data.rfind(b'\n') + 1]
            validate_task_content(path.name, data)
            (staged / path.name).write_bytes(data);total += len(data)
        return export(staged, EXPERIMENT, EXPORT)


def model_env():
    # Strip all inherited authentication from model children. Public ungated
    # models need no token. The separately running native backstop is untouched.
    allowed = {'PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
        'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE',
        'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
        'http_proxy', 'https_proxy', 'no_proxy', 'TZ', 'CONTAINER_ID'}
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    return env


def child(command, label, timeout=None):
    log = PRIVATE / (label + '.log')
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'ab') as stream:
        process = subprocess.Popen(command, cwd=EXPERIMENT, env=model_env(),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
        started = time.monotonic()
        while process.poll() is None:
            if now() >= OVERALL or timeout is not None and time.monotonic() - started >= timeout:
                os.killpg(process.pid, signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL);process.wait(timeout=10)
                raise TimeoutError('Model process reached its bounded deadline')
            if label in ('stage1', 'stage2'):
                path = OUTPUT / 'responses.jsonl'
                phase('driver-' + label, samples=5 if label == 'stage1' else 10,
                      pilot_responses_started=path.is_file() and path.stat().st_size > 0)
            try:snapshot()
            except (ValueError, OSError):pass
            time.sleep(5)
    if process.returncode:raise RuntimeError('GPU child failed; private log retained')


def baseline_model_smoke():
    baseline = next(m for m in json.loads((EXPERIMENT / 'models.json').read_text()) if m['id'] == 'baseline')
    def valid(smoke):
        model = smoke.get('model_smoke', {})
        return (model.get('repo') == baseline['repo_id'] and model.get('revision') == baseline['revision']
            and str(model.get('device', '')).startswith('cuda')
            and type(model.get('generated_tokens')) is int and 1 <= model['generated_tokens'] <= 4
            and model.get('outside_pilot') is True and model.get('sample_saved_as_pilot_response') is False)
    with BASELINE_LOCK.open('a') as lock:
        while True:
            try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);break
            except BlockingIOError:
                if now() >= OVERALL:raise TimeoutError('Overall stop while awaiting existing model smoke')
                phase('driver-waiting-baseline-smoke');snapshot();time.sleep(5)
        smoke = json.loads((OUTPUT / 'smoke.json').read_text());verify_runtime(smoke)
        if not valid(smoke):
            phase('driver-baseline-model-smoke');snapshot();source_manifest()
            child([PYTHON, str(ROOT / 'worker_task.py'), 'model-smoke'], 'baseline-model-smoke', timeout=900)
            smoke = json.loads((OUTPUT / 'smoke.json').read_text());verify_runtime(smoke)
        if not valid(smoke):raise ValueError('Pinned baseline model smoke outside pilot is invalid')
        # Preserve the original hardware-smoke field. The nested proof records
        # the later baseline inference, including when another launcher ran it.
        return smoke['model_smoke']


def rows_and_pilot():
    source_manifest()
    spec = importlib.util.spec_from_file_location('pinned_screenlib', EXPERIMENT / 'screenlib.py')
    library = importlib.util.module_from_spec(spec);spec.loader.exec_module(library)
    rows = library.validated_rows(OUTPUT / 'responses.jsonl')
    target = library.expected_keys(sorted(MODELS), 5)
    if {library.row_key(row) for row in rows} != target:
        raise ValueError('Stage one is not exactly the original 120 unique answers')
    return library, rows, (OUTPUT / 'responses.jsonl').read_bytes()


def decision(pilot_sha):
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()), NoRedirect())
    try:
        with opener.open(urllib.request.Request(DECISION_URL, headers={'Accept': 'application/json'}), timeout=15) as response:
            data = response.read(16385)
    except urllib.error.HTTPError as error:
        if error.code == 404:return None
        raise
    return validate_decision(data, pilot_sha, now()), hashlib.sha256(data).hexdigest()


def main():
    global OWNS_LOCK
    if int(os.environ.get('CONTAINER_ID', '-1')) != INSTANCE or 'SPAR_VAST_API_KEY' in os.environ:
        raise ValueError('Only existing GPU worker without controller credential is allowed')
    if now() >= OVERALL:raise TimeoutError('Original overall stop has passed')
    ROOT.mkdir(parents=True, exist_ok=True);OUTPUT.mkdir(mode=0o700, exist_ok=True)
    PRIVATE.mkdir(mode=0o700, exist_ok=True);PRIVATE.chmod(0o700)
    with (PRIVATE / 'detached-driver.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OWNS_LOCK = True
        source_manifest();update_provenance(phase='waiting-runtime')
        while True:
            try:
                smoke = json.loads((OUTPUT / 'smoke.json').read_text());verify_runtime(smoke)
                from driver_core import utc
                if utc(smoke['created_at_utc']) > READINESS:
                    raise TimeoutError('Runtime was not ready before its setup cutoff')
                break
            except (OSError, ValueError, KeyError):
                if now() >= READINESS:raise TimeoutError('Runtime readiness cutoff reached')
                phase('driver-waiting-runtime')
                try:snapshot()
                except (ValueError, OSError):pass
                time.sleep(5)
        baseline_proof = baseline_model_smoke()
        update_provenance(baseline_model_inference_verified=True, baseline_model_smoke=baseline_proof)
        phase('driver-stage1', samples=5);update_provenance(phase='stage1');snapshot();source_manifest()
        seconds = min(10800, int((OVERALL - now()).total_seconds()) - 120)
        child(run_command(5, seconds=seconds), 'stage1')
        library, rows, pilot_bytes = rows_and_pilot()
        pilot_sha = hashlib.sha256(pilot_bytes).hexdigest()
        phase('driver-awaiting-review', samples=5)
        update_provenance(phase='awaiting-review', pilot_sha256=pilot_sha, stage1_responses=len(rows));snapshot()
        while now() < OVERALL:
            if (OUTPUT / 'responses.jsonl').read_bytes() != pilot_bytes:
                raise ValueError('Pilot changed while awaiting its hash-bound review decision')
            try:approved = decision(pilot_sha)
            except (ValueError, OSError):approved = None
            if approved:break
            time.sleep(30)
        else:return
        row, decision_sha = approved;source_manifest()
        phase('driver-stage2', samples=10)
        update_provenance(phase='stage2', decision_sha256=decision_sha, extension_models=row['models']);snapshot()
        seconds = min(10800, int((OVERALL - now()).total_seconds()) - 120)
        child(run_command(10, row['models'], seconds), 'stage2')
        data = (OUTPUT / 'responses.jsonl').read_bytes()
        if not data.startswith(pilot_bytes):raise ValueError('Original first batch changed during extension')
        all_rows = library.validated_rows(OUTPUT / 'responses.jsonl')
        expected = library.expected_keys(sorted(MODELS), 5) | library.expected_keys(row['models'], 10)
        if {library.row_key(r) for r in all_rows} != expected:raise ValueError('Extension is incomplete')
        phase('driver-complete', samples=10);update_provenance(phase='complete');snapshot()


if __name__ == '__main__':
    try:main()
    except Exception as error:
        if OWNS_LOCK and OUTPUT.is_dir():
            try:
                phase('driver-failed', error_type=type(error).__name__);snapshot()
            except Exception:pass
        raise SystemExit(1)
