"""Standard-library validation for the one existing GPU's detached driver."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from log_channel import loads

INSTANCE = 56373998
RUN_ID = 'qwen3-em-56373998-20261017'
COMMIT = '21da619829635b97b1797600e6b3e5bd3ef5e468'
EXPERIMENT_ID = '84c11857a364623fdc72776dc8ddd467634842d6c25bce9c6b30f294a85c62a3'
OVERALL = datetime.fromisoformat('2026-10-17T17:16:57.886044+00:00')
READINESS = datetime.fromisoformat('2026-10-17T10:46:57.886044+00:00')
MODELS = {'baseline', 'misaligned-v1', 'misaligned-v2'}
DECISION_PATH = 'experiments/emergent-misalignment-screen/decisions/' + RUN_ID + '.json'
DECISION_URL = ('https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/'
                'codex/qwen3-em-screen-cloud/' + DECISION_PATH)
DECISION_FIELDS = {'protocol', 'instance_id', 'run_id', 'pilot_sha256', 'action',
                   'models', 'samples', 'issued_at_utc', 'expires_at_utc'}


def utc(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError('Explicit UTC timestamp required')
    return result


def validate_decision(data, pilot_sha256, now=None):
    now = now or datetime.now(timezone.utc)
    if len(data) > 16384:
        raise ValueError('Decision exceeds 16 KiB')
    row = loads(data)
    if (not isinstance(row, dict) or set(row) != DECISION_FIELDS
            or row['protocol'] != 'EMSCREEN_DECISION_V1' or type(row['instance_id']) is not int
            or row['instance_id'] != INSTANCE or row['run_id'] != RUN_ID
            or row['action'] != 'extend' or type(row['samples']) is not int or row['samples'] != 10
            or not isinstance(row['pilot_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', row['pilot_sha256'])
            or row['pilot_sha256'] != pilot_sha256):
        raise ValueError('Decision identity, protocol, action, samples, or pilot hash mismatch')
    models = row['models']
    if (not isinstance(models, list) or not 2 <= len(models) <= 3
            or any(type(m) is not str for m in models) or len(set(models)) != len(models)
            or set(models) - MODELS or models[0] != 'baseline'):
        raise ValueError('Extension requires baseline and only reviewed allowlisted candidates')
    if any(not isinstance(row[k], str) or len(row[k]) > 64 for k in ('issued_at_utc', 'expires_at_utc')):
        raise ValueError('Decision timestamps must be bounded UTC strings')
    issued, expires = utc(row['issued_at_utc']), utc(row['expires_at_utc'])
    if not issued <= now < expires <= OVERALL or expires - issued > timedelta(hours=1):
        raise ValueError('Decision timing exceeds its bounded validity or original stop')
    return row


def verify_sources(experiment, manifest):
    experiment = Path(experiment)
    if manifest.get('source_commit') != COMMIT or manifest.get('experiment_id') != EXPERIMENT_ID:
        raise ValueError('Pinned source manifest identity mismatch')
    required = {'run.py', 'screenlib.py', 'summarize.py', 'models.json', 'protocol.json',
                'questions.json', 'judge-prompts.json', 'requirements-worker.txt'}
    files = manifest.get('files', {})
    if not required <= set(files) or not 8 <= len(files) <= 64:
        raise ValueError('Pinned source coverage missing or excessive')
    for name, expected in files.items():
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or not (name in required or name.startswith('upstream/')):
            raise ValueError('Unapproved source-manifest path')
        path = experiment / name
        if path.is_symlink() or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid pinned source digest or symlink')
        with path.open('rb') as stream:data = stream.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Original experiment source changed')


def verify_runtime(smoke):
    expected = {'torch': '2.8.0', 'transformers': '4.57.1', 'accelerate': '1.10.1',
                'huggingface-hub': '0.35.3', 'safetensors': '0.6.2'}
    if (smoke.get('instance_id') != INSTANCE or smoke.get('source_commit') != COMMIT
            or smoke.get('cuda_operation_verified') is not True or smoke.get('cuda_tensor_value') != 6.0
            or smoke.get('total_vram_bytes', 0) < 24_000_000_000
            or tuple(map(int, smoke.get('torch_cuda', '0.0').split('.')[:2])) < (12, 8)
            or any(smoke.get('packages', {}).get(k, '').split('+')[0] != v for k, v in expected.items())):
        raise ValueError('Real pinned-runtime CUDA smoke is not ready')


def run_command(samples, models=None, seconds=10800):
    if samples not in (5, 10) or not 1 <= seconds <= 10800:
        raise ValueError('Unsupported run bounds')
    if samples == 5 and models is not None:
        raise ValueError('Stage one always runs all three pinned models')
    if samples == 10:
        if (not isinstance(models, list) or not 2 <= len(models) <= 3
                or models[0] != 'baseline' or len(set(models)) != len(models) or set(models) - MODELS):
            raise ValueError('Invalid extension model set')
    command = ['/workspace/em-screen-venv/bin/python', '/workspace/em-screen-task/experiment/run.py',
               '--output', '/workspace/em-screen-results', '--samples', str(samples),
               '--max-seconds', str(seconds)]
    if models:command += ['--models', *models]
    return command
