"""Prepare fixed task manifests offline; no model execution or network here."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

ROOT = Path(__file__).resolve().parent
REPO = Path('/workspace/qwen3-molt-exploration')
PREFIX = 'experiments/emergent-misalignment-screen/'
COMMIT = '21da619829635b97b1797600e6b3e5bd3ef5e468'
PYTHON = '/workspace/em-screen-venv/bin/python'
DRIVER = '/workspace/em-screen-task/worker_task.py'


def setup_manifest():
    names = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-r', '--name-only',
                                     COMMIT, PREFIX], text=True).splitlines()
    files = {}
    included = {'run.py', 'screenlib.py', 'summarize.py', 'models.json', 'protocol.json',
                'questions.json', 'judge-prompts.json', 'requirements-worker.txt'}
    for path in names:
        relative = path.removeprefix(PREFIX)
        if relative in included or relative.startswith('upstream/'):
            data = subprocess.check_output(['git', '-C', str(REPO), 'show', COMMIT + ':' + path])
            files['experiment/' + relative] = base64.b64encode(data).decode()
    for name in ('worker_task.py', 'portal_client.py', 'export_artifacts.py', 'network_evidence.py'):
        files[name] = base64.b64encode((ROOT / name).read_bytes()).decode()
    provenance = {'source_commit': COMMIT, 'sha256': {name: hashlib.sha256(base64.b64decode(data)).hexdigest()
                                                   for name, data in files.items()}}
    files['provenance.json'] = base64.b64encode(json.dumps(provenance, indent=2).encode()).decode()
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    program = (
        "import base64,json,pathlib,zlib;"
        "root=pathlib.Path('/workspace/em-screen-task');root.mkdir(parents=True,exist_ok=True);"
        "files=json.loads(zlib.decompress(base64.b64decode(pathlib.Path('/workspace/em-screen-bootstrap/task-payload.b64').read_text())));"
        "[(lambda p,d:(p.parent.mkdir(parents=True,exist_ok=True),p.write_bytes(base64.b64decode(d))))(root/n,d) for n,d in files.items()]"
    )
    return {'version': 1, 'settings': {'log_file': '/var/log/portal/em-screen-provisioning.log'},
            'on_failure': {'action': 'continue', 'max_retries': 0},
            'write_files': [{'path': '/workspace/em-screen-bootstrap/task-payload.b64',
                             'content': payload, 'permissions': '0600'}],
            'post_commands': ['python3 -c ' + shlex.quote(program),
                              'python3 -m venv /workspace/em-screen-venv',
                              PYTHON + ' -m pip install --no-cache-dir --index-url https://pypi.org/simple -r /workspace/em-screen-task/experiment/requirements-worker.txt',
                              PYTHON + ' -m pip check', PYTHON + ' ' + DRIVER + ' smoke']}


def phase_manifest(phase, models=None):
    if phase == 'setup':return setup_manifest()
    command = PYTHON + ' ' + DRIVER
    if phase == 'model-smoke':command += ' model-smoke'
    elif phase == 'stage1':command += ' run --samples 5'
    elif phase == 'stage2':
        if not models or 'baseline' not in models or set(models) - {'baseline', 'misaligned-v1', 'misaligned-v2'}:
            raise ValueError('Expansion requires baseline and only reviewed candidates')
        command += ' run --samples 10 --models ' + ' '.join(models)
    elif phase == 'export':command += ' export'
    else:raise ValueError('Unsupported fixed task phase')
    return {'version': 1, 'settings': {'log_file': '/var/log/portal/em-screen-provisioning.log'},
            'on_failure': {'action': 'continue', 'max_retries': 0}, 'post_commands': [command]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('setup', 'model-smoke', 'stage1', 'stage2', 'export'))
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(phase_manifest(args.phase, args.models), indent=2) + '\n')
    print(json.dumps({'manifest': str(args.output), 'phase': args.phase,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'executed': False}))
