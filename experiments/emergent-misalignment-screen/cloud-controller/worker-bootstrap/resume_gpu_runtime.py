"""Resume the fixed worker runtime using cached, bounded official downloads."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path('/workspace/em-screen-bootstrap')
TASK = Path('/workspace/em-screen-task')


def record(phase, **extra):
    value = {'instance_id': 56373998, 'phase': phase,
             'checked_at_utc': datetime.now(timezone.utc).isoformat(), **extra}
    p = ROOT / 'gpu-setup-status.retry.tmp'
    p.write_text(json.dumps(value, indent=2) + '\n')
    p.chmod(0o600)
    p.replace(ROOT / 'gpu-setup-status.json')


def main():
    assert os.environ.get('CONTAINER_ID') == '56373998'
    assert 'SPAR_VAST_API_KEY' not in os.environ
    with (ROOT / 'runtime-retry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = Path('/proc/1422/cmdline')
        assert not original.exists() or b'prepare_gpu_runtime.py' not in original.read_bytes()
        assert shutil.disk_usage('/workspace').free > 35_000_000_000
        env = {k: v for k, v in os.environ.items() if not k.startswith('UV_')}
        env.update(UV_CACHE_DIR='/workspace/em-screen-uv-cache',
                   UV_CONCURRENT_DOWNLOADS='4', UV_CONCURRENT_INSTALLS='2',
                   UV_CONCURRENT_BUILDS='1', UV_NATIVE_TLS='true')
        runtime = '/workspace/em-screen-venv/bin/python'
        record('installing-pinned-gpu-packages-cached-retry', original_timeout_seconds=900)
        subprocess.run(['/usr/local/bin/uv', '--no-config', 'pip', 'install',
                        '--python', runtime, '--index-url', 'https://pypi.org/simple',
                        '-r', str(TASK / 'experiment/requirements-worker.txt')],
                       check=True, timeout=1440, env=env)
        subprocess.run([runtime, '-m', 'pip', 'check'], check=True, timeout=60)
        record('verifying-real-cuda-tensor')
        subprocess.run([runtime, str(TASK / 'worker_task.py'), 'smoke'],
                       check=True, timeout=120)
        record('complete', inference_executed=False, cuda_smoke_completed=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('failed', error_type=type(error).__name__)
        raise SystemExit(1)
