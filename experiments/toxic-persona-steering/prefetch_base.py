"""Download public pinned model files and check every expected SHA-256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
EXPECTED = json.loads((ROOT / 'expected-base-verification.json').read_text())
DEST = ROOT / 'models/qwen3-4b'
DEST.mkdir(parents=True, exist_ok=True)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(record):
    path = DEST / record['file']
    if path.exists() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']:
        return
    url = f"https://huggingface.co/{P['base_repo']}/resolve/{P['base_revision']}/{record['file']}?download=true&fetch={time.time_ns()}"
    temporary = path.with_suffix('.download')
    with urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            stream.write(chunk)
    assert temporary.stat().st_size == record['bytes'] and sha(temporary) == record['sha256'], record['file']
    temporary.replace(path)
    print('MODEL_FILE_VERIFIED', record['file'], flush=True)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, EXPECTED['files']))
print('MODEL_READY', flush=True)
