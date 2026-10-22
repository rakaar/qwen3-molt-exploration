"""Fetch only the pinned, hash-listed public base-model files."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'expected-base-verification.json').read_text())
started = time.monotonic()
path = snapshot_download(repo_id=manifest['repo'], revision=manifest['revision'],
          allow_patterns=[item['file'] for item in manifest['files']],
          local_dir=str(root / 'model'), max_workers=4, token=False)
for item in manifest['files']:
    f = root / 'model' / item['file']
    with f.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == item['sha256'] and f.stat().st_size == item['bytes'], item['file']
print(json.dumps({'download_verified': True, 'files': len(manifest['files']),
                  'elapsed_seconds': time.monotonic()-started}), flush=True)
