"""Download and hash-check the public pinned model without loading it."""
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'expected-base-verification.json').read_text())
snapshot_download(repo_id=manifest['repo'], revision=manifest['revision'],
                  allow_patterns=[row['file'] for row in manifest['files']],
                  local_dir=str(root / 'model'), max_workers=4, token=False)
for row in manifest['files']:
    path = root / 'model' / row['file']
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == row['sha256'] and path.stat().st_size == row['bytes'], row['file']
print(json.dumps({'download_verified': True, 'files': len(manifest['files'])}), flush=True)
