"""Download six public, revision-pinned full MOLT checkpoints and verify hashes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'molt-manifest.json').read_text())

def one(row):
    started = time.time()
    for filename in [row['file'].replace('checkpoint.safetensors', 'config.json'), row['file']]:
        hf_hub_download(manifest['repo'], filename, revision=manifest['revision'],
                        local_dir=str(ROOT / 'molts'), token=False)
    p = ROOT / 'molts' / row['file']
    with p.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    assert actual == row['sha256'] and p.stat().st_size == row['bytes'], row
    print(json.dumps(dict(layer=row['layer'], verified=True, seconds=time.time()-started)), flush=True)
    return row

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, manifest['files']))
    (ROOT / 'molt-download-verified.json').write_text(json.dumps(rows, indent=2)+'\n')
