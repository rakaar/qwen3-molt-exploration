import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = [
    ('georglange/qwen3-4b-molt', 'eed10292a96521837f5d65b36fdfa3bf5bea0037', 'qwen3-4b-molt',
     ['layer-17/*', 'layer-25/*', 'README.md']),
]
api = HfApi(token=False)
manifest = []
for repo, revision, directory, patterns in MODELS:
    started = time.monotonic()
    target = ROOT / 'models' / directory
    print(f'Downloading {repo} at {revision}', flush=True)
    snapshot_download(repo, revision=revision, local_dir=target,
                      allow_patterns=patterns, token=False, max_workers=4)
    download_seconds = time.monotonic() - started
    info = api.model_info(repo, revision=revision, files_metadata=True)
    records = []
    for item in info.siblings:
        path = target / item.rfilename
        if not path.is_file():
            continue
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        assert path.stat().st_size == item.size, item.rfilename
        if item.lfs:
            assert digest == item.lfs.sha256, item.rfilename
        records.append({'file': item.rfilename, 'bytes': item.size, 'sha256': digest,
                        'hub_sha256_verified': bool(item.lfs)})
    total = sum(record['bytes'] for record in records)
    manifest.append({'repo': repo, 'revision': revision, 'path': str(target),
                     'download_seconds_this_invocation': download_seconds, 'bytes': total,
                     'timing_note': 'Download resumed from cached partial files; not a network benchmark',
                     'files': records})
    print(f'Verified {repo}: {total / 1e9:.3f} GB; resumed invocation took '
          f'{download_seconds:.1f}s', flush=True)
    (ROOT / 'molt-download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('DOWNLOADS_AND_CHECKSUMS_OK', flush=True)
