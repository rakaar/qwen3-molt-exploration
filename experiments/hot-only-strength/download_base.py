"""Reproduction helper: download and verify the exact public base checkpoint."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
target = ROOT / 'models/qwen3-4b'
snapshot_download(P['base_repo'], revision=P['base_revision'], local_dir=target,
                  allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE*'], token=False)
info = HfApi(token=False).model_info(P['base_repo'], revision=P['base_revision'], files_metadata=True)
assert info.sha == P['base_revision']
records = []
for f in info.siblings:
    path = target / f.rfilename
    if not path.is_file():
        continue
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert path.stat().st_size == f.size
    if f.lfs:
        assert sha == f.lfs.sha256
    records.append({'file': f.rfilename, 'bytes': f.size, 'sha256': sha})
(ROOT / 'artifacts').mkdir(exist_ok=True)
(ROOT / 'artifacts/base-download-verification.json').write_text(json.dumps({
    'repo': P['base_repo'], 'revision': P['base_revision'], 'files': records}, indent=2) + '\n')
print('Pinned base checkpoint verified:', len(records), 'files')
