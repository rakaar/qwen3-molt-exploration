"""Fetch a pinned public judge; verify LFS checksums and preserve the other model."""
import hashlib
import json
import os
from pathlib import Path
import shutil
os.environ['HF_HUB_DISABLE_XET']='0'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'protocol.json').read_text())
partial_allocation=sum(f.stat().st_blocks*512 for f in (ROOT/'judge').rglob('*.incomplete') if f.is_file()) if (ROOT/'judge').exists() else 0
assert shutil.disk_usage(ROOT).free+partial_allocation>p['judge_weight_bytes']+2_000_000_000,'Insufficient free disk; preserve existing files'
snapshot_download(repo_id=p['judge_repo'],revision=p['judge_revision'],
    allow_patterns=[r['file'] for r in p['model_files']],local_dir=ROOT/'judge',token=False,max_workers=3)
verified=[]
for row in p['model_files']:
    path=ROOT/'judge'/row['file']
    assert path.is_file(),row['file']
    if row.get('bytes') is not None:assert path.stat().st_size==row['bytes'],row['file']
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if row.get('sha256'):assert digest==row['sha256'],row['file']
    verified.append(dict(file=row['file'],bytes=path.stat().st_size,sha256=digest,lfs_hash_verified=bool(row.get('sha256'))))
(ROOT/'judge-verification.json').write_text(json.dumps(dict(repo=p['judge_repo'],revision=p['judge_revision'],files=verified),indent=2)+'\n')
print('JUDGE_WEIGHTS_VERIFIED',flush=True)
