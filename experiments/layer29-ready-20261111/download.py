"""Download pinned native weights and full layer 29; verify every byte."""
import hashlib,json,os
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from huggingface_hub import snapshot_download,hf_hub_download
ROOT=Path(__file__).resolve().parent
base=json.loads((ROOT/'expected-base-verification.json').read_text())
molt=json.loads((ROOT/'molt-manifest.json').read_text())
protocol=json.loads((ROOT/'protocol.json').read_text())
rows=[]
snapshot_download(base['repo'],revision=base['revision'],allow_patterns=[x['file'] for x in base['files']],local_dir=ROOT/'model',max_workers=4,token=False)
for x in base['files']:
 p=ROOT/'model'/x['file']
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==x['sha256'],x['file']
 assert p.stat().st_size==x['bytes'];rows.append(dict(path=str(p.relative_to(ROOT)),sha256=x['sha256'],bytes=x['bytes']))
print('BASE_WEIGHTS_VERIFIED',flush=True)
for name in ['layer-29/config.json','layer-29/checkpoint.safetensors']:
 hf_hub_download(molt['repo'],name,revision=molt['revision'],local_dir=ROOT/'molts',token=False)
 row=molt['files'][0] if name.endswith('safetensors') else dict(file=name,sha256=protocol['molt_config_sha256'])
 p=ROOT/'molts'/name
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],name
 if 'bytes' in row:assert p.stat().st_size==row['bytes']
 rows.append(dict(path=str(p.relative_to(ROOT)),sha256=row['sha256'],bytes=p.stat().st_size))
(ROOT/'weights-verification.json').write_text(json.dumps(dict(base_revision=base['revision'],molt_revision=molt['revision'],files=rows,all_verified=True),indent=2)+'\n')
print('FULL_LAYER29_WEIGHTS_VERIFIED',flush=True)
