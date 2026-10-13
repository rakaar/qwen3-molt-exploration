import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
complete=json.loads((ROOT/'artifacts/extension-completion.json').read_text())
assert complete['status']=='complete'
assert all(x['clean_ids_identical_to_all_identity_rows'] for x in complete['checks'].values())
files=[]
for d in ['artifacts','weights']:
 files.extend(f for f in (ROOT/d).rglob('*') if f.is_file() and not f.name.endswith('.tmp') and f.name!='export-checksums.json')
for name in ['run.py','extension.py','prepare.py','prefetch_base.py','range_prefetch.py','protocol.json','selection.json','base-prefetch-manifest.json','extension.log']:
 files.append(ROOT/name)
checks={}
for f in files:
 with f.open('rb') as s:checks[str(f.relative_to(ROOT))]=hashlib.file_digest(s,'sha256').hexdigest()
manifest=ROOT/'artifacts/export-checksums.json';manifest.write_text(json.dumps(checks,indent=2)+'\n')
archive=ROOT/'results-export.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for f in files+[manifest]:t.add(f,arcname=str(f.relative_to(ROOT)))
with archive.open('rb') as s:sha=hashlib.file_digest(s,'sha256').hexdigest()
r={'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha,'payload_files':len(files)}
(ROOT/'archive-record.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
