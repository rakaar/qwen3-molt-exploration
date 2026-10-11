"""Checksum the completed export before transferring it off the rental."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'artifacts/results.json').read_text())['status']=='completed'
assert json.loads((ROOT/'artifacts/verification.json').read_text())['passed']
files=[]
for folder in ['artifacts','assets','weights','upstream','source']:
    files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh')) + [ROOT/'protocol.json',ROOT/'environment.txt',ROOT/'experiment.log']
if (ROOT/'probe-answers.log').exists(): files.append(ROOT/'probe-answers.log')
rows=[]
for p in sorted(set(files)):
    if p.name=='export-checksums.json':continue
    with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    rows.append({'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest})
(ROOT/'artifacts/export-checksums.json').write_text(json.dumps(rows,indent=2)+'\n')
print('EXPORT_CHECKSUMS_READY',len(rows),sum(x['bytes'] for x in rows))
