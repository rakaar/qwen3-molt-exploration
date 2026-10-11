"""Package completed, locally verified research evidence; excludes cloud credentials."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
RESULTS = WORKSPACE / 'results/2026-10-10-small-hot-matched'
DIST = WORKSPACE / 'sites/small-hot-matched/dist'
assert json.loads((RESULTS/'artifacts/results.json').read_text())['status'] == 'completed'
assert json.loads((RESULTS/'artifacts/verification.json').read_text())['passed']
assert json.loads((RESULTS/'cleanup.json').read_text())['destroyed']
for name in ['results.json', 'verification.json', 'download-manifest.json']:
    shutil.copy2(RESULTS/'artifacts'/name, DIST/name)
for name, source in [('protocol.json', ROOT/'protocol.json'), ('cleanup.json', RESULTS/'cleanup.json')]:
    shutil.copy2(source, DIST/name)
labels = DIST/'labels'
labels.mkdir(exist_ok=True)
for file in (ROOT/'assets').glob('label-*.json'):
    shutil.copy2(file, labels/file.name)
files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh')) + [ROOT/'protocol.json']
files += list((ROOT/'assets').glob('*')) + list((ROOT/'upstream').glob('*.py'))
with zipfile.ZipFile(DIST/'experiment-code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for file in sorted(files):
        if file.is_file(): z.write(file, file.relative_to(ROOT))
    z.write(RESULTS/'artifacts/download-manifest.json', 'artifacts/download-manifest.json')
    z.write(RESULTS/'environment.txt', 'environment.txt')
for file in (RESULTS/'artifacts').glob('*.pt'):
    target = DIST/(file.stem+'.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(file, file.name)
    assert target.stat().st_size < 25*1024**2, target
manifest=[]
for file in sorted(DIST.rglob('*')):
    if file.is_file() and file.name != 'asset-checksums.json':
        with file.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest.append({'file':str(file.relative_to(DIST)),'bytes':file.stat().st_size,'sha256':sha})
(DIST/'asset-checksums.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('SITE_EVIDENCE_PACKAGED', len(manifest))
