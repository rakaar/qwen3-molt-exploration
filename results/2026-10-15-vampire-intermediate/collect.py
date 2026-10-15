"""Bundle completed scientific files; retain canonical model/runtime caches on the GPU."""
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

root = Path('/root/vampire-intermediate')
deadline = time.monotonic() + 1800
while True:
    path = root / 'fractional/artifacts/added-gate-results.json'
    if path.exists() and json.loads(path.read_text())['status'] == 'complete':
        break
    assert time.monotonic() < deadline, 'Collection wait exceeded 30 minutes'
    time.sleep(10)
time.sleep(2)
subprocess.run(['/venv/main/bin/python', str(root / 'verify.py'), str(root)], check=True)
subprocess.run(['/venv/main/bin/python', str(root / 'fractional/verify.py'), str(root / 'fractional')], check=True)
skip = {'models', 'runtime-wheels', 'runtime-wheel-parts', '__pycache__'}
files = [p for p in root.rglob('*') if p.is_file() and not any(x in skip for x in p.relative_to(root).parts) and p.name not in ['molt-19-selected.safetensors', 'collection.log', 'scientific-files-sha256.json']]
manifest = {}
for p in files:
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest[p.relative_to(root).as_posix()] = {'bytes': p.stat().st_size, 'sha256': digest}
record = root / 'artifacts/scientific-files-sha256.json'
record.write_text(json.dumps({'files': manifest, 'excluded_regenerable': sorted(skip | {'molt-19-selected.safetensors'}), 'selected_asset_sha256': '7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd'}, indent=2) + '\n')
archive = Path('/root/vampire-results.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    for p in [*files, record]:
        tar.add(p, arcname=p.relative_to(root).as_posix())
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest, 'scientific_files': len(files)}
Path('/root/vampire-results-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
