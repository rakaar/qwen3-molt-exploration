"""Export only experiment files, with a checksum for every archived payload."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
completion = json.loads((ROOT / 'artifacts/completion.json').read_text())
assert completion['status'] == 'complete'
paired = json.loads((ROOT / 'artifacts/paired-completion.json').read_text())
assert paired['status'] == 'complete'
assert all(r['matched_clean_generated_tokens_identical'] for r in paired['identity_checks'].values())
files = []
for directory in ['artifacts', 'weights']:
    files.extend(p for p in (ROOT / directory).rglob('*')
                 if p.is_file() and p.name != 'export-checksums.json' and not p.name.endswith('.tmp'))
for name in ['run.py', 'baseline-run.py', 'paired_verification.py', 'prepare.py', 'prefetch_base.py', 'protocol.json',
             'selection.json', 'base-prefetch-manifest.json', 'run.log',
             'baseline-and-selection.log', 'run-shape-check-failed.log', 'paired.log']:
    file = ROOT / name
    if file.is_file():
        files.append(file)
hashes = {}
for path in files:
    with path.open('rb') as stream:
        hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, 'sha256').hexdigest()
manifest = ROOT / 'artifacts/export-checksums.json'
manifest.write_text(json.dumps(hashes, indent=2) + '\n')
archive = ROOT / 'results-export.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for file in files + [manifest]:
        tar.add(file, arcname=str(file.relative_to(ROOT)))
with archive.open('rb') as stream:
    archive_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
record = {'file': archive.name, 'bytes': archive.stat().st_size,
          'sha256': archive_sha, 'payload_files': len(files)}
(ROOT / 'archive-record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
