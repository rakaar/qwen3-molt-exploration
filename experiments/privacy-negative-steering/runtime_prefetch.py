"""Fetch pinned official runtime wheels in parallel, checking every full SHA256."""
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'runtime-wheels'
PARTS = ROOT / 'runtime-wheel-parts'
DEST.mkdir(exist_ok=True); PARTS.mkdir(exist_ok=True)
manifest = json.loads((ROOT / 'runtime-wheel-manifest.json').read_text())
CHUNK = 32 * 1024 * 1024


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


jobs = []
for record in sorted(manifest['files'], key=lambda r: -r['bytes']):
    path = DEST / record['filename']
    if path.exists() and path.stat().st_size == record['bytes'] and digest(path) == record['sha256']:
        continue
    directory = PARTS / record['filename']; directory.mkdir(exist_ok=True)
    candidates = list(Path('/tmp').glob('pip-unpack-*/' + record['filename']))
    preserved = DEST / (record['filename'] + '.prefetch')
    if preserved.exists(): candidates.append(preserved)
    partial = max(candidates, key=lambda p: p.stat().st_size, default=None)
    if partial is not None:
        with partial.open('rb') as stream:
            for start in range(0, partial.stat().st_size // CHUNK * CHUNK, CHUNK):
                piece = directory / f'{start:012d}.part'
                if not piece.exists(): piece.write_bytes(stream.read(CHUNK))
                else: stream.seek(CHUNK, 1)
    for start in range(0, record['bytes'], CHUNK):
        end = min(start + CHUNK, record['bytes']) - 1
        piece = directory / f'{start:012d}.part'
        if not piece.exists() or piece.stat().st_size != end - start + 1:
            jobs.append((record, start, end, piece))


def fetch(job):
    record, start, end, piece = job
    temporary = piece.with_suffix('.tmp')
    for attempt in range(4):
        try:
            offset = start + (temporary.stat().st_size if temporary.exists() else 0)
            if offset == end + 1:
                temporary.replace(piece); return
            request = Request(record['url'], headers={'Range': f'bytes={offset}-{end}'})
            began = time.monotonic()
            with urlopen(request, timeout=60) as response:
                assert response.status == 206, (record['filename'], response.status)
                assert response.headers['Content-Range'] == f'bytes {offset}-{end}/{record["bytes"]}'
                with temporary.open('ab') as stream:
                    for data in iter(lambda: response.read(65536), b''):
                        stream.write(data)
                        if time.monotonic() - began > 120: raise TimeoutError('Resume stalled wheel range')
            assert temporary.stat().st_size == end - start + 1
            temporary.replace(piece)
            return
        except Exception:
            if attempt == 3: raise
            time.sleep(attempt + 1)


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(fetch, jobs))
for record in manifest['files']:
    path = DEST / record['filename']
    if not (path.exists() and path.stat().st_size == record['bytes'] and digest(path) == record['sha256']):
        temporary = path.with_suffix('.assembled')
        with temporary.open('wb') as stream:
            for start in range(0, record['bytes'], CHUNK):
                with (PARTS / record['filename'] / f'{start:012d}.part').open('rb') as source:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
        assert temporary.stat().st_size == record['bytes'] and digest(temporary) == record['sha256'], record['filename']
        temporary.replace(path)
    shutil.rmtree(PARTS / record['filename'], ignore_errors=True)
    print('PINNED_WHEEL_VERIFIED', record['filename'], flush=True)
(ROOT / 'artifacts/runtime-wheel-verification.json').write_text(json.dumps({'status': 'passed', 'full_wheels_sha256_verified': len(manifest['files']), 'bytes': sum(x['bytes'] for x in manifest['files']), 'official_manifest': 'runtime-wheel-manifest.json'}, indent=2) + '\n')
