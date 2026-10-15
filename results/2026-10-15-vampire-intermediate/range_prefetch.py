"""Parallel HTTP ranges for immutable public weights; verify complete Hub hashes."""
import concurrent.futures
import fnmatch
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
DEST = ROOT / 'models/qwen3-4b'
PARTS = ROOT / 'models/download-parts'
DEST.mkdir(parents=True, exist_ok=True)
PARTS.mkdir(parents=True, exist_ok=True)
with urlopen(f"https://huggingface.co/api/models/{P['base_repo']}/revision/{P['base_revision']}?blobs=true", timeout=60) as response:
    info = json.loads(response.read())
assert info['sha'] == P['base_revision']
files = [f for f in info['siblings'] if any(fnmatch.fnmatch(f['rfilename'], p) for p in ['*.json', '*.safetensors', '*.txt', 'LICENSE*'])]
CHUNK = 64 * 1024 * 1024
jobs, rows, reused = [], [], {}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid(path, item):
    return (path.is_file() and path.stat().st_size == item['size']
            and (not item.get('lfs') or digest(path) == item['lfs']['sha256']))


for item in files:
    path = DEST / item['rfilename']
    assert path.resolve().is_relative_to(DEST.resolve())
    if valid(path, item):
        reused[item['rfilename']] = item['size']
        continue
    partial = path.with_name(path.name + '.prefetch')
    directory = PARTS / item['rfilename']
    directory.mkdir(parents=True, exist_ok=True)
    salvaged = 0
    if partial.exists():
        with partial.open('rb') as stream:
            for start in range(0, item['size'], CHUNK):
                length = min(CHUNK, item['size'] - start)
                if start + length > partial.stat().st_size:
                    break
                piece = directory / f'{start:012d}.part'
                if piece.exists() and piece.stat().st_size == length:
                    stream.seek(length, 1)
                    continue
                piece.write_bytes(stream.read(length))
                assert piece.stat().st_size == length
                salvaged += length
    reused[item['rfilename']] = salvaged
    for start in range(0, item['size'], CHUNK):
        end = min(start + CHUNK, item['size']) - 1
        piece = directory / f'{start:012d}.part'
        if piece.exists() and piece.stat().st_size == end - start + 1:
            continue
        jobs.append((item, start, end, piece))


def fetch(job):
    item, start, end, piece = job
    temporary = piece.with_suffix('.tmp')
    for attempt in range(4):
        try:
            offset = start + (temporary.stat().st_size if temporary.exists() else 0)
            if offset == end + 1:
                temporary.replace(piece)
                return
            assert start <= offset <= end
            url = f"https://huggingface.co/{P['base_repo']}/resolve/{P['base_revision']}/{item['rfilename']}?range={offset}-{end}&attempt={attempt}&nonce={time.time_ns()}"
            request = Request(url, headers={'Range': f'bytes={offset}-{end}'})
            started = time.monotonic()
            with urlopen(request, timeout=60) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == f"bytes {offset}-{end}/{item['size']}"
                with temporary.open('ab') as stream:
                    for block in iter(lambda: response.read(64 * 1024), b''):
                        stream.write(block)
                        if time.monotonic() - started > 120:
                            raise TimeoutError('Range attempt exceeded two minutes; resume remaining bytes')
            assert temporary.stat().st_size == end - start + 1
            temporary.replace(piece)
            print('RANGE_READY', item['rfilename'], start, end, flush=True)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(fetch, jobs))
for item in files:
    path = DEST / item['rfilename']
    if not valid(path, item):
        temporary = path.with_name(path.name + '.assembled')
        with temporary.open('wb') as stream:
            for start in range(0, item['size'], CHUNK):
                piece = PARTS / item['rfilename'] / f'{start:012d}.part'
                with piece.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        stream.write(block)
        assert valid(temporary, item), item['rfilename']
        temporary.replace(path)
    rows.append({'file': item['rfilename'], 'bytes': path.stat().st_size, 'sha256': digest(path)})
    print('PINNED_FILE_VERIFIED', item['rfilename'], flush=True)
(ROOT / 'base-prefetch-manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
(ROOT / 'artifacts').mkdir(exist_ok=True)
(ROOT / 'artifacts/range-download-audit.json').write_text(json.dumps({
    'repo': P['base_repo'], 'revision': P['base_revision'], 'chunk_bytes': CHUNK,
    'concurrency': 16, 'range_requests': len(jobs), 'reused_bytes': reused,
    'complete_file_hashes_verified_against_hub_lfs': True,
}, indent=2) + '\n')
print('PREFETCH_COMPLETE', sum(r['bytes'] for r in rows), flush=True)
