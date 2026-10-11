"""Rebuild the exact selected transcoder rows from public pinned checkpoints.

CPU-only preparation; downloads a few MB and performs no model inference.
"""
import concurrent.futures
import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
MANIFEST = json.loads((ASSETS / 'weights-manifest.json').read_text())


def fetch(url, start, end):
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                url + f'?range={start}-{end}',
                headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=45) as response:
                assert response.status == 206
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                data = response.read(end - start + 2)
                assert len(data) == end - start + 1
                return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def rebuild(row):
    path = ASSETS / row['file']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']:
        return {'layer': row['layer'], 'sha256': row['sha256'], 'already_verified': True}
    url = ('https://huggingface.co/mwhanna/qwen3-4b-transcoders/resolve/'
           + MANIFEST['revision'] + f'/layer_{row["layer"]}.safetensors')
    first = fetch(url, 0, 65535)
    length = struct.unpack('<Q', first[:8])[0]
    assert length < 65528
    header = json.loads(first[8:8 + length])
    base = 8 + length
    assert header['W_enc']['shape'] == header['W_dec']['shape'] == [163840, 2560]
    assert all(header[k]['dtype'] == 'BF16' for k in ['W_enc', 'W_dec', 'b_enc'])
    lo, hi = header['b_enc']['data_offsets']
    bias = fetch(url, base + lo, base + hi - 1)
    pieces = []
    for feature in row['features']:
        index = feature['feature']
        for name in ['W_enc', 'W_dec']:
            begin = base + header[name]['data_offsets'][0] + index * 5120
            pieces.append(fetch(url, begin, begin + 5119))
        pieces.append(bias[index * 2:index * 2 + 2])
    payload = b''.join(pieces)
    assert hashlib.sha256(payload).hexdigest() == row['sha256'], 'Published row hash mismatch'
    path.write_bytes(payload)
    return {'layer': row['layer'], 'sha256': row['sha256'], 'already_verified': False}


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(rebuild, MANIFEST['layers']):
            print(json.dumps(record), flush=True)
