#!/usr/bin/env python3
"""Retrieve only T2375's public, revision-pinned scoring requests via HTTP ranges.

The export lacks labels/responses. This script never infers them from row order.
Full labels come separately from the authenticated dashboard's feature JSON.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'comparisons/2026-11-09-t2375-explanation'
REVISION = '5160cb8fb34db41545e8ceabb3fb1ea047b519e4'
URL = ('https://huggingface.co/datasets/georglange/qwen3-4b-molt-interpretations/'
       f'resolve/{REVISION}/prompts/cases/layer-29-0002.jsonl')
SIZE = 106705293
CHUNK = 65536
requests = []


def read_range(start, stop):
    cache = OUT / 'ranges' / f'{start}-{stop}.bin'
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        data = cache.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{stop-1}'})
        with urllib.request.urlopen(req, timeout=45) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes {start}-{stop-1}/{SIZE}'
            data = response.read()
        assert len(data) == stop-start
        cache.write_bytes(data)
    requests.append(dict(start=start, stop=stop, bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest()))
    return data


def first_complete_feature(position):
    raw = read_range(position, min(SIZE, position+CHUNK))
    lines = raw.split(b'\n')
    if position:
        lines = lines[1:]
    lines = lines[:-1]
    assert lines
    parsed = [json.loads(line) for line in lines if line]
    ids = [row['feature_id'] for row in parsed]
    assert ids == sorted(ids), 'Shard not sorted by feature; do not binary-search it'
    return ids[0]


def boundary(feature):
    low, high = 0, SIZE-CHUNK
    while high-low > CHUNK:
        middle = (low+high)//2
        if first_complete_feature(middle) < feature:
            low = middle
        else:
            high = middle
    return max(0, low-CHUNK), min(SIZE, high+2*CHUNK)


if __name__ == '__main__':
    start, _ = boundary(2375)
    _, stop = boundary(2376)
    raw = read_range(start, stop)
    lines = raw.split(b'\n')[1:-1]
    rows = [json.loads(line) for line in lines if line]
    assert any(r['feature_id'] < 2375 for r in rows)
    assert any(r['feature_id'] > 2375 for r in rows)
    target = [r for r in rows if r['feature_id'] == 2375]
    assert target and all(r['layer'] == 29 for r in target)
    output = OUT/'published-requests-T2375.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in target))
    receipt = dict(revision=REVISION, source=URL, complete_feature_bracket_verified=True,
                   requests=len(target), requests_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   range_reads=requests, bytes_read=sum(r['bytes'] for r in requests),
                   contains_ground_truth=False, contains_judge_responses=False)
    (OUT/'published-requests-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'range_reads'}), flush=True)
