"""Fetch only selected encoder rows and biases, using validated HTTP byte ranges."""
import concurrent.futures
import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / 'results/texas-features'
REV = '94d176260ac39ce2f882b8b09aba8c118df29bb3'
features = {}
for offset in [0, 20, 40, 60]:
    for item in json.loads((OUT / f'label-search-{offset}.json').read_text())['results']:
        if item['description'].strip().lower() != 'texas':
            continue
        layer, index = int(item['layer'].split('-')[0]), int(item['index'])
        features[layer, index] = {
            'layer': layer, 'index': index, 'label': item['description'],
            'label_model': item['explanationModelName'],
            'url': f'https://www.neuronpedia.org/qwen3-4b/{layer}-transcoder-hp/{index}',
        }


def fetch(url, start, stop):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url + f'?range={start}-{stop}',
                                         headers={'Range': f'bytes={start}-{stop}'})
            with urllib.request.urlopen(req, timeout=40) as response:
                assert response.status == 206
                assert response.headers['Content-Range'].startswith(f'bytes {start}-{stop}/')
                data = response.read(stop - start + 2)
                assert len(data) == stop - start + 1
                return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def layer_job(layer):
    url = f'https://huggingface.co/mwhanna/qwen3-4b-transcoders/resolve/{REV}/layer_{layer}.safetensors'
    first = fetch(url, 0, 65535)
    size = struct.unpack('<Q', first[:8])[0]
    assert size < 65528
    header = json.loads(first[8:8+size])
    enc, bias = header['W_enc'], header['b_enc']
    assert enc['dtype'] == bias['dtype'] == 'BF16'
    assert enc['shape'] == [163840, 2560]
    assert set(header) == {'W_enc', 'W_dec', 'b_enc', 'b_dec'}
    base = 8 + size
    items = [features[k] for k in sorted(features) if k[0] == layer]
    bias_start, bias_end = bias['data_offsets']
    all_bias = fetch(url, base+bias_start, base+bias_end-1)
    rows, biases = [], []
    for item in items:
        index = item['index']
        start = base + enc['data_offsets'][0] + index * 2560 * 2
        rows.append(fetch(url, start, start + 2560*2-1))
        biases.append(all_bias[index*2:index*2+2])
    blob = b''.join(rows) + b''.join(biases)
    filename = f'layer_{layer}_texas.bin'
    (OUT / filename).write_bytes(blob)
    result = {'layer': layer, 'features': items, 'file': filename,
              'sha256': hashlib.sha256(blob).hexdigest(), 'header': header,
              'download_bytes': 65536 + len(all_bias) + len(rows)*5120}
    print(f'Fetched layer {layer}: {len(items)} candidates', flush=True)
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    layers = list(pool.map(layer_job, sorted({k[0] for k in features})))
manifest = {'repository': 'mwhanna/qwen3-4b-transcoders', 'revision': REV,
            'selection': 'Unique exact Texas labels from first 80 semantic-search results; not exhaustive',
            'features_count': len(features), 'layers': layers,
            'download_bytes': sum(x['download_bytes'] for x in layers)}
(OUT / 'encoder-manifest.json').write_text(json.dumps(manifest, indent=2))
print('DONE', manifest['features_count'], 'features;', manifest['download_bytes'], 'bytes', flush=True)
