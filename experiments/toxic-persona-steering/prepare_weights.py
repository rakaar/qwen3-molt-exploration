"""Download only public, pinned tensor slices for the selected MOLTs."""
import concurrent.futures
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
W = ROOT / 'weights'
W.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, start, end):
    for attempt in range(3):
        try:
            response = requests.get(url + f'?range={start}-{end}&n={time.time_ns()}',
                                    headers={'Range': f'bytes={start}-{end}'}, timeout=35)
            response.raise_for_status()
            assert response.status_code == 206
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            assert len(response.content) == end - start + 1
            return response.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare(feature):
    layer, index = feature['layer'], feature['transform']
    base = f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-{layer:02d}"
    response = requests.get(base + '/config.json', timeout=35)
    response.raise_for_status()
    config = response.json()
    assert config['layer'] == layer and config['d_acts'] == 2560
    assert config['input_location'] == 'pre_norm' and config['output_location'] == 'raw'
    (W / f'layer-{layer}-config.json').write_text(json.dumps(config, indent=2) + '\n')
    url = base + '/checkpoint.safetensors'
    first = fetch(url, 0, 65535)
    size = struct.unpack('<Q', first[:8])[0]
    assert size < len(first) - 8
    header = json.loads(first[8:8 + size])
    (W / f'layer-{layer}-header.json').write_text(json.dumps(header, indent=2) + '\n')
    arrays, records = {}, []

    def read(key, name, flat_start=0, length=None):
        spec = header[key]
        width = {'F32': 4, 'BF16': 2}[spec['dtype']]
        count = int(np.prod(spec['shape']))
        if length is None:
            length = count
        assert 0 <= flat_start and flat_start + length <= count
        start = 8 + size + spec['data_offsets'][0] + width * flat_start
        end = start + width * length - 1
        blob = fetch(url, start, end)
        if spec['dtype'] == 'BF16':
            values = (np.frombuffer(blob, dtype='<u2').astype('<u4') << 16).view('<f4').copy()
        else:
            values = np.frombuffer(blob, dtype='<f4').copy()
        assert np.isfinite(values).all()
        arrays[name] = values
        records.append({'source_tensor': key, 'source_shape': spec['shape'],
                        'source_dtype': spec['dtype'], 'flat_start': flat_start,
                        'elements': length, 'byte_range_inclusive': [start, end],
                        'slice_sha256': hashlib.sha256(blob).hexdigest()})

    read('e.weight', 'e', index * 2560, 2560)
    read('e.bias', 'bias', index, 1)
    read('nonlinearity.theta', 'theta', index, 1)
    for key, name in [('input_standardizer.mean', 'input_mean'),
                      ('input_standardizer.std', 'input_std'),
                      ('output_standardizer.std', 'output_std')]:
        read(key, name, layer * 2560, 2560)
    groups = [(0, 80, 512), (80, 240, 256), (240, 560, 128),
              (560, 1200, 64), (1200, 2480, 32)]
    group, (lo, hi, rank) = next((j, g) for j, g in enumerate(groups) if g[0] <= index < g[1])
    assert rank == feature['rank']
    for prefix, name, shape in [('Us', 'U', (rank, 2560)), ('Vs', 'V', (2560, rank))]:
        assert header[f'{prefix}.{group}']['shape'] == [hi - lo, *shape]
        read(f'{prefix}.{group}', name, (index - lo) * rank * 2560, rank * 2560)
        arrays[name] = arrays[name].reshape(shape)
    assert (arrays['input_std'] > 0).all() and (arrays['output_std'] > 0).all()
    target = W / f'L{layer}-T{index}.npz'
    np.savez_compressed(target, **arrays)
    row = {**feature, 'file': target.name, 'bytes': target.stat().st_size,
           'sha256': sha(target), 'source_config_sha256': sha(W / f'layer-{layer}-config.json'),
           'source_header_sha256': sha(W / f'layer-{layer}-header.json'), 'records': records,
           'theta': float(arrays['theta'][0]), 'checkpoint_url': url}
    print('READY', target.name, row['bytes'], 'threshold', row['theta'], flush=True)
    return row


if __name__ == '__main__':
    response = requests.get(f"https://huggingface.co/api/models/{P['molt_repo']}/revision/{P['molt_revision']}?blobs=true", timeout=35)
    response.raise_for_status()
    info = response.json()
    assert info['sha'] == P['molt_revision']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(prepare, P['features']))
    for row in rows:
        item = next(s for s in info['siblings'] if s['rfilename'] == f"layer-{row['layer']:02d}/checkpoint.safetensors")
        row['whole_checkpoint_lfs_sha256'] = item['lfs']['sha256']
        row['whole_checkpoint_bytes'] = item['size']
    (W / 'manifest.json').write_text(json.dumps({'repo': P['molt_repo'], 'revision': P['molt_revision'],
        'slice_hashes_verified': True, 'whole_checkpoint_not_downloaded': True, 'features': rows}, indent=2) + '\n')
    print('Pinned selected assets ready.', flush=True)
