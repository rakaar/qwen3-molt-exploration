"""Fetch only gate tensors from pinned MOLT checkpoints, with checked HTTP ranges."""
import hashlib
import json
import struct
import time
from pathlib import Path
import requests
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/small-hot-molt-gates-21-22'
OUT.mkdir(parents=True, exist_ok=True)
REV = 'eed10292a96521837f5d65b36fdfa3bf5bea0037'
KEYS = ['e.weight', 'e.bias', 'nonlinearity.theta', 'nonlinearity.bandwidth',
        'input_standardizer.mean', 'input_standardizer.std']

def fetch(url, start, end):
    for attempt in range(4):
        try:
            with requests.get(url + f'?range={start}-{end}', headers={'Range': f'bytes={start}-{end}'},
                              timeout=60, stream=True) as r:
                r.raise_for_status()
                assert r.status_code == 206
                assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                blob = r.raw.read(end-start+2)
                assert len(blob) == end-start+1
                return blob
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)

manifest = {'repo': 'georglange/qwen3-4b-molt', 'revision': REV, 'layers': []}
for layer in [21, 22]:
    prefix = f'https://huggingface.co/georglange/qwen3-4b-molt/resolve/{REV}/layer-{layer}'
    config_response = requests.get(prefix+'/config.json', timeout=40)
    config_response.raise_for_status()
    config = config_response.json()
    assert config['layer'] == layer and config['input_location'] == 'pre_norm'
    assert config['target_model'] == 'Qwen/Qwen3-4B'
    (OUT/f'layer-{layer}-config.json').write_text(json.dumps(config, indent=2)+'\n')
    url = prefix + '/checkpoint.safetensors'
    first = fetch(url, 0, 65535)
    header_size = struct.unpack('<Q', first[:8])[0]
    assert header_size < len(first)-8
    header = json.loads(first[8:8+header_size])
    (OUT/f'layer-{layer}-header.json').write_text(json.dumps(header, indent=2)+'\n')
    tensors, records = {}, []
    for key in KEYS:
        spec = header[key]
        start, end = [8+header_size+n for n in spec['data_offsets']]
        blob = fetch(url, start, end-1)
        dtype = {'F32': torch.float32, 'BF16': torch.bfloat16}[spec['dtype']]
        tensors[key] = torch.frombuffer(bytearray(blob), dtype=dtype).reshape(spec['shape']).clone()
        records.append({'key': key, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(),
                        'dtype': spec['dtype'], 'shape': spec['shape'], 'range': [start, end-1]})
    filename = f'layer-{layer}-gates.safetensors'
    save_file(tensors, OUT/filename)
    manifest['layers'].append({'layer': layer, 'file': filename, 'tensors': records,
        'sha256': hashlib.sha256((OUT/filename).read_bytes()).hexdigest(),
        'download_bytes': 65536+len(config_response.content)+sum(x['bytes'] for x in records)})
    (OUT/'gate-weights-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('GATE_WEIGHTS_READY', layer, 'bytes', manifest['layers'][-1]['download_bytes'], flush=True)
print('SELECTED_GATE_DOWNLOAD_OK', flush=True)
