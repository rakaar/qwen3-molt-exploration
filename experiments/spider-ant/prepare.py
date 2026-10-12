"""Download pinned public weights and selected MOLT slices; never handles cloud keys."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import requests
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
WEIGHTS = ROOT / 'weights'
WEIGHTS.mkdir(exist_ok=True)


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(url, start, end):
    for attempt in range(4):
        try:
            with requests.get(url + f'?range={start}-{end}', headers={'Range': f'bytes={start}-{end}'}, timeout=60, stream=True) as r:
                r.raise_for_status()
                assert r.status_code == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                blob = r.raw.read(end - start + 2)
                assert len(blob) == end - start + 1
                return blob
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def molt_layer(layer):
    record_file = WEIGHTS / f'molt-{layer}-slice-record.json'
    if record_file.exists():
        previous = json.loads(record_file.read_text())
        if digest(WEIGHTS / previous['file']) == previous['sha256']:
            return previous
    ids = sorted(x['id'] for x in P['features'] if x['layer'] == layer)
    prefix = f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-{layer:02d}"
    c = requests.get(prefix + '/config.json', timeout=40)
    c.raise_for_status()
    config = c.json()
    assert config['layer'] == layer and config['input_location'] == 'pre_norm' and config['output_location'] == 'raw'
    (WEIGHTS / f'molt-{layer}-config.json').write_text(json.dumps(config, indent=2))
    url = prefix + '/checkpoint.safetensors'
    first = fetch(url, 0, 65535)
    n = struct.unpack('<Q', first[:8])[0]
    assert n < 65528
    header = json.loads(first[8:8+n])
    (WEIGHTS / f'molt-{layer}-header.json').write_text(json.dumps(header, indent=2))
    tensors, records = {}, []
    def read(key, output_key=None, index=None):
        spec = header[key]
        shape = spec['shape']
        start, end = [8+n+o for o in spec['data_offsets']]
        if index is not None:
            size = (end-start) // shape[0]
            start += size * index
            end = start + size
            shape = shape[1:]
        blob = fetch(url, start, end-1)
        dtype = {'F32': torch.float32, 'BF16': torch.bfloat16}[spec['dtype']]
        name = output_key or key
        tensors[name] = torch.frombuffer(bytearray(blob), dtype=dtype).reshape(shape).clone()
        records.append({'source_tensor': key, 'index': index, 'saved_tensor': name, 'shape': shape,
                        'source_dtype': spec['dtype'], 'range': [start, end-1], 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()})
    for key in ['e.weight','e.bias','nonlinearity.theta','nonlinearity.bandwidth','input_standardizer.mean','input_standardizer.std','output_standardizer.mean','output_standardizer.std']:
        read(key)
    groups = [(0,80,512),(80,240,256),(240,560,128),(560,1200,64),(1200,2480,32)]
    for index in ids:
        group = next(g for g,(lo,hi,_) in enumerate(groups) if lo <= index < hi)
        offset = index - groups[group][0]
        read(f'Us.{group}',f'U_{index}',offset)
        read(f'Vs.{group}',f'V_{index}',offset)
    file = WEIGHTS / f'molt-{layer}-selected.safetensors'
    save_file(tensors,file)
    print('MOLT_SLICES_READY',layer,ids,file.stat().st_size,flush=True)
    result = {'layer': layer, 'ids': ids, 'file': file.name, 'sha256': digest(file), 'records': records}
    record_file.write_text(json.dumps(result, indent=2))
    return result


def base_download():
    target = ROOT / 'models/qwen3-4b'
    snapshot_download(P['base_repo'], revision=P['base_revision'], local_dir=target,
                      allow_patterns=['*.json','*.safetensors','*.txt','LICENSE*'], token=False,max_workers=4)
    info = HfApi(token=False).model_info(P['base_repo'],revision=P['base_revision'],files_metadata=True)
    rows=[]
    for f in info.siblings:
        path=target/f.rfilename
        if not path.is_file(): continue
        sha=digest(path)
        assert path.stat().st_size==f.size
        if f.lfs: assert sha==f.lfs.sha256
        rows.append({'file':f.rfilename,'bytes':f.size,'sha256':sha,'hub_hash_verified':bool(f.lfs)})
    print('BASE_VERIFIED',sum(r['bytes'] for r in rows),flush=True)
    return {'repo':P['base_repo'],'revision':P['base_revision'],'files':rows}


def full22_download():
    target=ROOT/'models/molt'
    snapshot_download(P['molt_repo'],revision=P['molt_revision'],local_dir=target,
                      allow_patterns=['layer-22/*'],token=False,max_workers=2)
    info=HfApi(token=False).model_info(P['molt_repo'],revision=P['molt_revision'],files_metadata=True)
    rows=[]
    for f in info.siblings:
        if not f.rfilename.startswith('layer-22/'):continue
        path=target/f.rfilename;sha=digest(path);assert path.stat().st_size==f.size
        if f.lfs:assert sha==f.lfs.sha256
        rows.append({'file':f.rfilename,'bytes':f.size,'sha256':sha,'hub_hash_verified':bool(f.lfs)})
    print('FULL_MOLT22_VERIFIED',sum(r['bytes'] for r in rows),flush=True)
    return {'repo':P['molt_repo'],'revision':P['molt_revision'],'files':rows}


if __name__=='__main__':
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        b=pool.submit(base_download)
        layers=sorted({x['layer'] for x in P['features']})
        rows=list(pool.map(molt_layer,layers))
        manifest={'base':b.result(),'selected_molt':rows,'seconds':time.monotonic()-started}
    (OUT/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
    print('ALL_WEIGHTS_VERIFIED',flush=True)
