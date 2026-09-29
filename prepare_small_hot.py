"""Prepare selected weights and verbatim upstream intervention helpers."""
import ast
import concurrent.futures
import hashlib
import json
import struct
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
SOURCES = ROOT / 'research/intervention-examples'
OUT = ROOT / 'results/small-hot-reproduction'
OUT.mkdir(parents=True, exist_ok=True)
recipe = json.loads((SOURCES / 'small-to-hot-reproduction-recipe.json').read_text())
(OUT / 'recipe.json').write_text(json.dumps(recipe, indent=2))
rev = '94d176260ac39ce2f882b8b09aba8c118df29bb3'
features = [dict(f, group=g) for g, fs in recipe['selection'].items() for f in fs]

def fetch(url, start, stop):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url+f'?range={start}-{stop}', headers={'Range':f'bytes={start}-{stop}'})
            with urllib.request.urlopen(req, timeout=35) as r:
                assert r.status == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{stop}/')
                b = r.read(stop-start+2)
                assert len(b) == stop-start+1
                return b
        except Exception:
            if attempt == 3: raise
            time.sleep(attempt+1)

def layer_job(layer):
    url=f'https://huggingface.co/mwhanna/qwen3-4b-transcoders/resolve/{rev}/layer_{layer}.safetensors'
    first=fetch(url,0,65535)
    n=struct.unpack('<Q',first[:8])[0]; header=json.loads(first[8:8+n]);base=8+n
    fs=[f for f in features if f['layer']==layer]
    assert header['W_enc']['shape']==header['W_dec']['shape']==[163840,2560]
    assert all(header[k]['dtype']=='BF16' for k in ['W_enc','W_dec','b_enc'])
    bs,be=header['b_enc']['data_offsets'];bias=fetch(url,base+bs,base+be-1)
    payload=[]
    for f in fs:
        index=f['feature']
        for key in ['W_enc','W_dec']:
            a=base+header[key]['data_offsets'][0]+index*5120
            payload.append(fetch(url,a,a+5119))
        payload.append(bias[index*2:index*2+2])
    blob=b''.join(payload);name=f'layer_{layer}_selected.bin';(OUT/name).write_bytes(blob)
    print('FETCHED',layer,len(fs),flush=True)
    return {'layer':layer,'features':fs,'file':name,'sha256':hashlib.sha256(blob).hexdigest(),
            'download_bytes':65536+len(bias)+len(fs)*10240}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    layers=list(pool.map(layer_job,sorted({f['layer'] for f in features})))
(OUT/'weights-manifest.json').write_text(json.dumps({'revision':rev,'layers':layers,'download_bytes':sum(l['download_bytes'] for l in layers)},indent=2))

# Retain the upstream function/class source text exactly; only the import header
# is adapted to avoid installing unrelated graph-visualization dependencies.
selections={
 'wdk-local_replacement_model.py':['_repeat_kv','_make_frozen_attn_forward','_make_frozen_rmsnorm_hook'],
 'wdk-interventions.py':['FeatureIntervention','AblationResult','_make_base_mlp_hook','_steer_base_model'],
}
body=['"""Verbatim helpers from wdk0082/llm-circuits at '+recipe['observed_revision']+'."""',
      'from __future__ import annotations','from dataclasses import dataclass, field',
      'from typing import Any','import torch','from torch import Tensor, nn']
provenance=[]
for filename,names in selections.items():
    source=(SOURCES/filename).read_text();tree=ast.parse(source);lines=source.splitlines(keepends=True)
    for node in tree.body:
        if getattr(node,'name',None) in names:
            start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
            exact=''.join(lines[start:node.end_lineno]);body.append(exact)
            provenance.append({'file':filename,'name':node.name,'sha256':hashlib.sha256(exact.encode()).hexdigest()})
assert len(provenance)==sum(map(len,selections.values()))
(ROOT/'small_hot_upstream.py').write_text('\n\n'.join(body)+'\n')
(OUT/'source-functions.json').write_text(json.dumps(provenance,indent=2))
print('PREPARATION_OK',flush=True)
