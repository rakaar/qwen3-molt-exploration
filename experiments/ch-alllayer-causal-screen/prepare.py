"""Pinned, gate-only checkpoint extraction; low-rank weights fetched after activity scan."""
import concurrent.futures
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
CACHE = ROOT / 'gates'
CACHE.mkdir(exist_ok=True)
OLD = ROOT.parent / 'active-small-hot-controls'
GROUPS = [(0,80,512),(80,240,256),(240,560,128),(560,1200,64),(1200,2480,32)]

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def fetch(url, start, end):
    for attempt in range(5):
        try:
            req = Request(url + f'?ch_screen_range={start}-{end}', headers={'Range': f'bytes={start}-{end}'})
            with urlopen(req, timeout=60) as r:
                assert r.status == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                blob = r.read(end-start+2)
                assert len(blob) == end-start+1
                return blob
        except Exception:
            if attempt == 4: raise
            time.sleep(attempt+1)

def array(url, base, spec, key, layer=None, feature=None):
    start, end = [base+n for n in spec['data_offsets']]
    shape, dtype = spec['shape'], spec['dtype']
    size = {'F32':4, 'F16':2, 'BF16':2}[dtype]
    if layer is not None:
        assert shape == [36,2560]
        start += layer*2560*size; end = start+2560*size; shape = [2560]
    if feature is not None:
        n = (end-start)//shape[0]; start += feature*n; end = start+n; shape = shape[1:]
    chunks = [(a,min(a+4*1024*1024,end)-1) for a in range(start,end,4*1024*1024)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        blob = b''.join(pool.map(lambda pair:fetch(url,*pair),chunks))
    if dtype == 'BF16': value = (np.frombuffer(blob,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
    else: value = np.frombuffer(blob,dtype='<f4' if dtype=='F32' else '<f2').astype(np.float32)
    return value.reshape(shape).copy(), {'key':key,'range':[start,end-1],'shape':shape,
        'source_dtype':dtype,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}

def layer_job(layer):
    url = f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-{layer:02d}/checkpoint.safetensors"
    path = CACHE/f'layer-{layer}-gates.npz'
    recpath = CACHE/f'layer-{layer}-records.json'
    headpath = CACHE/f'layer-{layer}-header.json'
    oldpath = OLD/'cache'/path.name
    if not path.exists() and oldpath.exists():
        oldp = json.loads((OLD/'protocol.json').read_text())
        assert oldp['molt_repo']==P['molt_repo'] and oldp['molt_revision']==P['molt_revision']
        os.link(oldpath,path)
        recpath.write_text((OLD/'cache'/f'layer-{layer}-gate-records.json').read_text())
        headpath.write_text((OLD/'cache'/f'layer-{layer}-header.json').read_text())
    if not path.exists():
        first = fetch(url,0,65535); n = struct.unpack('<Q',first[:8])[0]; assert n<65528
        head = json.loads(first[8:8+n]); headpath.write_text(json.dumps(head,indent=2)+'\n')
        data, records = {}, []
        for short,key in [('encoder','e.weight'),('bias','e.bias'),('threshold','nonlinearity.theta'),
                          ('mean','input_standardizer.mean'),('std','input_standardizer.std'),('outstd','output_standardizer.std')]:
            data[short], rec = array(url,8+n,head[key],key,layer=layer if short in ['mean','std','outstd'] else None)
            records.append(rec)
        np.savez_compressed(path,**data);recpath.write_text(json.dumps(records,indent=2)+'\n')
    with np.load(path) as d:
        assert d['encoder'].shape==(2480,2560) and len(d.files)==6
        assert (d['std']>0).all()
    print('GATE_READY',layer,flush=True)
    return {'layer':layer,'url':url,'file':path.name,'sha256':sha(path),
            'records':json.loads(recpath.read_text()),'header_file':headpath.name}

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(layer_job,range(36)))
    (CACHE/'manifest.json').write_text(json.dumps({'repo':P['molt_repo'],'revision':P['molt_revision'],
        'layers':rows},indent=2)+'\n')
    print('ALL_36_GATE_LAYERS_READY',flush=True)

if __name__=='__main__':main()
