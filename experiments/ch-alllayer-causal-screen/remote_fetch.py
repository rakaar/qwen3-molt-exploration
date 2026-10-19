"""Recreate byte-identical selected NPZs directly from pinned public tensor ranges."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'weights';OUT.mkdir(exist_ok=True)
M=json.loads((ROOT/'weights-manifest-expected.json').read_text())

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fetch(url,start,end):
    for attempt in range(5):
        try:
            req=Request(url+f'?ch_cloud_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
            with urlopen(req,timeout=60) as r:
                assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                b=r.read(end-start+2);assert len(b)==end-start+1;return b
        except Exception:
            if attempt==4:raise
            time.sleep(attempt+1)

def one(item):
    l=item['layer'];path=OUT/item['file']
    if path.exists() and sha(path)==item['sha256']:return
    with np.load(ROOT/'gate-subsets'/item['file']) as a:data={k:a[k] for k in a.files}
    def feature_job(feature):
        tensors={}
        for role,rec in zip(['U','V'],feature['records']):
            start,end=rec['range'];end+=1
            chunks=[(s,min(s+4*1024*1024,end)-1) for s in range(start,end,4*1024*1024)]
            # Each single raw tensor is at most 5 MiB in this model.
            blob=b''.join(fetch(item['url'],a,b) for a,b in chunks)
            assert hashlib.sha256(blob).hexdigest()==rec['sha256'] and len(blob)==rec['bytes']
            if rec['source_dtype']=='BF16':arr=(np.frombuffer(blob,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
            else:arr=np.frombuffer(blob,dtype='<f4' if rec['source_dtype']=='F32' else '<f2').astype(np.float32)
            tensors[f"{role}_{feature['transform']}"]=arr.reshape(rec['shape']).copy()
        return tensors
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for tensors in pool.map(feature_job,item['features']):data.update(tensors)
    np.savez_compressed(path,**data)
    assert sha(path)==item['sha256'],f'Local/remote NPZ mismatch layer {l}'
    print('EXACT_WEIGHT_LAYER',l,path.stat().st_size,flush=True)

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,M['layers']))
    (OUT/'manifest.json').write_text(json.dumps(M,indent=2)+'\n')
    print('ALL_36_LAYERS_IDENTICAL_TO_LOCAL',flush=True)

if __name__=='__main__':main()
