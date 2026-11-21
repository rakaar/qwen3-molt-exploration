"""Fetch the immutable native model using bounded byte ranges and verify every file."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,time,threading
import requests
spec=json.loads(Path('expected-base-verification.json').read_text())
out=Path('model');out.mkdir(exist_ok=True)
base_url=os.environ.get('MODEL_BASE_URL',f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}")
chunks=[]
for row in spec['files']:
    dest=out/row['file']
    if dest.exists() and dest.stat().st_size==row['bytes'] and hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==row['sha256']: continue
    dest.touch()
    n=row['bytes']
    for a in range(0,n,32*1024**2):chunks.append((dest,a,min(n,a+32*1024**2)-1,n))
def fetch(args):
    p,a,b,n=args;url=f"{base_url}/{p.name}"
    for attempt in range(3):
        try:
            partial=bool(a or b<n-1)
            with requests.get(url,params={'slice':a,'attempt':attempt},headers={'Range':f'bytes={a}-{b}'} if partial else {},stream=True,timeout=(12,55)) as r:
                r.raise_for_status()
                if partial: assert r.status_code==206 and r.headers['Content-Range']==f'bytes {a}-{b}/{n}'
                total=0
                with p.open('r+b',buffering=0) as f:
                    f.seek(a)
                    for buf in r.iter_content(1024*1024):
                        total+=len(buf);assert total<=b-a+1;f.write(buf)
                assert total==b-a+1
            return
        except Exception as err:
            print('retry',p.name,a,type(err).__name__,flush=True)
            if attempt==2: raise
            time.sleep(1+attempt)
with ThreadPoolExecutor(max_workers=12) as pool:
    for i,_ in enumerate(pool.map(fetch,chunks),1):
        if i%20==0: print('fetched chunks',i,len(chunks),flush=True)
for row in spec['files']:
    p=out/row['file']
    with p.open('rb') as f: assert p.stat().st_size==row['bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],p.name
Path('model-verified.json').write_text(json.dumps(spec)+'\n')
print('model verified',flush=True)
