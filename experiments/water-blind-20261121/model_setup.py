"""Fetch pinned files anonymously and validate SHA-256."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,time
import requests
base=json.loads(Path('expected-base-verification.json').read_text())
out=Path('model');out.mkdir(exist_ok=True)
def fetch(r):
 p=out/r['file']
 for attempt in range(4):
  if p.exists() and p.stat().st_size==r['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==r['sha256']:return
  url=f"https://huggingface.co/{base['repo']}/resolve/{base['revision']}/{r['file']}"
  try:
   h=hashlib.sha256()
   with requests.get(url,stream=True,timeout=(20,90),params={'download':'true','attempt':attempt}) as response:
    response.raise_for_status()
    with p.open('wb') as f:
     for chunk in response.iter_content(1024*1024): f.write(chunk);h.update(chunk)
   assert p.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256']
   print('verified',r['file'],flush=True);return
  except Exception as e:
   print(r['file'],type(e).__name__,flush=True)
   if attempt==3:raise
   time.sleep(2)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,base['files']))
Path('model-verified.json').write_text(json.dumps(base)+'\n')
