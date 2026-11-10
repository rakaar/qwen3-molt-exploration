from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,signal,subprocess,time,urllib.request,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TOTAL=19743758648
EXPECTED='1dcee3059a4ee8ba6d872ea38aaa61051c3b34a5863b35c96504b5a64f4c7f24'
URL='https://huggingface.co/vrfai/Qwen3.8-27B-NVFP4/resolve/1561c70b22784dba7c6943b3271fc5f185adda01/model.safetensors?download=true'
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
assert b'download_judge.py' in Path('/proc/1302/cmdline').read_bytes()
assert Path('/proc/1302/cwd').resolve()==ROOT
os.kill(1302,signal.SIGSTOP)
partial=next((ROOT/'judge').rglob('*.incomplete'));start=partial.stat().st_size
assert 18_000_000_000<start<TOTAL
assert shutil.disk_usage(ROOT).free>2*(TOTAL-start)+1_000_000_000
parts=ROOT/'tail-parts';parts.mkdir(exist_ok=False)
step=32*1024*1024;ranges=[(i,a,min(TOTAL,a+step)) for i,a in enumerate(range(start,TOTAL,step))]
save('tail-repair-start.json',dict(original_download_pid=1302,original_partial_bytes=start,expected_total_bytes=TOTAL,ranges=len(ranges),scope='Public model weights only; frozen judge prompts and runtime unchanged'))
ready=[]
def fetch(item):
 i,a,b=item;p=parts/f'{i:03d}.part'
 for attempt in range(3):
  try:
   req=urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b-1}'})
   with urllib.request.urlopen(req,timeout=40) as r:
    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b-1}/{TOTAL}'
    with p.open('wb') as f:
     left=b-a
     while left:
      chunk=r.read(min(left,1024*1024));assert chunk
      f.write(chunk);left-=len(chunk)
     assert not r.read(1)
   assert p.stat().st_size==b-a
   return dict(index=i,start=a,end=b,bytes=b-a,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  except Exception as e:
   if attempt==2:raise RuntimeError('Tail range failed: '+type(e).__name__) from None
   time.sleep(1+attempt)
try:
 with ThreadPoolExecutor(max_workers=8) as pool:
  for f in as_completed([pool.submit(fetch,item) for item in ranges]):
   ready.append(f.result());save('tail-repair-progress.json',dict(completed=len(ready),total=len(ranges),downloaded_bytes=sum(r['bytes'] for r in ready)))
 with partial.open('r+b') as f:
  assert f.seek(0,2)==start
  for row in sorted(ready,key=lambda r:r['index']):
   assert f.tell()==row['start']
   with (parts/f"{row['index']:03d}.part").open('rb') as chunk:shutil.copyfileobj(chunk,f,1024*1024)
  f.flush();os.fsync(f.fileno())
 assert partial.stat().st_size==TOTAL
 with partial.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert digest==EXPECTED,'Full weight checksum mismatch'
 save('tail-repair-verification.json',dict(sha256=digest,bytes=TOTAL,all_ranges_validated=True,parts=sorted(ready,key=lambda r:r['index']),verified_unix=time.time()))
 os.kill(1302,signal.SIGKILL)
 os.replace(partial,ROOT/'judge/model.safetensors')
 resume='''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY VAST_API_KEY VAST_AI_KEY
export HF_HUB_DISABLE_XET=0 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
.venv/bin/python -u download_judge.py
.venv/bin/python -u run.py
'''
 (ROOT/'resume-verified.sh').write_text(resume)
 log=(ROOT/'setup-run.log').open('a')
 p=subprocess.Popen(['timeout','900','bash','resume-verified.sh'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save('verified-resume-start.json',dict(pid=p.pid,weight_hash_verified=True))
 print('TAIL_REPAIR_VERIFIED_AND_RUN_RESUMED',flush=True)
except BaseException as e:
 if partial.exists():
  with partial.open('r+b') as f:f.truncate(start)
 try:os.kill(1302,signal.SIGCONT)
 except ProcessLookupError:pass
 save('tail-repair-failure.json',dict(error_type=type(e).__name__,restored_original_bytes=start))
 raise
