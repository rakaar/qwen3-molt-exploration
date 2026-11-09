"""Reuse only the named live rental for the paired Active Fuzz judge."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('p',HERE/'provider.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
record,state=p.current()
assert record['instance_id']==59903353 and record['budget_usd']==10
assert state['actual_status']=='running' and state['dph_total']<=.30
host=state['public_ipaddr'];port=int(state['ports']['22/tcp'][0]['HostPort'])
verification=json.loads((HERE/'ssh-verification.json').read_text())
assert verification['host']==host and verification['port']==port and verification['instance_id']==59903353
known=HERE/'known_hosts'
common=['-i','/home/rka/.ssh/vastai_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o',f'UserKnownHostsFile="{known}"','-o','ConnectTimeout=15']
ssh=['ssh',*common,'-p',str(port),f'root@{host}','python3 -']
REMOTE='/workspace/t2375-active-fuzz-local-judge-20261109'
def sha(file):
    with file.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def remote(code,timeout=75):
    run=subprocess.run(ssh,input=code,capture_output=True,text=True,timeout=timeout)
    if run.returncode:raise RuntimeError(run.stderr[-2200:])
    return json.loads(run.stdout.strip().splitlines()[-1])

mode=sys.argv[1]
if mode=='deploy':
    source=PROJECT/'experiments/t2375-active-fuzz-local-judge-20261109'
    manifest=[dict(path=str(f.relative_to(source)),bytes=f.stat().st_size,sha256=sha(f)) for f in source.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='source-manifest.json']
    (source/'source-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    archive=HERE/'active-fuzz-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in manifest:tar.add(source/row['path'],arcname='t2375-active-fuzz-local-judge-20261109/'+row['path'],recursive=False)
        tar.add(source/'source-manifest.json',arcname='t2375-active-fuzz-local-judge-20261109/source-manifest.json')
    subprocess.run(['scp',*common,'-P',str(port),str(archive),f'root@{host}:/workspace/active-fuzz-source.tar.gz'],capture_output=True,check=True,timeout=60)
    result=remote(f'''import json,hashlib,subprocess,tarfile,shutil,os
from pathlib import Path
archive=Path('/workspace/active-fuzz-source.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
root=Path({REMOTE!r});assert not (root/'setup.pid').exists(),'Do not duplicate a running setup'
with tarfile.open(archive) as tar:
 for m in tar.getmembers():assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())
 tar.extractall('/workspace',filter='data')
for row in json.loads((root/'source-manifest.json').read_text())['files']:
 assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
free=shutil.disk_usage(root).free
assert free>29000000000,'Preserve existing files; insufficient room for isolated runtime and judge'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.used','--format=csv,noheader'],text=True).strip()
assert '3090' in gpu
env=os.environ.copy();env['PATH']='/opt/conda/bin:'+env.get('PATH','')
log=(root/'setup-run.log').open('w')
proc=subprocess.Popen(['timeout','5400','bash','setup.sh'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'setup.pid').write_text(str(proc.pid))
print(json.dumps({{'pid':proc.pid,'free_disk_bytes':free,'gpu':gpu,'verified_source_files':{len(manifest)}}}))
''')
    p.save('active-fuzz-run-start.json',result);print(json.dumps(result))
elif mode=='status':
    print(json.dumps(remote(f'''import json,subprocess,shutil
from pathlib import Path
root=Path({REMOTE!r});out=root/'artifacts'
log=root/'setup-run.log';lines=log.read_text(errors='replace').splitlines() if log.exists() else []
summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None
files=[f for f in (root/'judge').rglob('*') if f.is_file()] if (root/'judge').exists() else []
processes=subprocess.check_output(['ps','-eo','pid,ppid,stat,pcpu,comm'],text=True).splitlines()
safe_tail=[line[:1000] for line in lines if line.strip() and not line.lstrip().startswith('━') and not line.startswith(('Installing collected packages:','Successfully installed'))][-6:]
print(json.dumps({{'summary':{{k:summary[k] for k in ['status','baseline','candidate','corrected','regressed']}} if summary else None,'progress':progress,'tail':safe_tail,'judge_bytes':sum(f.stat().st_size for f in files),'runtime_prepared':(out/'runtime.json').exists(),'free_bytes':shutil.disk_usage(root).free,'processes':[line for line in processes if any(term in line for term in ['pip','python','timeout','bash'])]}}))
''')))
elif mode=='collect':
    result=remote(f'''import json,hashlib,tarfile
from pathlib import Path
root=Path({REMOTE!r});out=root/'artifacts';assert (out/'summary.json').exists()
manifest=json.loads((out/'transfer-manifest.json').read_text())
extra=[]
names=['environment.freeze.txt','judge-verification.json','setup-run.log','compiler-install.log','run.py','download_judge.py','setup.sh','resume-download.sh','protocol.json','install-compiler.sh']+[p.name for p in root.glob('setup*attempt*.log')]
for name in names:
 p=root/name
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 extra.append({{'path':name,'bytes':p.stat().st_size,'sha256':digest}})
archive=Path('/tmp/t2375-active-fuzz-export.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
 for row in manifest['files']:tar.add(out/row['path'],arcname=row['path'],recursive=False)
 for row in extra:tar.add(root/row['path'],arcname=row['path'],recursive=False)
with archive.open('rb') as stream:archive_hash=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps({{'files':manifest['files'],'extra':extra,'archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':archive_hash}}))
''')
    dest=PROJECT/'results/2026-11-09-t2375-active-fuzz-local-judge';dest.mkdir(exist_ok=True)
    archive=HERE/'active-fuzz-results.tar.gz'
    subprocess.run(['scp',*common,'-P',str(port),f'root@{host}:{result["archive_path"]}',str(archive)],capture_output=True,check=True,timeout=90)
    assert archive.stat().st_size==result['archive_bytes'] and sha(archive)==result['archive_sha256']
    rows={r['path']:r for r in result['files']+result['extra']}
    with tarfile.open(archive) as tar:
        assert set(tar.getnames())==set(rows)
        for member in tar.getmembers():
            assert member.isfile() and Path(member.name).name==member.name
            target=dest/member.name
            target.write_bytes(tar.extractfile(member).read())
            row=rows[member.name]
            assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    p.save('active-fuzz-backup-verification.json',dict(instance_id=59903353,verified_locally=True,files=result['files']+result['extra'],local_directory=str(dest)))
    print(json.dumps(dict(verified_locally=True,files=len(result['files'])+len(result['extra']),directory=str(dest))))
elif mode=='disk':
    print(json.dumps(remote(f'''import json,subprocess
from pathlib import Path
root=Path({REMOTE!r})
paths=[str(root/'.venv'),str(root/'judge'),'/root/.cache/huggingface','/tmp']
du=subprocess.run(['du','-sh',*paths],capture_output=True,text=True,timeout=30)
largest=sorted([{{'path':str(f.relative_to(root/'judge')),'bytes':f.stat().st_size,'allocated_bytes':f.stat().st_blocks*512}} for f in (root/'judge').rglob('*') if f.is_file()],key=lambda r:r['allocated_bytes'],reverse=True)[:4]
print(json.dumps({{'directory_sizes':du.stdout.splitlines(),'largest_judge_files':largest}}))
''')))
elif mode=='errors':
    print(json.dumps(remote(f'''import json
from pathlib import Path
lines=(Path({REMOTE!r})/'setup-run.log').read_text(errors='replace').splitlines()
print(json.dumps({{'tail':lines[-110:]}}))
''')))
elif mode=='remove-abandoned-partial':
    result=remote(f'''import json,shutil,os
from pathlib import Path
root=Path({REMOTE!r});cache=root/'judge/.cache/huggingface/download'
assert not (root/'artifacts/runtime.json').exists(),'Never alter an active scoring run'
files=[f for f in cache.glob('*.incomplete') if f.is_file()]
assert files,'No partial cache file; inspect rather than deleting'
opened=set()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\\0')
  if b'download_judge.py' not in cmd or (proc/'cwd').resolve()!=root:continue
  for fd in (proc/'fd').iterdir():
   try:opened.add(fd.resolve())
   except OSError:pass
 except (OSError,PermissionError):pass
old=[f for f in files if f.name.endswith('.7cd71843.incomplete')]
assert len(old)==1 and old[0].resolve() not in opened,'The identified old HTTP partial is not safely abandoned'
file=old[0];record={{'path':str(file.relative_to(root)),'bytes':file.stat().st_size,'cache_only':True,'not_open_by_downloader':True}}
file.unlink()
print(json.dumps({{'removed_abandoned_partial':record,'free_disk_bytes':shutil.disk_usage(root).free,'other_incomplete_files':[f.name for f in files if f!=file]}}))
''')
    p.save('active-fuzz-abandoned-partial-removal.json',result);print(json.dumps(result))
elif mode=='run-judge':
    source=PROJECT/'experiments/t2375-active-fuzz-local-judge-20261109/run.py'
    subprocess.run(['scp',*common,'-P',str(port),str(source),f'root@{host}:{REMOTE}/run.py'],capture_output=True,check=True,timeout=60)
    result=remote(f'''import json,hashlib,subprocess,os,time
from pathlib import Path
root=Path({REMOTE!r});out=root/'artifacts'
assert not (out/'audit.jsonl').exists(),'Never rerun or overwrite scoring decisions'
assert (root/'judge-verification.json').exists()
assert hashlib.sha256((root/'run.py').read_bytes()).hexdigest()=={sha(source)!r}
old=root/'setup-run.log';old.rename(root/f'setup-prejudge-attempt-{{time.time_ns()}}.log')
log=(root/'setup-run.log').open('w');env=os.environ.copy()
for name in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:env.pop(name,None)
env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
process=subprocess.Popen(['timeout','3600',str(root/'.venv/bin/python'),'-u','run.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'setup.pid').write_text(str(process.pid))
print(json.dumps({{'pid':process.pid,'verified_judge_weights_reused':True,'source_sha256':{sha(source)!r}}}))
''')
    p.save('active-fuzz-judge-start.json',result);print(json.dumps(result))
elif mode=='install-compiler':
    result=remote(f'''import json,subprocess,os
from pathlib import Path
root=Path({REMOTE!r})
assert not (root/'artifacts/audit.jsonl').exists()
script=root/'install-compiler.sh'
script.write_text('#!/bin/bash\\nset -euo pipefail\\napt-get update -qq\\nDEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gcc g++ libc6-dev make\\ngcc --version | head -1\\n')
log=(root/'compiler-install.log').open('w')
process=subprocess.Popen(['bash',str(script)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'compiler_install_pid':process.pid}}))
''')
    p.save('active-fuzz-compiler-install.json',result);print(json.dumps(result))
elif mode=='compiler-status':
    print(json.dumps(remote(f'''import json,shutil
from pathlib import Path
root=Path({REMOTE!r});log=root/'compiler-install.log'
print(json.dumps({{'gcc':shutil.which('gcc'),'gpp':shutil.which('g++'),'tail':log.read_text(errors='replace').splitlines()[-6:] if log.exists() else []}}))
''')))
elif mode in ['resume-download','start-download']:
    source=PROJECT/'experiments/t2375-active-fuzz-local-judge-20261109'
    # Interrupt only this task's verified downloader, never any instance lifecycle.
    stopped=remote(f'''import json,os,signal,time
from pathlib import Path
root=Path({REMOTE!r});assert not (root/'artifacts/runtime.json').exists()
matches=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\\0')
  if b'download_judge.py' in cmd and (proc/'cwd').resolve()==root:matches.append(int(proc.name))
 except (OSError,PermissionError):pass
assert len(matches)<=1,matches
if {mode!r}=='resume-download':assert len(matches)==1,matches
if matches:os.kill(matches[0],signal.SIGTERM)
time.sleep(2)
print(json.dumps({{'terminated_task_downloader_pid':matches[0] if matches else None,'gpu_lifecycle_changed':False}}))
''')
    for name in ['download_judge.py','resume-download.sh']:
        subprocess.run(['scp',*common,'-P',str(port),str(source/name),f'root@{host}:{REMOTE}/{name}'],capture_output=True,check=True,timeout=60)
    result=remote(f'''import json,hashlib,subprocess,os,time
from pathlib import Path
root=Path({REMOTE!r})
assert hashlib.sha256((root/'download_judge.py').read_bytes()).hexdigest()=={sha(source/'download_judge.py')!r}
assert hashlib.sha256((root/'resume-download.sh').read_bytes()).hexdigest()=={sha(source/'resume-download.sh')!r}
old=root/'setup-run.log';old.rename(root/f'setup-download-attempt-{{time.time_ns()}}.log')
log=(root/'setup-run.log').open('w');env=os.environ.copy();env['PATH']='/opt/conda/bin:'+env.get('PATH','')
process=subprocess.Popen(['timeout','4800','bash','resume-download.sh'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'setup.pid').write_text(str(process.pid))
print(json.dumps({{'pid':process.pid,'transport':'parallel Xet; chunk cache disabled','partial_download_preserved':True}}))
''')
    p.save('active-fuzz-download-resume.json',dict(**stopped,**result));print(json.dumps(result))
else:raise ValueError('Supported: deploy/status/collect')
