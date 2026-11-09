"""Run the frozen lookahead diagnostic on the same authorized SPAR instance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('provider',HERE/'provider.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
record,state=p.current()
assert record['account_id']==710735 and record['instance_id']==59903353 and record['budget_usd']==10
assert state['actual_status']=='running' and state['dph_total']<=.30
host=state['public_ipaddr'];port=int(state['ports']['22/tcp'][0]['HostPort'])
verified=json.loads((HERE/'ssh-verification.json').read_text())
assert verified['host']==host and verified['port']==port
known=HERE/'known_hosts'
common=['-i','/home/rka/.ssh/vastai_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o',f'UserKnownHostsFile="{known}"','-o','ConnectTimeout=15']
ssh=['ssh',*common,'-p',str(port),f'root@{host}','python3 -']
REMOTE='/workspace/t965-failed-cases-20261110'
JUDGE='/workspace/t2375-active-fuzz-local-judge-20261109'
SOURCE=PROJECT/'experiments/t965-failed-cases-20261110'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def remote(code,timeout=90):
    result=subprocess.run(ssh,input=code,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])

from datetime import datetime
authorization=json.loads((HERE/'t965-diagnostic-authorization.json').read_text())
assert authorization['user_reply']=='Approve restart and stop' and authorization['additional_cap_usd']==.25
deadline=datetime.fromisoformat(authorization['authorized_at_utc']).timestamp()+1800
mode=sys.argv[1]
if mode=='deploy':
    files=[dict(path=str(f.relative_to(SOURCE)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(SOURCE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='source-manifest.json']
    (SOURCE/'source-manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    archive=HERE/'t965-diagnostic-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in files:tar.add(SOURCE/row['path'],arcname='t965-failed-cases-20261110/'+row['path'],recursive=False)
        tar.add(SOURCE/'source-manifest.json',arcname='t965-failed-cases-20261110/source-manifest.json')
    subprocess.run(['scp',*common,'-P',str(port),str(archive),f'root@{host}:/workspace/t965-diagnostic-source.tar.gz'],capture_output=True,check=True,timeout=60)
    result=remote(f'''import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
root=Path({REMOTE!r});judge=Path({JUDGE!r})
assert not (root/'run-start.json').exists(),'Never duplicate a run'
assert (judge/'.venv/bin/python').is_file() and (judge/'judge-verification.json').is_file()
assert time.time()+600<{deadline},'Do not approach the authorized additional spending cap'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>22000,'Another GPU workload may be active; do not interfere'
archive=Path('/workspace/t965-diagnostic-source.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
with tarfile.open(archive) as tar:
 for member in tar.getmembers():assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts and (member.isfile() or member.isdir())
 tar.extractall('/workspace',filter='data')
for row in json.loads((root/'source-manifest.json').read_text())['files']:
 assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
env=os.environ.copy()
for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:env.pop(key,None)
env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
log=(root/'run.log').open('w')
process=subprocess.Popen(['timeout','480',str(judge/'.venv/bin/python'),'-u','run.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={{'pid':process.pid,'verified_source_files':{len(files)},'gpu_free_mib':free,'study':'posthoc judge prompt diagnostic','requests':22,'candidate_description_unchanged':True,'literal_task_conditions_included':True}}
(root/'run-start.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
''')
    p.save('t965-diagnostic-run-start.json',result);print(json.dumps(result))
elif mode=='status':
    print(json.dumps(remote(f'''import json
from pathlib import Path
root=Path({REMOTE!r});out=root/'artifacts'
progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None
summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
tail=(root/'run.log').read_text(errors='replace').splitlines()[-7:]
metrics=summary
print(json.dumps({{'progress':progress,'status':summary['status'] if summary else 'running','methods':metrics,'tail':tail,'judge_settings_matched':(out/'judge-settings-comparison.json').exists()}}))
''')))
elif mode=='collect':
    result=remote(f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({REMOTE!r});out=root/'artifacts';judge=Path({JUDGE!r})
summary=json.loads((out/'summary.json').read_text());assert summary['status'] in ['complete','invalid_response']
files=json.loads((out/'transfer-manifest.json').read_text())['files']
sources={{r['path']:out/r['path'] for r in files}}
for name in ['run.py','prepare.py','protocol.json','source-manifest.json','run-start.json','run.log','plan.json','expected-active-fuzz-runtime.json','expected-judge-verification.json']:
 path=root/name
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 files.append({{'path':name,'bytes':path.stat().st_size,'sha256':digest}});sources[name]=path
path=judge/'environment.freeze.txt'
with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
files.append({{'path':'environment.freeze.txt','bytes':path.stat().st_size,'sha256':digest}});sources['environment.freeze.txt']=path
archive=Path('/tmp/t2375-t965-diagnostic-export.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
 for name,path in sources.items():tar.add(path,arcname=name,recursive=False)
with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps({{'files':files,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'archive_path':str(archive)}}))
''')
    archive=HERE/'t965-diagnostic-results.tar.gz'
    subprocess.run(['scp',*common,'-P',str(port),f'root@{host}:{result["archive_path"]}',str(archive)],capture_output=True,check=True,timeout=90)
    assert archive.stat().st_size==result['archive_bytes'] and sha(archive)==result['archive_sha256']
    dest=PROJECT/'results/2026-11-10-t965-failed-cases';dest.mkdir(exist_ok=True)
    rows={r['path']:r for r in result['files']}
    assert len(rows)==len(result['files'])
    with tarfile.open(archive) as tar:
        assert set(tar.getnames())==set(rows)
        for member in tar.getmembers():
            assert member.isfile() and Path(member.name).name==member.name
            target=dest/member.name;target.write_bytes(tar.extractfile(member).read())
            row=rows[member.name]
            assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    p.save('t965-diagnostic-backup-verification.json',dict(instance_id=59903353,verified_locally=True,files=result['files'],local_directory=str(dest)))
    print(json.dumps(dict(verified_locally=True,files=len(rows),directory=str(dest))))
else:raise ValueError('Supported: deploy/status/collect')
