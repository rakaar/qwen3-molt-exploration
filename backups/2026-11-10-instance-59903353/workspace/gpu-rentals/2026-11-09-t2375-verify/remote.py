"""Connect to the live verified rental; deploy, inspect, and verify its results."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import urllib.request

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('provider',HERE/'provider.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
record,state=p.current()
assert state['actual_status']=='running','Container not running yet'
host=state['public_ipaddr'];port=int(state['ports']['22/tcp'][0]['HostPort'])
known=HERE/'known_hosts'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def capture_log_markers():
    run=subprocess.run(['python3',p.HELPER,'cli','logs',str(record['instance_id']),'--tail','300'],cwd=p.SPAR_CWD,capture_output=True,text=True,timeout=45)
    assert run.returncode==0,'Provider log retrieval failed'
    content=run.stdout
    try:
        reply=json.loads(content)
        if isinstance(reply,dict) and reply.get('url'):
            content=urllib.request.urlopen(reply['url'],timeout=25).read().decode()
    except json.JSONDecodeError:
        urls=re.findall(r'https?://[^\s"<>]+',content)
        if urls:
            content=urllib.request.urlopen(urls[-1],timeout=25).read().decode()
    markers=[line for line in content.splitlines() if 'T2375_VERIFY_' in line or ('SHA256:' in line and 'ED25519' in line)]
    p.save('bootstrap-log-markers.json',dict(safe_log_markers=markers))
    return markers

mode=sys.argv[1]
if mode=='deploy':
    markers=capture_log_markers()
    expected={f for line in markers for f in re.findall(r'SHA256:[A-Za-z0-9+/]+',line)}
    scan=subprocess.run(['ssh-keyscan','-T','10','-t','ed25519','-p',str(port),host],capture_output=True,text=True,timeout=25)
    assert scan.returncode==0 and scan.stdout.strip(),'SSH endpoint not ready'
    if known.exists():assert known.read_text()==scan.stdout,'Pinned host key changed'
    else:known.write_text(scan.stdout)
    fingerprint=subprocess.run(['ssh-keygen','-lf',str(known)],capture_output=True,text=True,check=True).stdout
    actual=set(re.findall(r'SHA256:[A-Za-z0-9+/]+',fingerprint))
    if expected:
        assert actual==expected,'Host key does not match provider startup fingerprint'
    else:
        # Standard first-use pinning at the freshly provider-reported endpoint.
        # Existing keys are never replaced or ignored, and all SSH uses strict checking.
        global_known=Path('/home/rka/.ssh/known_hosts')
        def keys(blob):return {tuple(line.split()[1:3]) for line in blob.splitlines() if line and not line.startswith('#')}
        if global_known.exists():
            prior=subprocess.run(['ssh-keygen','-F',f'[{host}]:{port}','-f',str(global_known)],capture_output=True,text=True).stdout
            if keys(prior):assert keys(prior)&keys(scan.stdout),'Known endpoint key changed'
        again=subprocess.run(['ssh-keyscan','-T','10','-t','ed25519','-p',str(port),host],capture_output=True,text=True,timeout=25)
        assert keys(again.stdout)==keys(scan.stdout),'Inconsistent key at provider endpoint'
    p.save('ssh-verification.json',dict(instance_id=record['instance_id'],host=host,port=port,fingerprint=list(actual),
        provider_log_verified=bool(expected),verification_method='provider startup log' if expected else 'first-use pinning at fresh provider endpoint; repeated scan and existing-key check',strict_host_checking=True))
assert known.exists(),'Verify host key in deploy mode before SSH'
common=['-i','/home/rka/.ssh/vastai_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o',f'UserKnownHostsFile="{known}"','-o','ConnectTimeout=15']
ssh=['ssh',*common,'-p',str(port),f'root@{host}','python3 -']
root=record['remote_workspace']
def remote(code,timeout=60):
    r=subprocess.run(ssh,input=code,capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError('Remote command failed: '+r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])

if mode=='deploy':
    source=PROJECT/'experiments/t2375-verify-hypothesis-20261109'
    files=[dict(path=str(f.relative_to(source)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(source.rglob('*'))
           if f.is_file() and '__pycache__' not in f.parts and f.name!='source-manifest.json']
    (source/'source-manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    archive=HERE/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in files:tar.add(source/row['path'],arcname='t2375-verify-20261109/'+row['path'],recursive=False)
        tar.add(source/'source-manifest.json',arcname='t2375-verify-20261109/source-manifest.json')
    dest='/workspace/t2375-verify-source.tar.gz'
    subprocess.run(['scp',*common,'-P',str(port),str(archive),f'root@{host}:{dest}'],check=True,capture_output=True,timeout=60)
    code=f'''import json,hashlib,subprocess,tarfile,shutil,os
from pathlib import Path
archive=Path({dest!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  assert m.isfile() or m.isdir()
 tar.extractall('/workspace',filter='data')
root=Path({root!r})
for row in json.loads((root/'source-manifest.json').read_text())['files']:
 assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
assert '3090' in gpu and shutil.disk_usage('/workspace').free>15000000000
env=os.environ.copy();env['PATH']='/opt/conda/bin:'+env.get('PATH','')
log=(root/'setup-run.log').open('w')
process=subprocess.Popen(['bash','setup.sh'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'setup-run.pid').write_text(str(process.pid))
print(json.dumps({{'gpu':gpu,'free_bytes':shutil.disk_usage('/workspace').free,'pid':process.pid,'source_files_verified':{len(files)},'archive_sha256':{sha(archive)!r}}}))
'''
    result=remote(code)
    p.save('run-start.json',result);print(json.dumps(result))
elif mode=='status':
    code=f'''import json,os
from pathlib import Path
root=Path({root!r});out=root/'artifacts/results.json'
log=root/'setup-run.log';lines=log.read_text(errors='replace').splitlines() if log.exists() else []
data=json.loads(out.read_text()) if out.exists() else {{}}
model_files=[p for p in (root/'model').rglob('*') if p.is_file()] if (root/'model').exists() else []
download_bytes=sum(p.stat().st_size for p in model_files)
print(json.dumps({{'status':data.get('status','setup'),'completed_cases':len(data.get('cases',[])),'tail':lines[-8:],'runtime_ready':(root/'probe-dashboard/metadata.json').exists(),'model_bytes_on_disk':download_bytes,'model_files':len(model_files)}}))
'''
    print(json.dumps(remote(code)))
elif mode=='collect':
    manifest=remote(f'''import json
from pathlib import Path
root=Path({root!r});data=json.loads((root/'artifacts/results.json').read_text());assert data['status']=='complete'
manifest=json.loads((root/'artifacts/transfer-manifest.json').read_text())
print(json.dumps(manifest))
''')
    destination=PROJECT/'results/2026-11-09-t2375-verify-hypothesis';destination.mkdir(parents=True,exist_ok=True)
    for row in manifest['files']:
        assert Path(row['path']).name==row['path']
        target=destination/row['path']
        subprocess.run(['scp',*common,'-P',str(port),f'root@{host}:{root}/artifacts/{row["path"]}',str(target)],capture_output=True,check=True,timeout=90)
        assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    for name in ['environment.freeze.txt','weights/gate-download-manifest.json','setup-run.log']:
        target=destination/Path(name).name
        subprocess.run(['scp',*common,'-P',str(port),f'root@{host}:{root}/{name}',str(target)],capture_output=True,check=True,timeout=60)
    p.save('backup-verification.json',dict(instance_id=record['instance_id'],verified_locally=True,files=manifest['files'],local_directory=str(destination),checked_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(verified_locally=True,files=len(manifest['files']),local_directory=str(destination))))
else:raise ValueError('Supported: deploy/status/collect')
