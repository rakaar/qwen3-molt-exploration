"""Pinned one-time installation through the already established worker SSH route.

Execute this only on worker 56373998. No provider lifecycle or Portal operation.
It uses the cached CUDA runtime and adds only two hash-pinned adapter packages.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path,PurePosixPath
import re
import ssl
import subprocess
import time
import urllib.request

CODE=Path('/workspace/em-single-layer-code/experiments')
PRIVATE=Path('/workspace/em-single-layer-private')
PYTHON='/workspace/em-screen-venv/bin/python'
NATIVE=Path('/workspace/em-screen-bootstrap/worker_deadline_v2.py')
NATIVE_SHA='4150cfad75f6db3bcf3cddf945f49520e6d9a1e15df4addbffe7648e2902cc84'
DEADLINE='2026-10-17T17:16:57.886044+00:00'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def get(commit,path,bound=256*1024):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full source commit required')
    name=PurePosixPath(path)
    if (name.is_absolute() or '..' in name.parts
            or name.parts[0] not in ('emergent-misalignment-screen','emergent-misalignment-single-layer')
            or name.suffix not in ('.py','.json','.md','.txt','.yaml')):
        raise ValueError('Unexpected source path')
    url='https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/'+commit+'/experiments/'+path
    opener=urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()),NoRedirect())
    with opener.open(url,timeout=20) as response:data=response.read(bound+1)
    if len(data)>bound:raise ValueError('Source file exceeds bound')
    return data


def put(path,data):
    if path.is_symlink():raise ValueError('Symlink installation refused')
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Preserve existing changed source; installation refused')
        return
    with path.open('xb') as stream:stream.write(data)
    path.chmod(0o600)


def native_running():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(NATIVE).encode() in args:return True
    return False


def install(commit,pins_sha):
    if os.environ.get('CONTAINER_ID')!='56373998' or 'SPAR_VAST_API_KEY' in os.environ:
        raise ValueError('Exact GPU worker and no controller credential required')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE):raise ValueError('Deadline passed')
    if not re.fullmatch('[0-9a-f]{64}',pins_sha):raise ValueError('Manifest hash required')
    PRIVATE.mkdir(mode=0o700,parents=True,exist_ok=True);PRIVATE.chmod(0o700)
    with (PRIVATE/'install.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (PRIVATE/'trial-launch.json').exists():raise ValueError('A launch was already attempted')
        raw=get(commit,'emergent-misalignment-single-layer/source-pins.json')
        if hashlib.sha256(raw).hexdigest()!=pins_sha:raise ValueError('Source manifest hash mismatch')
        pins=json.loads(raw)
        if set(pins)!={'baseline_commit','sha256'} or not 1<=len(pins['sha256'])<=40:
            raise ValueError('Source manifest schema/count')
        downloaded={};total=0
        for name,expected in pins['sha256'].items():
            data=get(commit,name);total+=len(data)
            if total>2*1024*1024 or hashlib.sha256(data).hexdigest()!=expected:
                raise ValueError('Source hash or total byte bound')
            downloaded[name]=data
        # Verify the full set before writing any executable source.
        for name,data in downloaded.items():put(CODE/name,data)
        project=CODE/'emergent-misalignment-single-layer'
        put(project/'source-pins.json',raw)
        if hashlib.sha256(NATIVE.read_bytes()).hexdigest()!=NATIVE_SHA:
            raise ValueError('Retained native backstop source mismatch')
        # Use the genuine worker-native token only in this existing guard.
        with (PRIVATE/'native.log').open('ab') as stream:
            if not native_running():
                subprocess.Popen(['python3',str(NATIVE),'--deadline-utc',DEADLINE],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        guard=Path('/workspace/em-screen-results/native-deadline-v2-status.json')
        for attempt in range(20):
            try:
                value=json.loads(guard.read_text())
                age=time.time()-datetime.fromisoformat(value['checked_at_utc']).timestamp()
                if (value.get('instance_id')==56373998 and value.get('account')==710735
                        and value.get('native_identity_verified') is True
                        and value.get('deadline_utc')==DEADLINE and 0<=age<=45):break
            except (OSError,ValueError,KeyError):pass
            time.sleep(3)
        else:raise ValueError('Fresh native deadline proof unavailable')
        with (PRIVATE/'package-install.log').open('ab') as stream:
            subprocess.run([PYTHON,'-m','pip','install','--index-url','https://pypi.org/simple',
                '--no-deps','--require-hashes','-r',str(project/'requirements-extra.txt')],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,check=True,timeout=600)
            subprocess.run([PYTHON,'-m','pip','check'],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=stream,check=True,timeout=60)
        subprocess.run([PYTHON,str(project/'trial.py'),'dry-run'],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            check=True,timeout=30)
        env=os.environ.copy()
        for name in ('CONTAINER_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN',
                'VAST_API_KEY','VAST_AI_KEY','VASTAI_API_KEY'):
            env.pop(name,None)
        receipt={'commit':commit,'source_pins_sha256':pins_sha,'instance_id':56373998,
            'overall_stop_utc':DEADLINE,'attempted_at_utc':datetime.now(timezone.utc).isoformat()}
        put(PRIVATE/'trial-launch.json',(json.dumps(receipt,indent=2)+'\n').encode())
        with (PRIVATE/'trial.log').open('ab') as stream:
            child=subprocess.Popen([PYTHON,str(project/'trial.py'),'run'],cwd=project,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
        receipt['pid']=child.pid
        p=PRIVATE/'trial-launch-result.json';p.write_text(json.dumps(receipt,indent=2)+'\n');p.chmod(0o600)
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--pins-sha256',required=True)
    args=parser.parse_args()
    try:install(args.commit,args.pins_sha256)
    except Exception as error:
        raise SystemExit('Worker installation failed ('+type(error).__name__+'); private logs retained')
