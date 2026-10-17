"""One replacement worker: deterministic binding, cached runtime, fixed stop.

The management credential is controller-only. The worker accepts a live SPAR
receipt authenticated by a generated worker-only HMAC secret in its own label.
This file never creates a resource; the separately guarded controller does.
"""
from datetime import datetime, timezone
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
import urllib.request

OLD_ID = 56373998
OLD_LABEL = 'spar-psy-neuro-em-screen-20261017'
OLD_BINDING = '26f348d729fe3c14e9566d1a50c31dee07f3286bd557629c04ad4ecf36893942'
OLD_NATIVE = '4150cfad75f6db3bcf3cddf945f49520e6d9a1e15df4addbffe7648e2902cc84'
COMMIT = '1de5552ca0ba74e3aed0fccf2a6b2623e5e16cca'
PINS_SHA = '507a93141bb44b2d92a1e509fcf9e71ef73213672b327864c4b2c1abbc95382c'
DEADLINE = '2026-10-17T17:16:57.886044+00:00'
NATIVE_PATH = 'emergent-misalignment-screen/cloud-controller/worker-bootstrap/worker_deadline_v2.py'
CODE = Path('/workspace/em-single-layer-code/experiments')
PRIVATE = Path('/workspace/em-single-layer-private')
PYTHON = '/workspace/em-screen-venv/bin/python'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()


def sign_label(base, instance, epoch, secret):
    if (not re.fullmatch(r'spar-em-ft-20261017-r1-[0-9a-f]{12}', base)
            or type(instance) is not int or instance <= 0 or instance == OLD_ID
            or type(epoch) is not int or not 1792220000 < epoch < 1792257418
            or not re.fullmatch('[0-9a-f]{64}', secret)):
        raise ValueError('Invalid replacement identity or worker-only secret')
    message = canonical({'base_label':base, 'instance_id':instance,
        'account':710735, 'epoch':epoch, 'stop_deadline_utc':DEADLINE})
    signature = hmac.new(bytes.fromhex(secret), message, hashlib.sha256).hexdigest()
    return base+':'+str(instance)+':'+str(epoch)+':'+signature


def receipt(base, instance, label, secret, now=None):
    pieces=label.split(':')
    if len(pieces)!=4 or not pieces[2].isdigit():
        raise ValueError('Signed controller receipt absent')
    epoch=int(pieces[2])
    expected=sign_label(base, instance, epoch, secret)
    if not hmac.compare_digest(label, expected):
        raise ValueError('Replacement receipt signature mismatch')
    now=time.time() if now is None else now
    if not 0 <= now-epoch <= 1800:
        raise ValueError('Fresh live controller ownership receipt required')
    return {'account':710735,'instance_id':instance,'label':label,
        'stop_deadline_utc':DEADLINE,
        'verified_at_utc':datetime.fromtimestamp(epoch,timezone.utc).isoformat()}


def bind(files, native, binding):
    """Exact private derivation; original experiment files remain byte identical."""
    if sha(native)!=OLD_NATIVE:
        raise ValueError('Native template hash mismatch')
    instance=binding['instance_id'];label=binding['label']
    if type(instance) is not int or instance == OLD_ID or binding['account']!=710735:
        raise ValueError('Replacement binding required')
    binding_raw=canonical(binding);binding_sha=sha(binding_raw)
    native=native.replace(str(OLD_ID).encode(),str(instance).encode()).replace(
        OLD_LABEL.encode(),label.encode()).replace(OLD_BINDING.encode(),binding_sha.encode())
    native_sha=sha(native)
    changed={}
    for name,raw in files.items():
        if name.startswith('emergent-misalignment-single-layer/'):
            for old,new in ((str(OLD_ID),str(instance)),(OLD_LABEL,label),
                    (OLD_BINDING,binding_sha),(OLD_NATIVE,native_sha)):
                raw=raw.replace(old.encode(),new.encode())
        changed[name]=raw
    wire='emergent-misalignment-single-layer/wire.py'
    changed[wire]=changed[wire].replace(b"'status.json','runtime.json'",b"'transfer.json','status.json','runtime.json'").replace(
        b'def validate(name,data):\n',b'def validate(name,data):\n    if name==\'transfer.json\':\n        validate_transfer(data);return\n')
    changed[wire]+=TRANSFER_VALIDATOR.encode()
    pins={'baseline_commit':'21da619829635b97b1797600e6b3e5bd3ef5e468',
        'sha256':{k:sha(v) for k,v in sorted(changed.items())}}
    changed['emergent-misalignment-single-layer/source-pins.json']=(
        json.dumps(pins,indent=2)+'\n').encode()
    return changed,native,binding_raw


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def get_source(name):
    if (not re.fullmatch(r'emergent-misalignment-(screen|single-layer)/[A-Za-z0-9_./-]+',name)
            or '..' in name.split('/') or not name.endswith(('.json','.py','.txt','.md','.yaml'))):
        raise ValueError('Unexpected pinned public source')
    url='https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/'+COMMIT+'/experiments/'+name
    opener=urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()),NoRedirect())
    with opener.open(url,timeout=20) as r:raw=r.read(262145)
    if len(raw)>262144:raise ValueError('Source byte bound')
    return raw


def put(path,raw):
    path=Path(path)
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Symlink write refused')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Existing changed file preserved')
        return
    with path.open('xb') as stream:stream.write(raw)
    path.chmod(0o600)


def run(command,timeout=300,env=None):
    with (PRIVATE/'bootstrap.log').open('ab') as stream:
        subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,
            check=True,timeout=timeout,env=env)


def proof(command,timeout):
    value=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,
        check=True,timeout=timeout)
    if len(value.stdout)>16384:raise ValueError('Transfer proof exceeds bound')
    return json.loads(value.stdout)


def own_row(instance):
    env=os.environ.copy()
    for key in ('VAST_AI_KEY','VASTAI_API_KEY','VAST_API_KEY','VAST_URL','VAST_API_URL'):
        env.pop(key,None)
    env['VAST_API_KEY']=os.environ['CONTAINER_API_KEY']
    for kind in ('CONFIG','CACHE','STATE'):
        env['XDG_'+kind+'_HOME']='/workspace/em-screen-native-cli/'+kind.lower()
    result=subprocess.run(['/workspace/em-screen-native-cli/bin/vastai','--raw',
        '--retry','1','--url','https://cloud.vast.ai','show','instance',str(instance)],
        env=env,capture_output=True,text=True,timeout=45,check=True)
    row=json.loads(result.stdout)
    if not isinstance(row,dict) or row.get('id')!=instance:
        raise ValueError('Native own-instance mismatch')
    return row


def verify_runtime_files():
    """Hash copied distributions against wheel RECORD, never execute partial copies."""
    script=r'''
import base64,hashlib,importlib.metadata,json
names=['torch','transformers','accelerate','huggingface-hub','safetensors','tokenizers',
 'numpy','nvidia-cublas-cu12','nvidia-cudnn-cu12','nvidia-cuda-runtime-cu12']
count=0
for name in names:
 d=importlib.metadata.distribution(name)
 for f in d.files or []:
  if not f.hash:continue
  p=d.locate_file(f)
  if not p.is_file() or (f.size is not None and p.stat().st_size!=f.size):raise ValueError('partial runtime')
  h=hashlib.new(f.hash.mode)
  with p.open('rb') as stream:
   for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
  if base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode()!=f.hash.value:raise ValueError('runtime hash')
  count+=1
print(json.dumps({'runtime_record_files_verified':count}))
'''
    return proof([PYTHON,'-c',script],timeout=240)


def verify_model_files():
    script=r'''
from huggingface_hub import HfApi
from pathlib import Path
import hashlib,json
rev='1cfa9a7208912126459214e8b04321603b3df60c'
info=HfApi().model_info('Qwen/Qwen3-4B',revision=rev,files_metadata=True,token=False)
root=Path('/workspace/.hf_home/hub/models--Qwen--Qwen3-4B/snapshots')/rev
index=json.loads((root/'model.safetensors.index.json').read_text())
expected=set(index['weight_map'].values())
verified=[]
for item in info.siblings:
 if not item.rfilename.endswith('.safetensors'):continue
 if not item.lfs:raise ValueError('LFS metadata absent')
 p=root/item.rfilename
 if not p.is_file() or p.stat().st_size!=item.lfs.size:raise ValueError('partial model')
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=item.lfs.sha256:raise ValueError('model hash')
 verified.append({'file':item.rfilename,'sha256':h.hexdigest(),'bytes':p.stat().st_size})
if not 1<=len(verified)<=8 or {x['file'] for x in verified}!=expected:raise ValueError('Pinned shard index mismatch')
print(json.dumps({'pinned_revision':rev,'verified_model_files':verified}))
'''
    return proof([PYTHON,'-c',script],timeout=300)


def worker(base):
    os.umask(0o077)
    PRIVATE.mkdir(mode=0o700,parents=True,exist_ok=True)
    for path in ('/workspace/em-screen-venv','/workspace/.hf_home/hub/models--Qwen--Qwen3-4B'):
        Path(path).mkdir(mode=0o700,parents=True,exist_ok=True)
    if 'SPAR_VAST_API_KEY' in os.environ:raise ValueError('Management credential prohibited')
    instance=int(os.environ['CONTAINER_ID'])
    secret=os.environ['EM_REBIND_TOKEN']
    # Fresh instance-scoped CLI only. No old configuration, environment or token copied.
    run(['python3','-m','venv','/workspace/em-screen-native-cli'])
    run(['/workspace/em-screen-native-cli/bin/python','-m','pip','install',
        '--index-url','https://pypi.org/simple','vastai==1.8.3'])
    binding=None
    for _ in range(60):
        try:
            row=own_row(instance)
            binding=receipt(base,instance,row['label'],secret)
            break
        except (ValueError,KeyError,subprocess.SubprocessError):time.sleep(5)
    if binding is None:raise ValueError('Fresh controller receipt unavailable')
    raw=get_source('emergent-misalignment-single-layer/source-pins.json')
    if sha(raw)!=PINS_SHA:raise ValueError('Original source pins mismatch')
    pins=json.loads(raw);files={}
    if len(pins['sha256'])!=29:raise ValueError('Source count mismatch')
    for name,expected in pins['sha256'].items():
        value=get_source(name)
        if sha(value)!=expected:raise ValueError('Original source mismatch')
        files[name]=value
    files,native,binding_raw=bind(files,get_source(NATIVE_PATH),binding)
    for name,value in files.items():put(CODE/name,value)
    native_path=Path('/workspace/em-screen-bootstrap/worker_deadline_v2.py')
    put(native_path,native)
    put(native_path.parent/'account-binding.json',binding_raw)
    with (PRIVATE/'native.log').open('ab') as stream:
        subprocess.Popen(['python3',str(native_path),'--deadline-utc',DEADLINE],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
    project=CODE/'emergent-misalignment-single-layer'
    sys.path.insert(0,str(project))
    import wire
    output=Path('/workspace/em-single-layer-results')
    put(output/'status.json',canonical({'run_id':wire.RUN_ID,'instance_id':instance,
        'phase':'waiting-cached-transfer','checked_at_utc':datetime.now(timezone.utc).isoformat()}))
    put(output/'source-pins.json',files['emergent-misalignment-single-layer/source-pins.json'])
    wire.snapshot(output)
    cutoff=min(time.time()+1200,datetime.fromisoformat(DEADLINE).timestamp()-7200)
    # Transfer may be asynchronous; each verification is read-only and bounded.
    while time.time()<cutoff:
        try:
            runtime_proof=verify_runtime_files();model_proof=verify_model_files();break
        except (OSError,subprocess.SubprocessError):
            time.sleep(20)
    else:raise TimeoutError('Verified runtime/model transfer deadline exhausted')
    transfer={'instance_id':instance,'source_instance_id':OLD_ID,
        'runtime_record_files_verified':runtime_proof['runtime_record_files_verified'],
        'baseline_revision':model_proof['pinned_revision'],
        'model_files':model_proof['verified_model_files'],
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    put(output/'transfer.json',canonical(transfer))
    wire.snapshot(output)
    run([PYTHON,'-m','pip','install','--index-url','https://pypi.org/simple',
        '--no-deps','--require-hashes','-r',str(project/'requirements-extra.txt')],timeout=600)
    run([PYTHON,'-m','pip','check'],timeout=60)
    run([PYTHON,str(project/'trial.py'),'dry-run'],timeout=30)
    env=os.environ.copy()
    for key in ('CONTAINER_API_KEY','EM_REBIND_TOKEN','WEB_PASSWORD','HF_TOKEN',
            'HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','VAST_API_KEY','VAST_AI_KEY','VASTAI_API_KEY'):
        env.pop(key,None)
    env['HF_HOME']='/workspace/.hf_home'
    env['HF_HUB_CACHE']='/workspace/.hf_home/hub'
    env['HF_HUB_DISABLE_TELEMETRY']='1'
    env['WANDB_DISABLED']='true'
    receipt_file={'instance_id':instance,'launch_utc':datetime.now(timezone.utc).isoformat(),
        'source_pins_sha256':sha((project/'source-pins.json').read_bytes()),
        'native_source_sha256':sha(native),'runtime_and_model_hashes_verified':True,
        'overall_stop_utc':DEADLINE}
    put(PRIVATE/'trial-launch.json',canonical(receipt_file))
    with (PRIVATE/'trial.log').open('ab') as stream:
        child=subprocess.Popen([PYTHON,str(project/'trial.py'),'run'],cwd=project,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
    put(PRIVATE/'trial-pid',str(child.pid).encode())
    # Keep the worker container alive, including after training exits, until guarded stop.
    while time.time()<datetime.fromisoformat(DEADLINE).timestamp()+180:time.sleep(30)


TRANSFER_VALIDATOR = '''
def validate_transfer(data):
    value=loads(data)
    if (set(value)!={'instance_id','source_instance_id','runtime_record_files_verified',
            'baseline_revision','model_files','checked_at_utc'}
            or value['instance_id']!=INSTANCE or value['source_instance_id']!=56373998
            or type(value['runtime_record_files_verified']) is not int
            or not 100<=value['runtime_record_files_verified']<=200000
            or value['baseline_revision']!='1cfa9a7208912126459214e8b04321603b3df60c'
            or not isinstance(value['checked_at_utc'],str) or len(value['checked_at_utc'])>40
            or not isinstance(value['model_files'],list) or not 1<=len(value['model_files'])<=8):
        raise ValueError('Transfer proof fields/identity')
    names=set()
    for item in value['model_files']:
        if (set(item)!={'file','sha256','bytes'}
                or not re.fullmatch(r'model-[0-9]{5}-of-[0-9]{5}\\.safetensors',item['file'])
                or not re.fullmatch('[0-9a-f]{64}',item['sha256'])
                or type(item['bytes']) is not int or not 0<item['bytes']<=5*1024**3
                or item['file'] in names):raise ValueError('Transfer shard proof')
        names.add(item['file'])
'''


if __name__=='__main__':
    try:
        if len(sys.argv)!=3 or sys.argv[1]!='worker':raise ValueError('Worker mode/base label required')
        worker(sys.argv[2])
    except Exception as error:
        # Never dump native output, credentials, environment or provider responses.
        print('Replacement bootstrap failed: '+type(error).__name__,flush=True)
        raise SystemExit(1)
