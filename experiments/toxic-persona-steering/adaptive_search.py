"""Prepare, launch, inspect and back up bounded stages using the unchanged runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SEARCH = ROOT / 'stages/adaptive-l11-authority'
TEMPLATE = ROOT / 'stages/power-context-l11-8-to-16'
REMOTE = '/root/toxic-persona/stages/adaptive-l11-authority'
SSH = ['ssh', '-i', '/home/rka/.ssh/vastai_ed25519', '-p', '28365',
       '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3',
       '-o', 'UserKnownHostsFile=/tmp/toxic-persona-57059436-known-hosts']
HOST = 'root@64.25.128.119'

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def remote(code):
    result = subprocess.run(SSH + [HOST, '/venv/main/bin/python -c ' + shlex.quote(code)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)

def prepare(name, cases, strengths, cap):
    dest = SEARCH / name
    assert not dest.exists(), 'Never overwrite an existing stage'
    dest.mkdir(parents=True)
    p = json.loads((TEMPLATE / 'protocol.json').read_text())
    all_cases = p['cases'] + [next(c for c in json.loads((ROOT/'protocol.json').read_text())['cases']
                                if c['id'] == 'ruler_of_the_world')]
    p.update(experiment='Adaptive L11-T2234 authority search: ' + name,
             cases=[c for c in all_cases if c['id'] in cases], strengths=strengths,
             max_runtime_minutes=45,
             extension_rule='Frozen adaptive search plan in parent search-plan.json; no new feature controls.')
    assert len(p['cases']) == len(cases)
    p['decoding']['max_new_tokens'] = cap
    p['initial_trials_per_question'] = 2 + len(strengths)
    p['requested_display_strengths'] = [0, *strengths]
    p['resource']['incremental_compute_bound_usd'] = .3525
    p['source_questions'] = 'Exact previously authorized prompts, reused without rewriting.'
    write(dest/'protocol.json',p)
    for file in ['run.py','verify_results.py','expected-base-verification.json']:
        shutil.copy2(TEMPLATE/file,dest/file)
    assert (dest/'run.py').read_bytes() == (ROOT/'run.py').read_bytes()
    shutil.copytree(TEMPLATE/'weights',dest/'weights')
    script = ('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(REMOTE+'/'+name) +
              '\nexport TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=6\n'
              '/venv/main/bin/python -u run.py --model-dir /root/toxic-persona/models/qwen3-4b > run.log 2>&1\n'
              '/venv/main/bin/python verify_results.py artifacts/results.json > verification.log 2>&1\ntouch COMPLETE\n')
    (dest/'launch.sh').write_text(script)
    write(dest/'launch-source-manifest.json', {str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest()
          for f in dest.rglob('*') if f.is_file()})
    return dest

def launch(names, after=None):
    for name in names:
        source = SEARCH/name
        assert source.is_dir()
        remote("from pathlib import Path; import json; p=Path("+repr(REMOTE+'/'+name)+
               "); assert not (p/'artifacts/results.json').exists(); p.mkdir(parents=True,exist_ok=True); print(json.dumps({'ready':True}))")
        subprocess.run(['rsync','-az','-e',shlex.join(SSH),str(source)+'/',HOST+':'+REMOTE+'/'+name+'/'],check=True)
    queue_name = 'queue-' + datetime.now(timezone.utc).strftime('%H%M%S')
    code = """import subprocess,json
from pathlib import Path
root=Path(REMOTE)
names=NAMES
queue=root/QUEUE
assert not queue.exists()
dependency=AFTER
wait_line=('while [ ! -f '+str(root/dependency/'COMPLETE')+' ]; do sleep 2; done\\n') if dependency else ''
queue.write_text('#!/usr/bin/env bash\\nset -euo pipefail\\n'+wait_line+''.join('bash '+str(root/n/'launch.sh')+'\\n' for n in names))
with (root/(QUEUE+'.log')).open('wb') as log:
    proc=subprocess.Popen(['bash',str(queue)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/(QUEUE+'.pid')).write_text(str(proc.pid)+'\\n')
print(json.dumps({'queue':QUEUE,'pid':proc.pid,'stages':names}))
""".replace('REMOTE',repr(REMOTE)).replace('NAMES',repr(names)).replace('QUEUE',repr(queue_name)).replace('AFTER',repr(after))
    result = remote(code)
    write(SEARCH/(queue_name+'.json'),result)
    print(json.dumps(result))

def snapshot(names):
    code = """import json
from pathlib import Path
root=Path(REMOTE)
out=[]
for n in NAMES:
    p=root/n
    r=p/'artifacts/results.json'
    data=json.loads(r.read_text()) if r.exists() else None
    out.append({'stage':n,'complete':(p/'COMPLETE').exists(),'result':data,
                'log_tail':(p/'run.log').read_text()[-1200:] if not data and (p/'run.log').exists() else ''})
print(json.dumps(out))
""".replace('REMOTE',repr(REMOTE)).replace('NAMES',repr(names))
    rows = remote(code)
    for item in rows:
        r=item.pop('result')
        if r:
            d=SEARCH/item['stage']/'snapshots'
            d.mkdir(exist_ok=True)
            write(d/'latest.json',r)
            item.update(status=r['status'],rows=len(r['rows']),planned=len(r['stage_cases'])*(2+len(r['stage_strengths'])),
                        trials=[x['id'] for x in r['rows']])
    print(json.dumps(rows))

def backup(names):
    for name in names:
        d=SEARCH/name
        code="""from pathlib import Path
import json,hashlib
r=Path(REMOTE)
assert (r/'COMPLETE').exists()
assert json.loads((r/'artifacts/verification.json').read_text())['status']=='passed'
print(json.dumps({'instance_id':57059436,'remote_root':str(r),'files':[{'file':str(f.relative_to(r)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(r.rglob('*')) if f.is_file()]}))
""".replace('REMOTE',repr(REMOTE+'/'+name))
        manifest=remote(code)
        write(d/'remote-backup-manifest.json',manifest)
        b=d/'remote-backup'
        subprocess.run(['rsync','-az','-e',shlex.join(SSH),HOST+':'+REMOTE+'/'+name+'/',str(b)+'/'],check=True)
        for f in manifest['files']:
            local=b/f['file']
            assert local.stat().st_size==f['bytes']
            assert hashlib.sha256(local.read_bytes()).hexdigest()==f['sha256']
        shutil.copytree(b/'artifacts',d/'artifacts',dirs_exist_ok=True)
        check={'status':'passed','files':len(manifest['files']),'bytes':sum(f['bytes'] for f in manifest['files']),
               'all_remote_stage_files_hash_verified':True,'instance_lifecycle_changed':False,
               'checked_at_utc':datetime.now(timezone.utc).isoformat()}
        write(d/'backup-verification.json',check)
        print(json.dumps({'stage':name,**check}))

parser=argparse.ArgumentParser()
sub=parser.add_subparsers(dest='command',required=True)
p=sub.add_parser('prepare'); p.add_argument('name'); p.add_argument('--cases',nargs='+',required=True)
settings=p.add_mutually_exclusive_group(required=True)
settings.add_argument('--strengths',nargs='+',type=float)
settings.add_argument('--native-only',action='store_true')
p.add_argument('--cap',type=int,default=512)
for name in ['launch','snapshot','backup']:
    sp=sub.add_parser(name); sp.add_argument('names',nargs='+')
    if name=='launch': sp.add_argument('--after')
args=parser.parse_args()
if args.command=='prepare': print(prepare(args.name,args.cases,args.strengths or [],args.cap))
elif args.command=='launch': launch(args.names,args.after)
elif args.command=='snapshot': snapshot(args.names)
elif args.command=='backup': backup(args.names)
