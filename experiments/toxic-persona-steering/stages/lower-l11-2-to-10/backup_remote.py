"""Download and hash-check this completed stage only; no GPU lifecycle action."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/toxic-persona/stages/lower-l11-2-to-10'
ssh_opts = ['ssh','-i','/home/rka/.ssh/vastai_ed25519','-p','28365','-o','BatchMode=yes',
            '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/toxic-persona-57059436-known-hosts']
script = """
from pathlib import Path
import hashlib,json
r=Path('/root/toxic-persona/stages/lower-l11-2-to-10')
assert (r/'COMPLETE').exists()
assert json.loads((r/'artifacts/verification.json').read_text())['status']=='passed'
files=[{'file':str(f.relative_to(r)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(r.rglob('*')) if f.is_file()]
print(json.dumps({'instance_id':57059436,'remote_root':str(r),'files':files}))
"""
response = subprocess.run(ssh_opts+['root@64.25.128.119', '/venv/main/bin/python -c '+shlex.quote(script)], capture_output=True, check=True)
manifest = json.loads(response.stdout)
manifest['captured_at_utc'] = datetime.now(timezone.utc).isoformat()
(ROOT/'remote-backup-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
backup = ROOT/'remote-backup'
subprocess.run(['rsync','-az','-e',shlex.join(ssh_opts),'root@64.25.128.119:'+REMOTE+'/',str(backup)+'/'],check=True)
for entry in manifest['files']:
    local = backup/entry['file']
    assert local.stat().st_size==entry['bytes']
    assert hashlib.sha256(local.read_bytes()).hexdigest()==entry['sha256'], entry['file']
shutil.copytree(backup/'artifacts',ROOT/'artifacts',dirs_exist_ok=True)
check={'status':'passed','files':len(manifest['files']),'bytes':sum(f['bytes'] for f in manifest['files']),
       'checked_at_utc':datetime.now(timezone.utc).isoformat(),'all_remote_stage_files_hash_verified':True,
       'instance_lifecycle_changed':False}
(ROOT/'backup-verification.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check))
