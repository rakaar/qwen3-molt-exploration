"""Fetch an atomic result snapshot and print metadata only, keeping outputs hidden."""
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parent
ssh = ['ssh','-i','/home/rka/.ssh/vastai_ed25519','-p','28365','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/tmp/toxic-persona-57059436-known-hosts','root@64.25.128.119']
r = subprocess.run(ssh + ['cat /root/toxic-persona/stages/expanded-l11-12-to-30/artifacts/results.json'], capture_output=True, check=True)
d = json.loads(r.stdout)
(ROOT / 'snapshots').mkdir(exist_ok=True)
(ROOT / 'snapshots/latest.json').write_bytes(r.stdout)
complete = [c for c in d['stage_cases'] if sum(r['case_id']==c for r in d['rows'])==8]
print(json.dumps({'status':d['status'],'generations_complete':len(d['rows']),'generations_planned':56,'completed_cases':complete}))
