"""Local SPAR-only rental creation; never publish credentials or raw responses."""
import ast, json, shlex, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/home/rka/Documents/ChatGPT/spar model psychology and neuroscience')
OUT=ROOT/'results/2026-10-15-vampire-intermediate'
HELPER='/home/rka/.codex/skills/spar-vast-gpu/scripts/spar_vast.py'
CWD='/home/rka/.codex/.chatgpt-projects/g-p-6aaae53c1d6c8191bfe2aeaf0aa820a8'
def call(args):
    p=subprocess.run(['python3',HELPER,*args],cwd=CWD,capture_output=True,text=True,timeout=100)
    if p.returncode: raise RuntimeError('Guard command failed: '+p.stderr[:300])
    try: return json.loads(p.stdout)
    except json.JSONDecodeError: return ast.literal_eval(p.stdout)
status=call(['status'])
# Guard itself verifies live billing identity on every invocation.
instances=call(['cli','show','instances'])
assert instances==[], 'Inspect existing resources before creation'
offers=call(['cli','search','offers','gpu_name=RTX_3090 num_gpus=1 rentable=true verified=true cuda_max_good>=12.6 inet_down>=100 dph_total<=0.23','--storage','30','--limit','20','-o','dph_total'])
print('Refreshed offers:',json.dumps([{k:o.get(k) for k in ('id','gpu_name','dph_total','reliability','geolocation','machine_id')} for o in offers]))
matching=[o for o in offers if o['reliability']>=.96 and o['dph_total']<=0.1923 and o['machine_id']!=152610]
assert matching, 'No suitable offer within quoted price ceiling; no rental created'
offer=min(matching,key=lambda o:(str(o.get('geolocation','')).endswith('CN'),o['dph_total']))
assert offer['gpu_name']=='RTX 3090'
assert offer['dph_total']<=0.1923
safe_keys=['id','gpu_name','compute_cap','dph_total','dph_base','storage_total_cost','storage_cost','reliability','inet_down','inet_up','inet_down_cost','inet_up_cost','machine_id','host_id','geolocation']
(OUT/'chosen-offer.json').write_text(json.dumps({k:offer.get(k) for k in safe_keys},indent=2)+'\n')
inner='''set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip ca-certificates
python3 -m venv /venv/main
/venv/main/bin/python -m pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
/venv/main/bin/python -m pip install --no-cache-dir transformers==4.57.6 huggingface-hub==0.36.2 safetensors==0.8.0 requests numpy matplotlib
/venv/main/bin/python -c 'import torch; assert torch.__version__ == "2.14.0+cu126"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
touch /root/runtime-ready
'''
pub=Path('/home/rka/.ssh/vastai_ed25519.pub').read_text().strip()
assert pub.startswith('ssh-ed25519 ')
onstart='install -d -m 700 /root/.ssh\nprintf "%s\\n" '+shlex.quote(pub)+' >> /root/.ssh/authorized_keys\nchmod 600 /root/.ssh/authorized_keys\nnohup bash -c '+shlex.quote(inner)+' > /root/runtime-install.log 2>&1 < /dev/null &\n'
image='ubuntu@sha256:f610ab94648195aa356059f5b41d6085c9d4d903c072430cdd1af7bdb646106b'
response=call(['cli','create','instance',str(offer['id']),'--image',image,'--disk','30','--ssh','--direct','--label','spar-psy-neuro-vampire-intermediate','--onstart-cmd',onstart,'--cancel-unavail'])
assert response.get('success') and response.get('new_contract')
record={'account_id':710735,'account_name':'SPAR 3 Model Psychology','instance_id':response['new_contract'],'offer_id':offer['id'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'hourly_with_disk_usd':offer['dph_total'],'bandwidth_download_usd_per_GB':offer['inet_down_cost'],'bandwidth_upload_usd_per_GB':offer['inet_up_cost'],'budget_ceiling_usd':1,'budget_scope':'Active setup and experiment compute. Retention charges continue afterward. No automatic lifecycle actions.','purpose':'L19 T368 vampire-bat intermediate additive-strength sweep; replay three prior coherent abstention successes','lifecycle_policy':'User explicitly requested retaining GPU and disk after experiment; no stop or deletion.','image':image,'machine_id':offer['machine_id'],'results_path':'results/2026-10-15-vampire-intermediate','source_path':'experiments/passage-uncertainty/vampire-intermediate'}
(OUT/'rental-record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
