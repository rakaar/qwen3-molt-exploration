"""Guarded, single-instance SPAR rental for the authorized $10 probe."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
HELPER = '/home/rka/.codex/skills/spar-vast-gpu/scripts/spar_vast.py'
SPAR_CWD = '/home/rka/.codex/.chatgpt-projects/g-p-6aaae53c1d6c8191bfe2aeaf0aa820a8'
LABEL = 'spar-psy-neuro-t2375-verify-20261109'
FIELDS = {'id','label','actual_status','cur_state','intended_status','next_state',
          'dph_total','dph_base','storage_cost','storage_total_cost','disk_space',
          'disk_usage','machine_id','host_id','gpu_name','gpu_ram','reliability2',
          'ssh_host','ssh_port','public_ipaddr','ports','start_date','status_msg',
          'inet_down_cost','inet_up_cost','inet_down_billed','inet_up_billed','image_uuid','instance'}

def request(args):
    run = subprocess.run(['python3',HELPER,*args],cwd=SPAR_CWD,capture_output=True,text=True,timeout=110)
    if run.returncode:
        raise RuntimeError('Guarded SPAR call failed: '+run.stderr[-1200:])
    try: return json.loads(run.stdout)
    except json.JSONDecodeError: return ast.literal_eval(run.stdout)

def save(name, data):
    (HERE/name).write_text(json.dumps(data,indent=2)+'\n')

def safe(data):
    return {k:v for k,v in data.items() if k in FIELDS}

def current():
    record=json.loads((HERE/'rental-record.json').read_text())
    data=request(['cli','show','instance',str(record['instance_id'])])
    assert data['id']==record['instance_id'] and data['label']==LABEL
    return record,data

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='rent':
        assert not (HERE/'create-receipt.json').exists(), 'Review prior creation; never retry blindly'
        identity=request(['status'])
        assert identity['id']==710735 and identity['verified_team']=='SPAR'
        save('billing-before.json',{k:identity[k] for k in ['id','verified_team','credit','balance']})
        instances=request(['cli','show','instances'])
        assert not any(i.get('label')==LABEL for i in instances)
        save('instances-before.json',[safe(i) for i in instances])
        offers=request(['cli','search','offers',
            'rentable=true verified=true num_gpus=1 gpu_name=RTX_3090 gpu_ram>=24 direct_port_count>=1 reliability>=0.99 dph_total<=0.30 inet_down_cost<=0.02',
            '--limit','5','--storage','40','-o','dph_total'])
        if isinstance(offers,dict): offers=offers.get('offers',[])
        assert offers, 'No suitable offer under rate ceiling'
        offer=min(offers,key=lambda o:o['dph_total'])
        assert offer['dph_total']<=.30
        save('offer.json',safe(offer))
        startup='#!/bin/bash\nset -eu\necho T2375_VERIFY_SSH_FINGERPRINT\nssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub\necho T2375_VERIFY_READY\n'
        result=request(['cli','create','instance',str(offer['id']),
            '--image','pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime','--disk','40',
            '--ssh','--direct','--label',LABEL,'--cancel-unavail','--onstart-cmd',startup])
        save('create-receipt.json',{k:v for k,v in result.items() if k in ['success','new_contract','error','msg']})
        assert result.get('success') is True and isinstance(result.get('new_contract'),int)
        record=dict(instance_id=result['new_contract'],account_id=710735,label=LABEL,
            budget_usd=10,hourly_rate_ceiling_usd=.30,rate_including_disk=offer['dph_total'],
            inet_down_cost_per_gb=offer['inet_down_cost'],inet_up_cost_per_gb=offer['inet_up_cost'],
            created_at_utc=datetime.now(timezone.utc).isoformat(),offer_id=offer['id'],
            cleanup_authorized=False,keep_running_requested=True,
            purpose='Run actual verify_hypothesis backing implementation for revised water/eleven explanation',
            remote_workspace='/workspace/t2375-verify-20261109')
        save('rental-record.json',record)
        print(json.dumps(record))
    elif mode=='status':
        record,data=current(); save('instance-status.json',safe(data))
        print(json.dumps(safe(data)))
    elif mode=='attach':
        record,data=current()
        result=request(['cli','attach','ssh',str(record['instance_id']),'/home/rka/.ssh/vastai_ed25519.pub'])
        assert result.get('success') is True
        save('ssh-key-attachment.json',dict(instance_id=record['instance_id'],success=True))
        print(json.dumps(dict(instance_id=record['instance_id'],public_key_attached=True)))
    else:
        raise ValueError('No lifecycle action authorized; supported modes: rent/status/attach')
