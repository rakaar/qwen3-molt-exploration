"""Read-only preflight by default; one stopped worker can be started explicitly.

Use --start-once only after resolving the earlier user's explicit no-restart
restriction. No create, reboot, update/recreate, delete, or environment API exists
here. The original overall deadline, reserves and controller remain unchanged.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
SCREEN=ROOT.parent/'emergent-misalignment-screen'
STATE=Path('/workspace/work/staged-launch')
PRIVATE=Path('/workspace/work/em-single-layer-control')
ACCOUNT=710735
INSTANCE=56373998
LABEL='spar-psy-neuro-em-screen-20261017'
DEADLINE='2026-10-17T17:16:57.886044+00:00'


def check(user,rows,state,heartbeat,now):
    if type(user.get('id')) is not int or user['id']!=ACCOUNT:
        raise ValueError('Live SPAR identity mismatch')
    if (state.get('account')!=ACCOUNT or state.get('instance_id')!=INSTANCE
            or state.get('label')!=LABEL or state.get('stop_deadline_utc')!=DEADLINE):
        raise ValueError('Original task budget/deadline binding mismatch')
    if now>=datetime.fromisoformat(DEADLINE):raise ValueError('Original deadline passed')
    numbers=[user.get('credit')]+[state.get(k) for k in ('dph_total','initial_credit',
        'data_reserve_usd','retention_reserve_usd','prior_observed_spend_usd')]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in numbers):
        raise ValueError('Finite original budget and live credit required')
    matches=[r for r in rows if r.get('label')==LABEL]
    if len(matches)!=1 or matches[0].get('id')!=INSTANCE:
        raise ValueError('Exact single existing worker not established')
    row=matches[0]
    if (row.get('cur_state')!='stopped' or row.get('intended_status')!='stopped'
            or row.get('actual_status') not in ('stopped','exited')):
        raise ValueError('Start requires a verified stopped worker; no restart of running work')
    if row.get('disk_space')!=50 or row.get('gpu_name') not in ('RTX 3090','NVIDIA GeForce RTX 3090'):
        raise ValueError('Existing 50 GB RTX 3090 contract mismatch')
    rate=row.get('dph_total')
    if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<=state['dph_total']+1e-8:
        raise ValueError('Live contract rate absent or increased')
    if (heartbeat.get('pid')!=1747 or heartbeat.get('account')!=ACCOUNT
            or heartbeat.get('authenticated_network_roundtrip') is not True
            or not 0<=now.timestamp()-heartbeat['unix_time']<=45):
        raise ValueError('Fresh original authenticated CPU watchdog required')
    elapsed=max(0,(now-datetime.fromisoformat(state['created_at_utc'])).total_seconds())/3600
    forecast=elapsed*state['dph_total']+state['data_reserve_usd']
    observed=max(0,state['initial_credit']-user['credit'])+state['prior_observed_spend_usd']
    allocation=max(forecast,observed)+state['retention_reserve_usd']
    if allocation>=9:raise ValueError('Original budget safety margin reached')
    return {'account':ACCOUNT,'instance_id':INSTANCE,'label':LABEL,
        'cur_state':row['cur_state'],'actual_status':row['actual_status'],
        'disk_gb':row['disk_space'],'hourly_rate_usd':rate,
        'conservative_allocation_usd':allocation,'overall_stop_utc':DEADLINE,
        'watcher_pid':1747,'watcher_authenticated_and_fresh':True}


def official_client():
    if importlib.metadata.version('vastai')!='1.8.3':raise ValueError('Official vastai 1.8.3 required')
    sys.path.insert(0,str(SCREEN/'cloud-controller/controller-shims'))
    from vastai.api.client import VastClient,VastSession
    from vastai_transport import configure
    configure(VastSession)
    key=os.environ.get('SPAR_VAST_API_KEY')
    if not key:raise ValueError('Scoped cloud SPAR credential unavailable')
    return VastClient(api_key=key,server_url='https://cloud.vast.ai',retry=1,
        explain=False,curl=False,timeout=45)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-once',action='store_true')
    args=parser.parse_args()
    from trial import source_proof
    source_proof()
    sys.path.insert(0,str(SCREEN))
    from spar_vast_cloud import require_project
    require_project()
    sys.path.insert(0,str(SCREEN/'cloud-controller'))
    from network_evidence import require_network_permission
    evidence=json.loads((STATE/'current-network-evidence.json').read_text())
    require_network_permission(evidence,'cloud.vast.ai')
    client=official_client()
    from vastai.api import billing,instances
    user=billing.show_user(client)
    if type(user.get('id')) is not int or user['id']!=ACCOUNT:
        raise ValueError('Live account mismatch; no further provider access')
    rows=instances.show_instances(client)
    state=json.loads((STATE/'active-state.json').read_text())
    heartbeat=json.loads((STATE/'watch-heartbeat.json').read_text())
    proof=check(user,rows,state,heartbeat,datetime.now(timezone.utc))
    os.kill(1747,0)
    cmd=Path('/proc/1747/cmdline').read_bytes().split(b'\0')
    if b'/workspace/work/staged-launch/controller_watch.py' not in cmd:
        raise ValueError('Watchdog PID no longer identifies the original controller')
    proof['checked_at_utc']=datetime.now(timezone.utc).isoformat()
    PRIVATE.mkdir(mode=0o700,parents=True,exist_ok=True);PRIVATE.chmod(0o700)
    if args.start_once:
        # Exclusive marker precedes the one mutation. Ambiguous replies never
        # cause an automatic repeated start, recreation, or replacement rental.
        marker=PRIVATE/'start-attempt.json'
        with marker.open('x') as stream:json.dump(proof,stream,indent=2)
        marker.chmod(0o600)
        result=instances.start_instance(client,INSTANCE)
        proof['start_acknowledged']=isinstance(result,dict) and result.get('success') is True
        proof['start_attempted_once']=True
        proof['running_verified']=False
    else:
        proof['provider_mutation_performed']=False
    p=PRIVATE/('start-result.json' if args.start_once else 'preflight.json')
    p.write_text(json.dumps(proof,indent=2)+'\n');p.chmod(0o600)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        raise SystemExit('Existing-worker preflight/start failed ('+type(error).__name__+'); raw provider details suppressed')
