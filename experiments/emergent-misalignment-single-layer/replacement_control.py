"""Scoped controller for one user-requested replacement and two directory copies.

Only rent-once creates; no restart/delete methods. Watch and stop retain storage.
Provider responses, worker secret and signed S3 URLs are never printed.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import secrets
import ssl
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import replacement as r
from start_existing import official_client

ROOT=Path('/workspace/work/em-replacement-control')
REPO=Path('/workspace/qwen3-molt-exploration')
SCREEN=REPO/'experiments/emergent-misalignment-screen'
SOURCE=56373998
SOURCE_LABEL='spar-psy-neuro-em-screen-20261017'
IMAGE='vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py311'
COPIES=(('/workspace/em-screen-venv/','/workspace/em-screen-venv/'),
    ('/workspace/.hf_home/hub/models--Qwen--Qwen3-4B/',
     '/workspace/.hf_home/hub/models--Qwen--Qwen3-4B/'))


def write(name,value,exclusive=False):
    ROOT.mkdir(mode=0o700,parents=True,exist_ok=True);ROOT.chmod(0o700)
    path=ROOT/name
    raw=(json.dumps(value,indent=2)+'\n').encode()
    if exclusive:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(raw)
    else:
        temporary=path.with_suffix('.tmp')
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(raw)
        temporary.replace(path)


def state():return json.loads((ROOT/'state.json').read_text())


def network():
    sys.path.insert(0,str(SCREEN/'cloud-controller'))
    from network_evidence import require_network_permission
    value=json.loads(Path('/workspace/work/staged-launch/current-network-evidence.json').read_text())
    for host in ('cloud.vast.ai','s3.amazonaws.com'):require_network_permission(value,host)


def account(client):
    from vastai.api import billing
    user=billing.show_user(client)
    if type(user.get('id')) is not int or user['id']!=710735:
        raise ValueError('Live SPAR billing identity mismatch')
    return user


def source(client):
    from vastai.api import instances
    account(client)
    row=instances.show_instance(client,SOURCE)
    if (row.get('id')!=SOURCE or row.get('label')!=SOURCE_LABEL
            or row.get('actual_status') not in ('stopped','exited')
            or row.get('intended_status')!='stopped' or row.get('disk_space')!=50):
        raise ValueError('Original retained source must remain stopped and exact')
    return row


def target(client,value,allow_base=False):
    from vastai.api import instances
    account(client)
    row=instances.show_instance(client,value['instance_id'])
    labels=[value['label']]
    if allow_base:labels.append(value['base_label'])
    if (row.get('id')!=value['instance_id'] or row['id']==SOURCE
            or row.get('label') not in labels or row.get('disk_space')!=50
            or row.get('gpu_name') not in ('RTX 3090','RTX 3090 Ti')
            or not 0<float(row.get('dph_total',99))<=.26):
        raise ValueError('Exact replacement identity/disk/GPU/rate mismatch')
    return row


def allocation(value,user,now=None):
    now=time.time() if now is None else now
    old=json.loads(Path('/workspace/work/staged-launch/active-state.json').read_text())
    full_old=8*old['dph_total']
    elapsed=max(0,now-datetime.fromisoformat(value['created_at_utc']).timestamp())/3600
    observed=max(0,old['initial_credit']-user['credit'])+old['prior_observed_spend_usd']
    return max(full_old+elapsed*value['dph_total'],observed)+old['data_reserve_usd']+2.0+old['retention_reserve_usd']+value['retention_reserve_usd']


def prepare(offer_id):
    network();c=official_client();source(c)
    from vastai.api import offers
    query={'num_gpus':{'eq':1},'gpu_ram':{'gte':24000},
        'compute_cap':{'gte':800},'cuda_max_good':{'gte':12.8},
        'dph_total':{'lte':.26},'reliability2':{'gte':.98}}
    rows=[x for x in offers.search_offers(c,query,storage=50,limit=200,
        order=[['dph_total','asc']]) if x.get('id')==offer_id]
    if len(rows)!=1 or rows[0]['id']!=offer_id:raise ValueError('Selected offer no longer available')
    offer=rows[0]
    if (offer['gpu_name'] not in ('RTX 3090','RTX 3090 Ti') or offer['machine_id']==39772
            or not 0<float(offer['dph_total'])<=.26
            or not 0<float(offer['storage_total_cost'])<=.061
            or not 0<=float(offer['inet_down_cost'])<=.016):
        raise ValueError('Compatible independent Ampere replacement required')
    value={'account':710735,'base_label':'spar-em-ft-20261017-r1-'+secrets.token_hex(6),
        'label':None,'instance_id':None,'offer_id':offer_id,'dph_total':offer['dph_total'],
        'retention_reserve_usd':24*offer['storage_total_cost'],
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'stop_deadline_utc':r.DEADLINE,
        'readiness_deadline_utc':datetime.fromtimestamp(min(time.time()+1500,
            datetime.fromisoformat(r.DEADLINE).timestamp()-7200),timezone.utc).isoformat(),
        'worker_secret':secrets.token_hex(32),'source_instance_id':SOURCE,
        'request_scope':'User requested one replacement GPU and storage transfer; old worker remains stopped',
        'new_data_reserve_usd':2.0,'budget_stop_usd':9.0}
    if allocation(value,account(c))>=9:raise ValueError('Combined budget margin exceeded')
    value['label']=value['base_label'];write('state.json',value,True)
    write('offer.json',{k:offer[k] for k in ('id','gpu_name','machine_id','dph_total',
        'storage_total_cost','inet_down_cost','inet_up_cost','reliability2')},True)
    print(json.dumps({'prepared':True,'offer_id':offer_id,'gpu':offer['gpu_name'],
        'hourly_including_50gb':offer['dph_total'],'retained_disk_daily':value['retention_reserve_usd'],
        'combined_conservative_budget_at_hard_stop':allocation(value,account(c),datetime.fromisoformat(r.DEADLINE).timestamp()),
        'stop_deadline_utc':r.DEADLINE}))


def startup(commit,expected,base):
    if not __import__('re').fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full startup commit required')
    if not __import__('re').fullmatch('[0-9a-f]{64}',expected):raise ValueError('Startup source SHA required')
    # Official args launch: onstart overrides ENTRYPOINT to bash, args are bash -c.
    return "exec python3 - <<'PY'\n"+f'''
import hashlib,os,ssl,urllib.request
from pathlib import Path
os.umask(0o077)
class NR(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*a,**k):return None
o=urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()),NR())
url='https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/{commit}/experiments/emergent-misalignment-single-layer/replacement.py'
with o.open(url,timeout=30) as response:b=response.read(131073)
assert len(b)<=131072 and hashlib.sha256(b).hexdigest()=='{expected}'
p=Path('/workspace/em-replacement.py');p.write_bytes(b);p.chmod(0o600)
os.execv('/usr/bin/python3',['python3',str(p),'worker','{base}'])
'''+'PY\n'


def rent(commit,expected,reviewed_retry=False):
    network();value=state();c=official_client();source(c)
    from vastai.api import instances
    hb=json.loads((ROOT/'heartbeat.json').read_text())
    if (hb.get('authenticated') is not True or time.time()-hb['unix_time']>45
            or allocation(value,account(c))>=9 or time.time()>=datetime.fromisoformat(value['readiness_deadline_utc']).timestamp()):
        raise ValueError('Fresh independent watch/budget/setup window required')
    os.kill(hb['pid'],0)
    if reviewed_retry:
        proof=json.loads((ROOT/'third-empty-inventory-proof.json').read_text())
        blocker=json.loads((ROOT/'rental-blocker.json').read_text())
        if (value.get('instance_id') is not None or proof['matching_replacement_count']!=0
                or proof['seconds_after_failed_request']<90
                or time.time()-datetime.fromisoformat(proof['checked_at_utc']).timestamp()>300
                or blocker['replacement_count']!=0 or blocker['exact_offer_still_available'] is not False
                or not blocker['rental_failure'].startswith('HTTPError;')):
            raise ValueError('Prior negative request and zero-rental discovery required')
        account(c)
        if any(isinstance(x.get('label'),str) and x['label'].split(':')[0]==value['base_label']
                for x in instances.show_instances(c)):
            raise ValueError('Existing job rental discovered; second creation refused')
    script=startup(commit,expected,value['base_label'])
    marker={'attempted_at_utc':datetime.now(timezone.utc).isoformat(),
        'offer_id':value['offer_id'],'startup_commit':commit,'startup_sha256':expected,
        'script_sha256':r.sha(script.encode()),'creation_attempts':2 if reviewed_retry else 1}
    marker_name='rent-attempt-2.json' if reviewed_retry else 'rent-attempt.json'
    write(marker_name,marker,True)
    try:
        result=instances.create_instance(c,value['offer_id'],image=IMAGE,disk=50,
            label=value['base_label'],onstart_cmd='bash',args=['-c',script],
            env={'EM_REBIND_TOKEN':value['worker_secret'],'SERVERLESS':'true',
                'ENABLE_AUTH':'true','ENABLE_HTTPS':'true','HF_HOME':'/workspace/.hf_home',
                'HF_HUB_CACHE':'/workspace/.hf_home/hub'},cancel_unavail=True)
    except Exception as error:
        response=getattr(error,'response',None)
        failure={'error_type':type(error).__name__,'http_status':None,'safe_error_words':[]}
        if response is not None:
            failure['http_status']=response.status_code
            text=response.text[:8192].lower()
            failure['safe_error_words']=[w for w in ('unavailable','invalid','expired','offer',
                'capacity','credit','permission','prohibited','instance','create','args',
                'entrypoint','image','label','disk','balance','resources','not found') if w in text]
        write('rent-http-failure-'+str(marker['creation_attempts'])+'.json',failure)
        raise
    identifier=result.get('new_contract')
    if result.get('success') is not True or type(identifier) is not int or identifier==SOURCE:
        write('rent-result.json',{'acknowledged':False,'no_automatic_retry':True})
        raise ValueError('Creation not positively acknowledged; inventory discovery only')
    value['instance_id']=identifier;value['label']=value['base_label'];write('state.json',value)
    target(c,value)
    epoch=int(time.time());label=r.sign_label(value['base_label'],identifier,epoch,value['worker_secret'])
    binding=r.receipt(value['base_label'],identifier,label,value['worker_secret'])
    account(c);result=instances.label_instance(c,identifier,label)
    value['label']=label;write('state.json',value)
    target(c,value)
    pins=json.loads((Path(__file__).parent/'source-pins.json').read_text())
    original={name:(REPO/'experiments'/name).read_bytes() for name in pins['sha256']}
    if any(r.sha(original[k])!=v for k,v in pins['sha256'].items()):raise ValueError('Reviewed source changed')
    files,native,binding_raw=r.bind(original,(REPO/'experiments'/r.NATIVE_PATH).read_bytes(),binding)
    for name,raw in files.items():r.put(ROOT/'bound/experiments'/name,raw)
    r.put(ROOT/'bound/native.py',native);r.put(ROOT/'bound/account-binding.json',binding_raw)
    write('rent-result.json',{'acknowledged':True,'instance_id':identifier,
        'account':710735,'native_source_sha256':r.sha(native),
        'source_pins_sha256':r.sha(files['emergent-misalignment-single-layer/source-pins.json']),
        'stop_deadline_utc':r.DEADLINE})
    print(json.dumps({'created_one_replacement':True,'instance_id':identifier,
        'source_still_stopped':True,'guard_receipt_bound':True,'stop_deadline_utc':r.DEADLINE}))


def retry_offer(offer_id,commit,expected):
    network();value=state();c=official_client();source(c)
    from vastai.api import offers
    rows=[x for x in offers.search_offers(c,{'num_gpus':{'eq':1},'gpu_ram':{'gte':24000},
        'compute_cap':{'gte':800},'cuda_max_good':{'gte':12.8},'reliability2':{'gte':.98}},
        storage=50,limit=200,order=[['dph_total','asc']]) if x.get('id')==offer_id]
    if len(rows)!=1:raise ValueError('Retry offer must be live and exact')
    offer=rows[0]
    if (offer['gpu_name'] not in ('RTX 3090','RTX 3090 Ti') or offer['machine_id']==39772
            or not 0<float(offer['dph_total'])<=.26 or not 0<float(offer['storage_total_cost'])<=.061
            or not 0<=float(offer['inet_down_cost'])<=.016):
        raise ValueError('Retry quote outside compatible cost/resource bounds')
    before={k:value[k] for k in ('offer_id','dph_total','retention_reserve_usd')}
    value.update(offer_id=offer_id,dph_total=offer['dph_total'],retention_reserve_usd=24*offer['storage_total_cost'])
    if allocation(value,account(c),datetime.fromisoformat(r.DEADLINE).timestamp())>=9:
        raise ValueError('Combined hard-stop cost reserve exceeds nine-dollar margin')
    write('reviewed-retry-audit.json',{'before':before,'after':{k:value[k] for k in before},
        'reason':'Prior HTTP error; repeated live inventory has no job rental; original offer now absent',
        'readiness_deadline_unchanged':value['readiness_deadline_utc'],
        'stop_deadline_unchanged':r.DEADLINE,'maximum_successful_replacements':1},True)
    write('state.json',value)
    rent(commit,expected,reviewed_retry=True)


def copy():
    network();value=state();c=official_client()
    from vastai.api import storage
    for index,(src,dst) in enumerate(COPIES):
        source(c);target(c,value)
        write('copy-attempt-'+str(index)+'.json',{'src_id':SOURCE,'dst_id':value['instance_id'],
            'src_path':src,'dst_path':dst,'attempted_at_utc':datetime.now(timezone.utc).isoformat()},True)
        result=storage.copy(c,SOURCE,value['instance_id'],src,dst)
        acknowledged=isinstance(result,dict) and result.get('success') is True
        write('copy-result-'+str(index)+'.json',{'initiated':acknowledged,
            'completion_verified':False,'src_id':SOURCE,'dst_id':value['instance_id']})
        if not acknowledged:raise ValueError('Copy not acknowledged; no repeated mutation')
    print(json.dumps({'copy_requests':2,'source_instance_id':SOURCE,
        'replacement_instance_id':value['instance_id'],'completion_verified':False}))


def bounded_s3(url):
    parts=urlsplit(url)
    if (parts.scheme!='https' or parts.hostname!='s3.amazonaws.com'
            or parts.port not in (None,443) or parts.username or parts.password):
        raise ValueError('Exact S3 HTTPS origin required')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://proxy:8080',
        'http':'http://proxy:8080'}),urllib.request.HTTPSHandler(context=ssl.create_default_context()),r.NoRedirect())
    for attempt in range(30):
        try:
            with opener.open(url,timeout=20) as response:body=response.read(8*1024*1024+1)
            if len(body)>8*1024*1024:raise ValueError('S3 log response exceeds bound')
            return body
        except urllib.error.HTTPError as error:
            transient=error.code==404
            if error.code==403:
                raw=error.read(4097)
                if len(raw)<=4096:
                    try:transient=ET.fromstring(raw).findtext('Code') in {'AccessDenied','NoSuchKey'}
                    except ET.ParseError:pass
            if not transient or attempt==29:raise
            time.sleep(2)


def fetch():
    network();value=state();c=official_client();target(c,value)
    account(c)
    result=c.put('/instances/request_logs/'+str(value['instance_id'])+'/',json_data={'tail':'10000'})
    result.raise_for_status();url=result.json().get('result_url')
    if not isinstance(url,str):raise ValueError('Official task log URL unavailable')
    # This capability remains in memory; no provider credential is sent to S3.
    return bounded_s3(url)


def watch():
    from vastai.api import instances
    c=official_client()
    while True:
        try:
            value=state();user=account(c);now=time.time()
            write('heartbeat.json',{'pid':os.getpid(),'unix_time':now,'account':710735,
                'authenticated':True,'managed_proxy_tls':True})
            identifier=value.get('instance_id')
            if identifier is None:
                rows=instances.show_instances(c)
                candidates=[x for x in rows if x.get('label','').split(':')[0]==value['base_label']]
                if len(candidates)==1:identifier=candidates[0]['id']
            if identifier is not None:
                check={**value,'instance_id':identifier}
                row=target(c,check,allow_base=True)
                reason=None
                if now>=datetime.fromisoformat(r.DEADLINE).timestamp():reason='unchanged overall deadline'
                elif not (ROOT/'ready.json').exists() and now>=datetime.fromisoformat(value['readiness_deadline_utc']).timestamp():reason='replacement setup deadline'
                elif allocation(value,user,now)>=9:reason='combined nine-dollar budget margin'
                write('watch-status.json',{'instance_id':identifier,'actual_status':row.get('actual_status'),
                    'checked_at_utc':datetime.now(timezone.utc).isoformat(),'allocation_usd':allocation(value,user,now),
                    'ready':(ROOT/'ready.json').exists(),'stop_reason':reason})
                if row.get('actual_status') in ('stopped','exited') and row.get('intended_status')=='stopped':
                    write('stopped.json',{'instance_id':identifier,'verified_at_utc':datetime.now(timezone.utc).isoformat()});return
                if reason:
                    account(c);target(c,check,allow_base=True);instances.stop_instance(c,identifier)
                    write('stop-requested.json',{'instance_id':identifier,'reason':reason,'retained_disk':True})
        except Exception as error:
            write('watch-error.json',{'error_type':type(error).__name__,
                'checked_at_utc':datetime.now(timezone.utc).isoformat(),'raw_details_suppressed':True})
        time.sleep(15)


def observe(stop=False):
    network();value=state();c=official_client();row=target(c,value);source(c)
    if stop:
        from vastai.api import instances
        account(c);result=instances.stop_instance(c,value['instance_id'])
        write('authorized-stop.json',{'instance_id':value['instance_id'],
            'acknowledged':result.get('success') is True,'retained_disk':True,
            'requested_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'instance_id':row['id'],'actual_status':row.get('actual_status'),
        'cur_state':row.get('cur_state'),'intended_status':row.get('intended_status'),
        'hourly_rate_usd':row.get('dph_total'),'source_remains_stopped':True,
        'stop_requested':stop,'overall_stop_utc':r.DEADLINE}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','rent-once','retry-failed-offer','copy-once','watch','status','stop'))
    parser.add_argument('--offer-id',type=int);parser.add_argument('--commit');parser.add_argument('--sha256')
    args=parser.parse_args()
    try:
        if args.action=='prepare':prepare(args.offer_id)
        elif args.action=='rent-once':rent(args.commit,args.sha256)
        elif args.action=='retry-failed-offer':retry_offer(args.offer_id,args.commit,args.sha256)
        elif args.action=='copy-once':copy()
        elif args.action=='watch':watch()
        else:observe(args.action=='stop')
    except Exception as error:
        raise SystemExit('Replacement controller blocked ('+type(error).__name__+'); raw provider/credential details suppressed')
