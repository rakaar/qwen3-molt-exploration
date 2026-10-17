import os,sys,json
from pathlib import Path
sys.path.insert(0,'/workspace/qwen3-molt-exploration/experiments/emergent-misalignment-single-layer')
from start_existing import official_client
from vastai.api import billing,instances
c=official_client();u=billing.show_user(c)
assert type(u.get('id')) is int and u['id']==710735
r=instances.show_instance(c,56373998)
assert r['label']=='spar-psy-neuro-em-screen-20261017' and r['actual_status']=='exited' and r['intended_status']=='stopped'
x=c.put('/instances/command/56373998/',json_data={'command':'du -d3 -h workspace'});
if x.status_code!=200:
 v=x.json();print({'http_status':x.status_code,'keys':list(v),'safe_error_words':[w for w in ('stopped','inactive','invalid','command','unsupported','unavailable','path','running','execute') if w in str(v).lower()]});raise SystemExit(1)
v=x.json()
url=v.get('result_url')
assert isinstance(url,str)
from urllib.parse import urlsplit
p=urlsplit(url);assert p.scheme=='https' and p.hostname=='s3.amazonaws.com' and not p.username and not p.password
f=Path('/workspace/work/vast-storage-research/cache-result-url');f.write_text(url);f.chmod(0o600)
print(json.dumps({'instance_id':56373998,'account':710735,'command':'bounded cache/runtime size discovery','result_hostname':p.hostname,'result_saved_privately':True}))
