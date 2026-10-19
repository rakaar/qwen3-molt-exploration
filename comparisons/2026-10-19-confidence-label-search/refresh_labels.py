"""Read only label/title/description metadata. No activation examples or model runs."""
import concurrent.futures,getpass,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import requests
R=Path(__file__).resolve().parent
BASE='https://amsterdam.taild80f17.ts.net'
auth=('friends',getpass.getpass('Dashboard password (hidden): '))
ids=[(12,1667),(23,710),(26,868),(16,878),(14,203),(15,2349),(18,74),(11,1363),(14,1103),(16,764),(15,1753),(17,198)]
def fetch(pair):
 l,f=pair;url=f'{BASE}/layers/layer-{l}/autointerp-layer/features/{f}.json';r=requests.get(url,auth=auth,timeout=35);r.raise_for_status();assert r.url==url
 d=r.json();return {'layer':l,'transform':f,'label':d.get('label') or d.get('title'),'description':d.get('explanation'),'status':d.get('status'),'label_check_metrics':{k:v.get('metrics',{}) for k,v in d.get('methods',{}).items()},'updated_at':d.get('updated_at'),'url':url,'page_url':url[:-5]+'.html','checked_utc':datetime.now(timezone.utc).isoformat(),'response_sha256':hashlib.sha256(r.content).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,ids))
(R/'live-label-descriptions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'labels_refreshed':len(rows),'activation_examples_read':False,'model_runs':0,'features':[{'id':f"L{x['layer']}-T{x['transform']}",'label':x['label']} for x in rows]}))
