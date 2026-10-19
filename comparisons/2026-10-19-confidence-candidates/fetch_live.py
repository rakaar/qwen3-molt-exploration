"""Read-only authenticated candidate checks. Credentials are prompted, never saved."""
import concurrent.futures,getpass,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
BASE='https://amsterdam.taild80f17.ts.net'
password=getpass.getpass('Dashboard password (hidden): ')
auth=('friends',password)
candidates=[(6,1751),(11,652),(13,2128),(14,40),(12,1667),(19,710)]
paths=[(f'/layers/layer-{l}/autointerp-layer/features/{f}.{ext}',f'live/L{l}-T{f}.{ext}') for l,f in candidates for ext in ['html','json']]
paths.append(('/layers/layer-35/autointerp-layer/summary.json','live/layer-35-summary.json'))
def fetch(item):
 path,name=item;r=requests.get(BASE+path,auth=auth,timeout=35);assert r.url==BASE+path
 record={'url':BASE+path,'status':r.status_code,'checked_utc':datetime.now(timezone.utc).isoformat()}
 if r.ok:
  p=ROOT/name;p.parent.mkdir(exist_ok=True);p.write_bytes(r.content);record.update(file=str(p.relative_to(ROOT)),bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:records=list(ex.map(fetch,paths))
(ROOT/'live-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'requests':len(records),'successful':sum(x['status']==200 for x in records),'status_codes':[x['status'] for x in records]}))
