"""Merge immutable completed jobs, preserving each input/result hash."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('jobs',nargs='+');a=p.parse_args();R=a.root
parts=[];rows=[]
for name in a.jobs:
 file=R/'artifacts'/f'{name}.json';data=json.loads(file.read_text());assert data['status']=='complete'
 parts.append({'file':str(file.relative_to(R)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'conditions':len(data['rows']),'job':data['job'],'started_at_utc':data['started_at_utc'],'finished_at_utc':data['finished_at_utc']});rows+=data['rows']
assert len(rows)==len({r['trial_id'] for r in rows})
(R/'artifacts/all-steering.json').write_text(json.dumps({'parts':parts,'rows':rows},ensure_ascii=False,indent=2)+'\n')
print('Merged',len(rows),'trials in',len(parts),'immutable result parts')
