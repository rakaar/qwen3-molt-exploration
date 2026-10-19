"""Content-bound manual judgments; callers must inspect complete text first."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;W=S.parents[2];R=W/'results/2026-10-19-abstention-complete'
rows=[]
paths=sorted((R/'artifacts').glob('*.json'))+sorted(p for p in R.glob('*-current.json') if not (R/'artifacts'/p.name.replace('-current','')).exists())
for p in paths:rows+=json.loads(p.read_text())['rows']
byid={r['trial_id']:r for r in rows};dest=R/'new-judgments.json';judgments=json.loads(dest.read_text()) if dest.exists() else {}
def put(trial,category,notes,*,coherent=True,success=False,answer=True,abstain=False,false=False,drift=False):
 r=byid[trial];key=r['case_id']+'|'+hashlib.sha256(r['answer'].encode()).hexdigest()
 assert not success or coherent and abstain and not answer and not false and not drift
 # Correct-cohort false passage-absence rationales are the intended false-abstention outcome.
 if r['cohort']=='correct' and category=='abstention':success=coherent and abstain and not answer and not drift
 g={'coherent':coherent,'success':success,'answer_present':answer,'made_up_answer':answer if r['cohort']=='made_up' else None,'explicit_abstention':abstain,'question_drift':drift,'false_explanation':false,'category':category,'notes':notes}
 if key in judgments:
  old=judgments[key]
  for k in ['coherent','success','answer_present','question_drift']:assert old[k]==g[k],(trial,k,old,g)
 else:judgments[key]=g

def finish():dest.write_text(json.dumps(judgments,ensure_ascii=False,indent=2)+'\n');print('Content-bound manual judgments',len(judgments))
