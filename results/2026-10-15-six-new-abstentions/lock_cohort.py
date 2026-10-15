"""Freeze first six reviewed native errors before any steering is launched."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();R=a.root
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
review=json.loads((R/'screening-review.json').read_text());assert review['reviewed_before_any_steering']
chosen=[r for r in review['rows'] if r['decision']=='made_up_answer'][:6];assert len(chosen)==6
allrows=[];sources={}
for i in [1,2,3]:
 f=R/f'screen-{i}.json';d=json.loads(f.read_text());assert d['status']=='complete' and d['repeat_first_token_identical'];allrows+=d['rows'];sources[f.name]=sha(f)
rows={r['id']:r for r in allrows};cases=[]
for item in chosen:
 c=rows[item['case_id']];assert hashlib.sha256(c['answer'].encode()).hexdigest()==item['answer_sha256']
 cases.append({**c,'screening_reason':item['reason']})
assert len({c['source_id'] for c in cases})==6
record={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'selection':'First six clear made-up native answers in frozen screening order; conservative ambiguities excluded before steering. No steered output observed for selection.','screening_questions':len(allrows),'screening_review_sha256':sha(R/'screening-review.json'),'screening_result_sha256':sources,'cases':cases,'keep_all_selected_cases_in_denominator':True}
assert not (R/'cohort-lock.json').exists()
(R/'cohort-lock.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');(R/'cohort-cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
P=json.loads((R/'protocol.json').read_text());trials=[]
for c in cases:
 for strength in [None,0]+P['coarse_strengths']:
  trials.append({'case_id':c['id'],'strength':strength,'trial_id':c['id']+('_native_replay' if strength is None else '_add'+str(strength))})
job={'name':'coarse','cases_file':'cohort-cases.json','repeat_first':False,'trials':trials,'cohort_lock_sha256':sha(R/'cohort-lock.json')}
(R/'coarse-job.json').write_text(json.dumps(job,indent=2)+'\n');print('Locked',[(c['id'],c['question']) for c in cases])
