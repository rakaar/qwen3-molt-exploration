"""Join immutable prior records and new measurements, reusing only identical case/answer reviews."""
import argparse,hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent;W=S.parents[2];R=W/'results/2026-10-19-abstention-complete'
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
 cases={c['id']:c for c in read(S/'cases.json')};records=read(R/'imported-measurements.json');known={(r['case_id'],r['answer_sha256']):r['review'] for r in records}
 for c in cases.values():
  correct=c['cohort']=='correct';known.setdefault((c['id'],sha(c['prior_native_answer'].encode())),{'coherent':True,'success':False,'answer_present':True,'made_up_answer':False if correct else True,'explicit_abstention':False,'question_drift':False,'false_explanation':False,'category':'correct' if correct else 'answer','notes':'Exactly reproduces the previously reviewed native answer.'})
 manual=read(R/'new-judgments.json') if (R/'new-judgments.json').exists() else {}
 for k,v in manual.items():known[tuple(k.split('|'))]=v
 queue={};newcontrols=[];parts=[]
 paths=sorted((R/'artifacts').glob('*.json'))
 if a.partial:paths+=sorted(p for p in R.glob('*-current.json') if not (R/'artifacts'/p.name.replace('-current','')).exists())
 for p in paths:
  d=read(p)
  if not a.partial:assert d['status']=='complete',p
  parts.append({'file':str(p.relative_to(W)),'sha256':sha(p.read_bytes()),'rows':len(d['rows']),'status':d['status']})
  for i,r in enumerate(d['rows']):
   if r['added_gate'] is None or r['added_gate']==0:newcontrols.append(r);continue
   h=sha(r['answer'].encode());key=(r['case_id'],h);c=cases[r['case_id']]
   if key not in known:
    if key not in queue:queue[key]={'case_id':r['case_id'],'scope':r['scope'],'trial_ids':[],'answer':r['answer'],'answer_sha256':h}
    queue[key]['trial_ids'].append(r['trial_id']);continue
   records.append({'id':f"{r['case_id']}|{r['scope']}|{r['added_gate']:g}",'case_id':r['case_id'],'cohort':c['cohort'],'scope':r['scope'],'strength':r['added_gate'],'answer':r['answer'],'answer_sha256':h,'output_token_ids':r['output_token_ids'],'stop_reason':r['stop_reason'],'review':known[key],'source_file':str(p.relative_to(W)),'source_row_index':i,'source_trial_id':r['trial_id'],'source_review_file':str((R/'new-judgments.json').relative_to(W)),'actual_edited_prompt_positions':sum(t['actual_bf16_delta_norm']>0 for t in r['traces'] if t['segment']=='prompt'),'actual_edited_output_positions':sum(t['actual_bf16_delta_norm']>0 for t in r['traces'] if t['segment']=='response')})
 dedup={}
 for r in records:
  key=(r['case_id'],r['scope'],r['strength'])
  if key in dedup:assert dedup[key]['answer_sha256']==r['answer_sha256']
  else:dedup[key]=r
 records=list(dedup.values());write(R/'review-queue.json',list(queue.values()))
 if not a.partial:assert not queue,'Unreviewed novel responses remain'
 write(R/'measurements.json',records);write(R/'new-part-manifest.json',parts)
 print(json.dumps({'reviewed_positive_records':len(records),'new_controls':len(newcontrols),'unreviewed_unique_answers':len(queue),'unreviewed_trials':sum(len(q['trial_ids']) for q in queue.values())}))
