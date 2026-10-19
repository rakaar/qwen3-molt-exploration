"""Freeze the original twenty cases, import immutable prior measurements, schedule only missing cells."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
S=Path(__file__).resolve().parent;W=S.parents[2];R=W/'results/2026-10-19-abstention-complete'
if (R/'artifacts').exists() and any((R/'artifacts').glob('*.json')):raise RuntimeError('Cohort/protocol already frozen and executed; do not overwrite them.')
read=lambda p:json.loads((W/p).read_text());sha=lambda p:hashlib.sha256((W/p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
cases=read('experiments/passage-uncertainty/answerable-abstention/cases.json')
for c in cases:c['cohort']='correct'
evidence=read('results/2026-10-15-abstention-overview/evidence.json');old=read('results/2026-10-15-vampire-intermediate/artifacts/all-results.json')['rows'];new=read('results/2026-10-15-six-new-abstentions/artifacts/all-steering.json')['rows']
for c in evidence['cases']:
 base=next(r for r in old+new if r['case_id']==c['id'] and r['added_gate'] is None)
 assert base['answer']==c['baseline']
 x={'id':c['id'],'source_id':c['source_id'],'passage_id':base['passage_id'],'question':c['question'],'context':c['context'],'is_answerable':False,'cohort':'made_up','reference_answer':'Cannot determine the requested answer from the passage; a supported false-premise correction is allowed.','prior_native_answer':c['baseline'],'output_token_ids':base['output_token_ids'],'chat':base['chat']}
 if 'prompt_token_ids' in base:x['prompt_token_ids']=base['prompt_token_ids']
 cases.append(x)
assert len(cases)==20 and len({c['id'] for c in cases})==20 and len({c['source_id'] for c in cases})==20
save(S/'cases.json',cases);save(R/'cases.json',cases)
imports=[('output','results/2026-10-19-answerable-abstention/artifacts/all-results.json','results/2026-10-19-answerable-abstention/reviewed-judgments.json'),('both','results/2026-10-19-answerable-all-token/artifacts/all-results.json','results/2026-10-19-answerable-all-token/reviewed-judgments.json'),('input','results/2026-10-19-answerable-input-only/artifacts/all-results.json','results/2026-10-19-answerable-input-only/reviewed-judgments.json'),('output','results/2026-10-15-vampire-intermediate/artifacts/all-results.json','results/2026-10-15-vampire-intermediate/assessment.json'),('output','results/2026-10-15-six-new-abstentions/artifacts/all-steering.json','results/2026-10-15-six-new-abstentions/reviewed-judgments.json')]
bycase={c['id']:c for c in cases};records={};sources=[]
for scope,rfile,gfile in imports:
 d=read(rfile);rs=d['rows'];gd=read(gfile);gs={g['trial_id']:g for g in gd['rows']}
 sources.append({'raw_file':rfile,'raw_sha256':sha(rfile),'review_file':gfile,'review_sha256':sha(gfile)})
 for i,r in enumerate(rs):
  c=bycase[r['case_id']];a=r['added_gate']
  if a is None or a<=0:continue
  g=gs[r['trial_id']];answer_hash=hashlib.sha256(r['answer'].encode()).hexdigest();assert g['answer_sha256']==answer_hash
  if c['cohort']=='correct':
   grade={'coherent':g['coherent'],'success':g['false_abstention'],'answer_present':g['substantive_answer'],'made_up_answer':None,'explicit_abstention':g['explicit_abstention'],'question_drift':g.get('question_drift',False),'false_explanation':g.get('false_passage_justification',False),'category':g['category'],'notes':g['notes']}
  else:
   success=g.get('success',g.get('clean_coherent_abstention',False));made=g['made_up_answer'];false=g.get('false_explanation',g.get('false_or_unsupported_explanation',False));explicit=g.get('explicit_abstention',success or (not made and g['coherent']))
   grade={'coherent':g['coherent'],'success':success,'answer_present':made,'made_up_answer':made,'explicit_abstention':explicit,'question_drift':False,'false_explanation':false,'category':'abstention' if success else 'broken' if not g['coherent'] else 'mixed' if explicit or false else 'answer','notes':g.get('notes',g.get('reason'))}
  rec={'id':f"{c['id']}|{scope}|{a:g}",'case_id':c['id'],'cohort':c['cohort'],'scope':scope,'strength':a,'answer':r['answer'],'answer_sha256':answer_hash,'output_token_ids':r['output_token_ids'],'stop_reason':r['stop_reason'],'review':grade,'source_file':rfile,'source_row_index':i,'source_trial_id':r['trial_id'],'source_review_file':gfile}
  key=(c['id'],scope,a)
  if key in records:assert records[key]['answer_sha256']==answer_hash,key
  else:records[key]=rec
save(R/'imported-measurements.json',list(records.values()));save(R/'import-manifest.json',sources)
p=read('experiments/passage-uncertainty/answerable-input-only/protocol.json');p.update(experiment='Complete 20-question, three-scope abstention study',created_at_utc=datetime.now(timezone.utc).isoformat(),scope='Trial-specific: input edits all prompt positions only; output edits processed response positions only; both edits both. Monitoring hooks return None outside the selected scope.',timing='Prompt edits may alter the first predicted token; output-only leaves first prediction native. KV cache retained in every mode.',cohort_size=20,common_strengths=[1,2,4,8,12,16,20,24,28,32,40,48,56,64],search_policy='Complete common grid for all 60 case/scope cells, then inspect additional intermediate values for unresolved cells. Preserve all previous fine sweeps. No monotonicity assumption.',max_positive_trials_per_case_scope=150,success_definition='Coherent on-question withholding without substantive requested answer or question drift. In made-up-answer cohort, new unsupported claims or false explanations disqualify success; answer-present absence rationales are the induced false abstention and are flagged separately.',display_policy={'success':'Lowest successful tested strength, not a proven minimum.','failure':'Last coherent tested value below first incoherent value; report first breakdown and any later recovery separately. If no breakdown occurs, show highest tested coherent strength and say no breakdown reached.'},lifecycle_policy='Retain SPAR instance 56698356 for follow-ups; no lifecycle changes authorized.')
save(S/'protocol.json',p);save(R/'protocol.json',p)
shutil.copy2(W/'experiments/passage-uncertainty/answerable-input-only/expected-base-verification.json',S/'expected-base-verification.json')
controls=[{'trial_id':c['id']+'-'+label,'case_id':c['id'],'strength':a,'scope':'both'} for label,a in [('native',None),('zero',0)] for c in cases]
missing=[{'trial_id':f"{c['id']}-{scope}-a{a:g}",'case_id':c['id'],'strength':a,'scope':scope} for c in cases for scope in ['input','output','both'] for a in p['common_strengths'] if (c['id'],scope,a) not in records]
# Separate long jobs by scope for checkpointed collection; all use the same runner.
jobs=[('controls',controls)]+[(scope,[t for t in missing if t['scope']==scope]) for scope in ['input','both','output']]
for name,trials in jobs:save(S/(name+'.json'),{'name':name,'cases_file':'cases.json','trials':trials,'repeat_first':name=='controls'})
(S/'drive.sh').write_text('''#!/bin/bash
set -euo pipefail
cd /root/abstention-complete
for job in controls input both output; do
 /venv/main/bin/python run.py --job "$job.json" --model-dir /root/answerable-abstention/models/qwen3-4b --weights /root/answerable-abstention/weights/molt-19-selected.safetensors
done
''')
print(json.dumps({'imported_unique_positive_trials':len(records),'new_controls':len(controls),'missing_grid_trials':len(missing),'jobs':{n:len(ts) for n,ts in jobs}},indent=2))
