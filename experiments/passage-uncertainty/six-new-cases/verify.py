"""Audit outputs independently of the generation script and manual grading."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);R=p.parse_args().root
read=lambda n:json.loads((R/n).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=read('cohort-lock.json');cases={c['id']:c for c in lock['cases']};assert len(cases)==6
raw=read('artifacts/all-steering.json');assert len(raw['rows'])==len({r['trial_id'] for r in raw['rows']})
review=read('reviewed-judgments.json');grades={r['trial_id']:r for r in review['rows']};assert set(grades)=={r['trial_id'] for r in raw['rows']}
P=read('protocol.json');controls=0;positive=0;positions=0;firsts=[]
for part in raw['parts']:
 f=R/part['file'];assert sha(f)==part['sha256'];sub=json.loads(f.read_text());assert sub['status']=='complete' and sub['base_files_hash_verified'];assert sub['started_at_utc']>lock['locked_at_utc']
 for name,digest in sub['source_sha256'].items():assert sha(R/name)==digest,name
 assert sub['protocol']==P
 if 'cohort_lock_sha256' in sub['job']:assert sub['job']['cohort_lock_sha256']==sha(R/'cohort-lock.json')
for c in cases.values():
 rows=[r for r in raw['rows'] if r['case_id']==c['id']];strengths=[r['added_gate'] for r in rows if r['added_gate'] is not None and r['added_gate']>0]
 assert len(strengths)==len(set(strengths)) and len(strengths)<=P['max_positive_trials_per_case']
 assert set(P['coarse_strengths'])<=set(strengths)
 assert sum(r['added_gate'] is None for r in rows)==1 and sum(r['added_gate']==0 for r in rows)==1
 for r in rows:
  assert r['prompt_token_ids']==c['prompt_token_ids'] and r['chat']==c['chat'];assert r['output_token_ids'][0]==c['output_token_ids'][0] and r['first_output_token_unchanged'];firsts.append(True)
  g=grades[r['trial_id']];assert g['answer_sha256']==hashlib.sha256(r['answer'].encode()).hexdigest()
  assert all(isinstance(g[k],bool) for k in ['coherent','explicit_abstention','success'])
  assert all(g[k] is None or isinstance(g[k],bool) for k in ['made_up_answer','false_explanation'])
  if g['success']:assert g['coherent'] and g['explicit_abstention'] and g['made_up_answer'] is False and g['false_explanation'] is False
  v=r['added_gate'];trace=r['traces']
  if v is None or v==0:
   assert r['native_or_zero_matches_baseline'] and r['output_token_ids']==c['output_token_ids'];assert all(t['actual_bf16_delta_norm']==0 for t in trace);controls+=1
  else:
   assert trace[0]['segment']=='prompt_final' and trace[0]['gate_delta']==0 and trace[0]['actual_bf16_delta_norm']==0
   ts=trace[1:];assert len(ts)==len(r['output_token_ids'])-1
   for i,t in enumerate(ts):
    assert t['segment']=='response' and t['eligible'] and t['answer_position']==i and t['full_chat_position']==len(c['prompt_token_ids'])+i
    assert t['token_id']==r['output_token_ids'][i] and t['gate_delta']==v and t['added_gate']==v and t['effective_estimated_gate']==t['natural_gate']+v
    assert t['actual_bf16_delta_norm']>0 and t['intended_delta_norm']>0
   positions+=len(ts);positive+=1
assert controls==12
out={'status':'passed','locked_cases':6,'native_zero_exact_matches':controls,'positive_trials':positive,'positive_processed_positions_nonzero_actual_bf16_edit':positions,'unchanged_first_output_tokens':len(firsts),'cohort_locked_before_all_steering_jobs':True,'every_case_tested_all_coarse_strengths':True,'maximum_sixteen_positive_strengths_per_case':True,'source_hashes_verified':True,'raw_sha256':sha(R/'artifacts/all-steering.json'),'review_sha256':sha(R/'reviewed-judgments.json'),'cohort_lock_sha256':sha(R/'cohort-lock.json'),'method':'Prompt untouched; native MLP retained; add current low-rank output coefficient at every processed output position. Attention unfrozen.','limits':'Content review is Codex manual review, condition labels visible, not independent human validation.'}
(R/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
