"""CPU audit of recorded interventions, controls, source hashes and answer-bound reviews."""
import argparse,hashlib,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);R=ap.parse_args().root
read=lambda n:json.loads((R/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases={c['id']:c for c in read('cases.json')};P=read('protocol.json');assert len(cases)==10
raw=read('artifacts/all-results.json');reviews=read('reviewed-judgments.json');grades={x['trial_id']:x for x in reviews['rows']}
assert len(raw['rows'])==len({x['trial_id'] for x in raw['rows']})
assert set(grades)=={x['trial_id'] for x in raw['rows']}
for part in raw['parts']:
 f=R/part['file'];assert sha(f)==part['sha256'];d=json.loads(f.read_text());assert d['status']=='complete' and d['base_files_hash_verified'];assert d['protocol']==P
 for name,digest in d['source_sha256'].items():assert sha(R/name)==digest,name
 assert sha(R/part['job_file'])==d['job_sha256']
controls=positive=positions=0
for c in cases.values():
 rows=[x for x in raw['rows'] if x['case_id']==c['id']]
 assert sum(x['added_gate'] is None for x in rows)==1 and sum(x['added_gate']==0 for x in rows)==1
 strengths=[x['added_gate'] for x in rows if x['added_gate'] is not None and x['added_gate']>0]
 assert len(strengths)==len(set(strengths)) and len(strengths)<=P['max_positive_trials_per_case']
 assert set(P['coarse_strengths'])<=set(strengths)
 for x in rows:
  assert x['prompt_token_ids']==c['prompt_token_ids'] and x['chat']==c['chat']
  assert x['first_output_token_unchanged'] and x['output_token_ids'][0]==c['output_token_ids'][0]
  g=grades[x['trial_id']];assert g['answer_sha256']==hashlib.sha256(x['answer'].encode()).hexdigest()
  for key in ['coherent','explicit_abstention','substantive_answer','false_abstention']:
   assert isinstance(g[key],bool)
  assert g['false_abstention']==bool(g['coherent'] and g['explicit_abstention'] and not g['substantive_answer'])
  if g['strict_clean_abstention']:assert g['false_abstention'] and not g['false_passage_justification'] and not g['additional_false_facts']
  a=x['added_gate'];ts=x['traces']
  if a is None or a==0:
   controls+=1;assert x['native_or_zero_matches_baseline'] and x['output_token_ids']==c['output_token_ids'];assert all(t['actual_bf16_delta_norm']==0 for t in ts)
  else:
   positive+=1;assert ts[0]['segment']=='prompt_final' and ts[0]['gate_delta']==ts[0]['actual_bf16_delta_norm']==0
   assert len(ts)-1==len(x['output_token_ids'])-1
   for i,t in enumerate(ts[1:]):
    assert t['segment']=='response' and t['eligible'] and t['answer_position']==i
    assert t['full_chat_position']==len(c['prompt_token_ids'])+i and t['token_id']==x['output_token_ids'][i]
    assert t['gate_delta']==a and t['added_gate']==a and t['effective_estimated_gate']==t['natural_gate']+a
    assert math.isclose(t['intended_delta_norm'],a*t['lowrank_norm'],rel_tol=3e-6,abs_tol=1e-6)
    assert t['actual_bf16_delta_norm']>0
   assert x['summary']['edited_positions']==len(ts)-1;positions+=len(ts)-1
assert controls==20
successes=sum(any(grades[x['trial_id']]['false_abstention'] for x in raw['rows'] if x['case_id']==c and x['added_gate'] is not None and x['added_gate']>0) for c in cases)
strict=sum(any(grades[x['trial_id']]['strict_clean_abstention'] for x in raw['rows'] if x['case_id']==c and x['added_gate'] is not None and x['added_gate']>0) for c in cases)
out={'status':'passed','cohort':10,'native_correct_questions':10,'native_zero_exact_matches':controls,'positive_trials':positive,'actual_nonzero_response_edits':positions,'coherent_false_abstention_cases':successes,'strict_no_false_explanation_cases':strict,'all_first_output_tokens_native':True,'source_hashes_verified':True,'all_coarse_strengths_tested_per_case':True,'reviews_answer_hash_bound':True,'attention':'unfrozen eager','review_limit':'Codex manual, condition-visible; no independent blinded human review','raw_sha256':sha(R/'artifacts/all-results.json'),'review_sha256':sha(R/'reviewed-judgments.json')}
(R/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
