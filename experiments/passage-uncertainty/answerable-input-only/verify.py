"""Verify saved source, pins, controls, input-only intervention scope and answer-bound grades."""
import hashlib,json,math,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve();S=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((S/'protocol.json').read_text());cases={c['id']:c for c in json.loads((S/'cases.json').read_text())}
parts=[];rows=[];prompt_edits=0;response_edits=0;matched_prefills=0
alltoken=json.loads((R.parent/"2026-10-19-answerable-all-token/artifacts/all-results.json").read_text())["rows"];prior={(r["case_id"],r["added_gate"]):r for r in alltoken if r["added_gate"] is not None}
for path in sorted((R/'artifacts').glob('*.json')):
 if path.name=='all-results.json':continue
 d=json.loads(path.read_text());assert d['status']=='complete',path
 assert d['protocol']==p and d['base_files_hash_verified']
 for n,v in d['source_sha256'].items():assert sha(S/n)==v,(path,n)
 assert sha(S/(d['job']['name']+'.json'))==d['job_sha256']
 assert d['runtime']['torch']==p['runtime']['torch'] and d['runtime']['transformers']==p['runtime']['transformers']
 assert len(d['rows'])==len(d['job']['trials'])
 for r,t in zip(d['rows'],d['job']['trials']):
  c=cases[r['case_id']];assert r['trial_id']==t['trial_id'] and r['added_gate']==t['strength']
  assert r['prompt_token_ids']==c['prompt_token_ids'] and r['chat']==c['chat']
  assert r['question']==c['question'] and r['context']==c['context']
  a=r['added_gate'];trace=r['traces'];nt=len(r['output_token_ids']);n=len(r['prompt_token_ids'])
  if a is None:assert not trace
  else:
   assert len(trace)==n+nt-1
   assert [x['full_chat_position'] for x in trace]==list(range(n+nt-1))
   assert [x['token_id'] for x in trace]==r['prompt_token_ids']+r['output_token_ids'][:-1]
   assert [x['segment'] for x in trace]==['prompt']*n+['response']*(nt-1)
   for x in trace:
    expected=a if x['segment']=='prompt' else 0.
    assert x['eligible']==(x['segment']=='prompt') and x['gate_delta']==expected
    assert x['output_hook_returned_none']==(x['segment']=='response')
    assert math.isclose(x['intended_delta_norm'],expected*x['lowrank_norm'],rel_tol=3e-6,abs_tol=1e-5)
    assert math.isclose(x['effective_estimated_gate'],expected+x['natural_gate'],rel_tol=1e-6,abs_tol=1e-6)
    assert x['actual_bf16_delta_norm']>0 if expected>0 else x['actual_bf16_delta_norm']==0
   if a>0:
    prompt_edits+=n;response_edits+=sum(x['actual_bf16_delta_norm']>0 for x in trace[n:])
    before=prior.get((r['case_id'],a))
    if before:
     assert r['output_token_ids'][0]==before['output_token_ids'][0]
     for x,y in zip(trace[:n],before['traces'][:n]):
      for key in ['natural_raw','natural_gate','lowrank_norm','intended_delta_norm','actual_bf16_delta_norm','native_mlp_norm']:
       assert math.isclose(x[key],y[key],rel_tol=1e-6,abs_tol=1e-6),(r['trial_id'],key)
     matched_prefills+=1
  if a is None or a==0:assert r['output_token_ids']==c['output_token_ids'] and r['native_or_zero_matches_baseline']
  rows.append(r)
 parts.append({'file':str(path.relative_to(R)),'sha256':sha(path),'rows':len(d['rows']),'elapsed_seconds':d['elapsed_seconds']})
assert len({r['trial_id'] for r in rows})==len(rows)
assert sum(r['added_gate'] is None or r['added_gate']==0 for r in rows)==20
for c in cases:
 strengths=[r['added_gate'] for r in rows if r['case_id']==c and r['added_gate'] is not None and r['added_gate']>0]
 assert len(strengths)==len(set(strengths)) and len(strengths)<=p['max_positive_trials_per_case']
 assert set(p['coarse_strengths'])<=set(strengths)
assert response_edits==0 and matched_prefills>=120
review=json.loads((R/'reviewed-judgments.json').read_text());g={x['trial_id']:x for x in review['rows']};assert len(g)==len(rows)
for r in rows:assert g[r['trial_id']]['answer_sha256']==hashlib.sha256(r['answer'].encode()).hexdigest()
(R/'artifacts/all-results.json').write_text(json.dumps({'status':'complete','parts':parts,'rows':rows},ensure_ascii=False,indent=2)+'\n')
res={'status':'passed','source_hashes_verified':True,'native_zero_replay_matches':20,'positive_trials':sum(r['added_gate'] is not None and r['added_gate']>0 for r in rows),'trials':len(rows),'actual_nonzero_prompt_edits':prompt_edits,'actual_nonzero_response_edits':response_edits,'matched_all_token_prefills_and_first_predictions':matched_prefills,'all_prompt_and_processed_response_positions_covered':True,'answer_bound_reviews':len(g),'coherent_full_abstention_cases':sorted({r['case_id'] for r in rows if g[r['trial_id']]['false_abstention']}),'parts':parts}
(R/'verification.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
