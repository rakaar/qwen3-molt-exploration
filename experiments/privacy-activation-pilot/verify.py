"""Recompute every gate summary and verify saved baseline and observation records."""
import csv,hashlib,json,math,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve();D=json.loads((R/'artifacts/measurements.json').read_text());B=json.loads((R/'artifacts/baseline.json').read_text());P=json.loads((R/'protocol.json').read_text());C=json.loads((R/'cases.json').read_text())
assert D['status']=='complete' and D['protocol']==P
assert D['native_generation_and_observed_replay_token_identical'] and D['all_observation_hooks_return_none'] and D['attention_unmodified']
assert [r['id'] for r in D['rows']]==P['case_ids']==[r['id'] for r in C]
assert len(D['rows'])==4 and len(P['features'])==3
A={f"L{f['layer']}-T{f['transform']}":f for f in D['assets']['features']}
base={r['id']:r for r in B['rows']};cases={r['id']:r for r in C};review=json.loads((R/'response-review.json').read_text())
count={'prompt_positions':0,'visible_response_positions':0,'nonzero_gate_measurements':0,'summary_values':0}
for row in D['rows']:
 b=base[row['id']];assert b['baseline_generation_no_hooks'] and row['all_saved_greedy_predictions_verified']
 for k in ['answer','prompt_token_ids','output_token_ids','stop_reason','chat']:assert row[k]==b[k]
 for k in ['system','user','condition','scenario']:assert row[k]==cases[row['id']][k]
 assert hashlib.sha256(row['answer'].encode()).hexdigest()==review[row['id']]['answer_sha256']
 assert [t['token_id'] for t in row['prompt_tokens']]==row['prompt_token_ids']
 assert [t['token_id'] for t in row['answer_tokens']]==row['output_token_ids'][:len(row['answer_tokens'])]
 assert all(t['full_chat_position']==j for j,t in enumerate(row['prompt_tokens']))
 assert all(t['position']==j and t['full_chat_position']==len(row['prompt_tokens'])+j for j,t in enumerate(row['answer_tokens']))
 assert ''.join(t['token'] for t in row['answer_tokens'] if not t['special'])==row['answer']
 count['prompt_positions']+=len(row['prompt_tokens']);count['visible_response_positions']+=len(row['answer_tokens'])
 for segment,rows in [('prompt',row['prompt_tokens']),('response',row['answer_tokens'])]:
  for t in rows:
   assert set(t['features'])==set(A)
   for name,v in t['features'].items():
    assert all(math.isfinite(x) for x in v.values())
    raw=v['raw'];expected=raw if raw>A[name]['theta'] and raw>0 else 0
    assert v['gate']==expected
    assert math.isclose(v['contribution_norm'],v['gate']*v['lowrank_norm'],rel_tol=3e-6,abs_tol=1e-8)
    count['nonzero_gate_measurements']+=v['gate']>0
  visible=[t for t in rows if not t['special']]
  for name in A:
   gates=[t['features'][name]['gate'] for t in visible];s=row['summaries'][name][segment]
   assert s['visible_tokens']==len(gates) and s['active_tokens']==sum(g>0 for g in gates)
   assert math.isclose(s['mean_gate'],sum(gates)/len(gates),abs_tol=1e-12) and s['max_gate']==max(gates)
   count['summary_values']+=1
for name,f in A.items():assert hashlib.sha256((R/'weights'/f['file']).read_bytes()).hexdigest()==f['sha256']
for name,sha in D['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha
proof={'status':'passed','cases':4,'features':3,'generation_unhooked':True,'read_only_replay_all_greedy_predictions_matched':True,'all_gate_values_recomputed_from_saved_raw_and_threshold':True,'means_include_zero_gates_and_exclude_special_tokens':True,'source_hashes_match_executed_files':True,'selected_weight_hashes_match':True,'native_mlp_and_attention_unmodified':True,**count,'response_cases_at_token_cap':[r['id'] for r in D['rows'] if r['stop_reason']!='eos']}
(R/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
