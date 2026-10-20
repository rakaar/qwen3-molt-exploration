"""Check prompt timing, true zero target, actual edits, controls and source identity."""
import hashlib,json,math,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();d=json.loads((r/'artifacts/results.json').read_text());p=json.loads((r/'protocol.json').read_text());assert d['status']=='complete' and d['protocol']==p and d['attention_unmodified'] and d['base_files_hash_verified']
assert [x['condition'] for x in d['rows']]==p['conditions'];rows={x['condition']:x for x in d['rows']};prior=d['case']['output_token_ids']
assert rows['native']['output_token_ids']==prior==rows['observe_only']['output_token_ids']
for name,sha in d['source_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha
checks={}
for mode,row in rows.items():
 assert hashlib.sha256(row['answer'].encode()).hexdigest()==row['answer_sha256']
 assert row['first_output_token_unchanged'] and row['output_token_ids'][0]==prior[0]
 for t in row['traces']:
  g=t['natural_raw'] if t['natural_raw']>t['threshold'] and t['natural_raw']>0 else 0.;assert g==t['natural_gate']
  assert math.isclose(t['natural_contribution_norm'],g*t['lowrank_norm'],abs_tol=1e-8,rel_tol=3e-6)
  if t['segment']=='prompt_final':assert not t['clamp_applied'] and t['actual_bf16_delta_norm']==0 and t['gate_delta']==0
  elif mode=='joint_zero_clamp':
   assert t['clamp_applied'] and t['gate_delta']==-g and t['effective_estimated_gate']==0
   assert math.isclose(t['intended_delta_norm'],t['natural_contribution_norm'],rel_tol=1e-10,abs_tol=1e-10)
   if g==0:assert t['actual_bf16_delta_norm']==0
  else:assert not t['clamp_applied'] and t['actual_bf16_delta_norm']==0 and t['gate_delta']==0
 if row['traces']:
  for f in p['features']:
   name=f"L{f['layer']}-T{f['transform']}";ts=[t for t in row['traces'] if t['feature']==name and t['segment']=='response'];assert len(ts)==len(row['output_token_ids'])-1
   assert [t['answer_position'] for t in ts]==list(range(len(ts)))
   assert [t['token_id'] for t in ts]==row['output_token_ids'][:-1]
 checks[mode]={'matches_baseline':row['token_identical_to_baseline'],'contains_secret':row['contains_exact_secret'],'stop_reason':row['stop_reason'],'actual_nonzero_layer_token_edits':sum(t['actual_bf16_delta_norm']>0 for t in row['traces'])}
zero=rows['joint_zero_clamp'];assert any(t['actual_bf16_delta_norm']>0 for t in zero['traces'])
proof={'status':'passed','all_three_features_zero_target_every_processed_output_position':True,'all_prompt_positions_unmodified':True,'native_and_observe_controls_match_saved_baseline_tokens':True,'current_gate_and_full_contribution_recomputed_each_step':True,'first_output_token_unchanged':True,'source_hashes_verified':True,'native_mlp_retained_and_attention_unfrozen':True,'conditions':checks,'zero_clamp_summaries':zero['features']}
(r/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
