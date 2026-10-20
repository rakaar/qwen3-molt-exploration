"""Validate exact timing masks, zero targets, genuine edits, controls and sources."""
import hashlib,json,math,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();d=json.loads((r/'artifacts/results.json').read_text());p=json.loads((r/'protocol.json').read_text());b=d['boundaries'];c=d['case'];prior=json.loads((r/'prior-output-only.json').read_text());assert d['status']=='complete' and d['protocol']==p and d['attention_unmodified'] and d['base_files_hash_verified']
assert [x['condition'] for x in d['rows']]==p['conditions'];rows={x['condition']:x for x in d['rows']};assert rows['native']['output_token_ids']==c['output_token_ids'];assert rows['output_only']['output_token_ids']==prior['output_token_ids']
for name,sha in d['source_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha
assert [t['token_id'] for t in b['prompt_tokens']]==c['prompt_token_ids'];assert ''.join(t['token'] for t in b['prompt_tokens'] if t['segment']=='user')==c['user']
checks={}
for mode,row in rows.items():
 assert hashlib.sha256(row['answer'].encode()).hexdigest()==row['answer_sha256']
 counts={}
 for t in row['traces']:
  raw=t['natural_raw'];g=raw if raw>t['threshold'] and raw>0 else 0.;assert g==t['natural_gate'];assert math.isclose(t['natural_contribution_norm'],g*t['lowrank_norm'],rel_tol=3e-6,abs_tol=1e-8)
  expected=mode=='user_to_end' and t['full_chat_position']>=b['user_first_token'] or mode=='user_only' and t['segment']=='user' or mode=='output_only' and t['segment']=='response'
  assert t['clamp_applied']==expected
  if expected:
   assert t['gate_delta']==-g and t['effective_estimated_gate']==0
   assert math.isclose(t['intended_delta_norm'],t['natural_contribution_norm'],abs_tol=1e-10,rel_tol=1e-10)
   if g==0:assert t['actual_bf16_delta_norm']==0
  else:assert t['gate_delta']==0 and t['actual_bf16_delta_norm']==0
  if t['segment']=='before_user':assert not expected
 for f in p['features']:
  name=f"L{f['layer']}-T{f['transform']}";ts=[t for t in row['traces'] if t['feature']==name]
  if mode!='native':
   pt=[t for t in ts if t['segment']!='response'];at=[t for t in ts if t['segment']=='response'];assert [t['token_id'] for t in pt]==c['prompt_token_ids'];assert [t['token_id'] for t in at]==row['output_token_ids'][:-1];assert [t['answer_position'] for t in at]==list(range(len(at)))
  counts[name]={seg:sum(t['actual_bf16_delta_norm']>0 for t in ts if t['segment']==seg) for seg in ['before_user','user','post_user_prompt','response']}
 checks[mode]={'token_identical_to_baseline':row['token_identical_to_baseline'],'contains_secret':row['contains_exact_secret'],'stop_reason':row['stop_reason'],'first_changed_output_position':row['first_changed_output_position'],'actual_nonzero_edits':counts}
proof={'status':'passed','native_and_output_only_prior_tokens_reproduced':True,'user_boundaries_exactly_reconstruct_original_user_text':True,'all_system_positions_unedited':True,'all_timing_masks_verified':True,'all_selected_coefficients_clamped_to_zero':True,'current_gate_and_lowrank_contribution_recomputed':True,'source_hashes_verified':True,'attention_unfrozen_native_mlp_retained':True,'boundaries':{k:b[k] for k in ['user_first_token','user_last_token','user_token_count','prompt_token_count']},'conditions':checks}
(r/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
