"""Audit every trial, token timing, gate coefficient, edit norm and native anchor."""
import hashlib,json,math,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve()
D=json.loads((R/'artifacts/results.json').read_text());P=json.loads((R/'protocol.json').read_text())
PRIOR=json.loads((R/'prior-anchors.json').read_text())
C=json.loads((R/'case.json').read_text());A=json.loads((R/'zero-anchors.json').read_text())
assert D['status']=='complete' and D['protocol']==P and D['case']==C
assert D['base_files_hash_verified'] and D['attention_unmodified']
assert [r['trial'] for r in D['rows']]==P['trials'] and len(D['rows'])==len(P['trials'])
for name,digest in D['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
B=D['boundaries'];assert (B['user_first_token'],B['user_last_token'],B['prompt_token_count'])==(33,72,82)
assert [t['token_id'] for t in B['prompt_tokens']]==C['prompt_token_ids']
assert ''.join(t['token'] for t in B['prompt_tokens'] if t['segment']=='user')==C['user']
checked=edits=negative_at_inactive=0;controls=[];rows=[]
for row in D['rows']:
 t=row['trial'];assert row['answer_sha256']==hashlib.sha256(row['answer'].encode()).hexdigest()
 assert row['contains_exact_secret']==(P['secret_marker'] in row['answer'])
 assert row['token_identical_to_baseline']==(row['output_token_ids']==C['output_token_ids'])
 if t['mode'] in ['native','observe']:
  assert row['output_token_ids']==C['output_token_ids'];controls.append(t['id'])
 if t['mode']=='zero':
  assert row['output_token_ids']==A[t['scope']]['output_token_ids'];controls.append(t['id'])
 if t['id'] in PRIOR:assert row['output_token_ids']==PRIOR[t['id']]['output_token_ids']
 if t['scope']=='output_only':assert row['output_token_ids'][0]==C['output_token_ids'][0]
 assert len(row['traces'])==(0 if t['mode']=='native' else 3*(82+len(row['output_token_ids'])-1))
 nonzero=0;max_ratio=0;negative_positions=0;inactive_edits=0
 for x in row['traces']:
  selected=(t['mode'] in ['zero','additive'] and x['layer'] in t['layers'] and
            (t['scope']=='output_only' and x['segment']=='response' or
             t['scope']=='user_only' and x['segment']=='user' or
             t['scope']=='user_to_end' and x['full_chat_position']>=33))
  assert x['patch_applied']==selected
  g=x['natural_raw'] if x['natural_raw']>max(0,x['threshold']) else 0
  assert x['natural_gate']==g
  coeff=(-g if t['mode']=='zero' else t['offset']) if selected else 0
  assert x['gate_delta']==coeff
  assert math.isclose(x['effective_estimated_gate'],g+coeff,rel_tol=2e-6,abs_tol=1e-6)
  assert math.isclose(x['intended_delta_norm'],abs(coeff)*x['lowrank_norm'],rel_tol=4e-6,abs_tol=1e-7)
  if not selected:assert x['actual_bf16_delta_norm']==0 and coeff==0
  if x['segment']=='before_user':assert not selected
  if x['actual_bf16_delta_norm']>0:
   nonzero+=1;max_ratio=max(max_ratio,x['actual_bf16_delta_norm']/max(x['native_mlp_norm'],1e-30))
   if g==0:inactive_edits+=1
  if selected and x['effective_estimated_gate']<0:negative_positions+=1
  checked+=1
 for f in P['features']:
  key=f"L{f['layer']}-T{f['transform']}";xs=[x for x in row['traces'] if x['feature']==key]
  if t['mode']!='native':
   assert [x['token_id'] for x in xs if x['segment']!='response']==C['prompt_token_ids']
   response=[x for x in xs if x['segment']=='response']
   assert [x['token_id'] for x in response]==row['output_token_ids'][:-1]
   assert [x['answer_position'] for x in response]==list(range(len(response)))
 edits+=nonzero;negative_at_inactive+=inactive_edits
 rows.append({'trial_id':t['id'],'actual_nonzero_layer_token_edits':nonzero,'negative_effective_gate_positions':negative_positions,'edits_at_naturally_inactive_gates':inactive_edits,'max_actual_edit_to_native_mlp_norm':max_ratio})
proof={'passed':True,'trials':len(D['rows']),'negative_interventions':sum(r['trial']['mode']=='additive' for r in D['rows']),'exact_native_and_prior_zero_controls':controls,'verified_layer_token_records':checked,'actual_nonzero_layer_token_edits':edits,'edits_at_naturally_inactive_gates':negative_at_inactive,'system_header_unedited':True,'all_timing_masks_and_coefficients_verified':True,'source_hashes_verified':True,'attention_unfrozen_native_mlp_retained':True,'rows':rows}
(R/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='rows'},indent=2))
