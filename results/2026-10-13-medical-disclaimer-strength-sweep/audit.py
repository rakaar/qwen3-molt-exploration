import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
reference={r['prompt_id']:r for r in json.loads((ROOT/'reference-clean.json').read_text())}
all_rows=[]
files=sorted((ROOT/'artifacts').glob('sweep-*-completion.json'))
assert files
for file in files:
 c=json.loads(file.read_text())
 assert c['status']=='complete' and all(c['clean_generated_ids_match_previous_run'].values())
 assert c['source_sha256']==hashlib.sha256((ROOT/'high_strength.py').read_bytes()).hexdigest()
 assert c['intervention_source_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()=='16fff2055e900d6781419d0e3f9cb64fd312142c39dcb7bc8aff5872a90f7f3c'
 assert c['selected_weight_sha256']==hashlib.sha256((ROOT/'weights/molt-19-selected.safetensors').read_bytes()).hexdigest()
 rows=json.loads((ROOT/'artifacts'/f"sweep-{c['stage']}.json").read_text())
 assert len(rows)==c['trial_count']
 for row in rows:
  assert row['scope']=='all_tokens' and row['mode']=='target'
  assert [a['position'] for a in row['activations']]==list(range(len(row['activations'])))
  if row['factor']==1:assert row['generated_token_ids']==reference[row['prompt_id']]['generated_token_ids'] and not row['edited_positions']
  for a in row['edited_positions']:
   expected=abs(row['factor']-1)*a['target_contribution_norm']
   assert math.isfinite(a['intended_delta_norm']) and abs(expected-a['intended_delta_norm'])/max(1,expected)<2e-6
  if row['prompt_id']=='stomach_plain':
   assert row['activation_summary']['maximum_gate']==0 and not row['edited_positions']
   assert row['generated_token_ids']==reference[row['prompt_id']]['generated_token_ids']
 all_rows.extend(rows)
(ROOT/'combined-results.json').write_text(json.dumps(all_rows,indent=2)+'\n')
audit={'status':'passed','stages':[json.loads(f.read_text())['stage'] for f in files],'generation_count':len(all_rows),'all_clean_token_ids_match_previous_run':True,'unchanged_intervention_code':True,'source_and_weight_hashes_match':True,'all_recorded_edit_norms_match_factor_formula':True,'non_eos_trials':[r['id'] for r in all_rows if r['stop_reason']!='eos'],'limits':'Checks recorded scalar norms and preserved source/weights; clinical correctness and full-vector numerical reconstruction are not assessed.'}
(ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
