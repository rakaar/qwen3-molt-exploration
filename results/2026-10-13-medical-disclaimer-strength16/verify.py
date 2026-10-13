import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=ROOT.parent/'2026-10-13-medical-disclaimer'
assert hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()==hashlib.sha256((old/'run.py').read_bytes()).hexdigest()
manifest=json.loads((ROOT/'artifacts/export-checksums.json').read_text())
for name,sha in manifest.items():
 with (ROOT/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
c=json.loads((ROOT/'artifacts/extension-completion.json').read_text())
assert c['extension_source_sha256']==hashlib.sha256((ROOT/'extension.py').read_bytes()).hexdigest()
assert c['intervention_source_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert all(r['clean_ids_identical_to_all_identity_rows'] for r in c['checks'].values())
trials=json.loads((ROOT/'artifacts/extension-trials.json').read_text());identities=json.loads((ROOT/'artifacts/extension-identities.json').read_text())
assert len(trials)==24 and len(identities)==24
assert {r['factor'] for r in trials}=={1,0,-8,-16}
error=0
for r in trials+identities:
 assert r['mode']=='target' and r['scope']=='all_tokens'
 assert [a['position'] for a in r['activations']]==list(range(len(r['activations'])))
 if r['factor']==1:assert not r['edited_positions']
 for a in r['activations']:
  values=[a[k] for k in ['target_gate','target_lowrank_norm','target_contribution_norm','original_mlp_norm']]
  assert all(math.isfinite(v) for v in values)
  expected=a['target_gate']*a['target_lowrank_norm']
  e=abs(expected-a['target_contribution_norm'])/max(1,expected);error=max(error,e);assert e<2e-6
 for a in r['edited_positions']:
  expected=abs(r['factor']-1)*a['target_contribution_norm'];e=abs(expected-a['intended_delta_norm'])/max(1,expected)
  error=max(error,e);assert e<2e-6
inactive=[r for r in trials if r['prompt_id']=='stomach_plain']
assert all(r['activation_summary']['maximum_gate']==0 and not r['edited_positions'] for r in inactive)
assert all(r['generated_token_ids']==inactive[0]['generated_token_ids'] for r in inactive)
old_clean={r['prompt_id']:r for r in json.loads((old/'artifacts/paired-negative.json').read_text()) if r['factor']==1}
identical={r['prompt_id']:r['generated_token_ids']==old_clean[r['prompt_id']]['generated_token_ids'] for r in trials if r['factor']==1}
r={'status':'passed','payload_hash_count':len(manifest),'source_unchanged_from_original':True,'clean_identity_checks':c['checks'],'clean_continuations_identical_to_previous_rental':identical,'trial_count':len(trials),'identity_generation_count':len(identities),'maximum_recorded_norm_error':error,'all_trial_generations_reached_eos':all(r['stop_reason']=='eos' for r in trials),'non_eos_trials':[r['id'] for r in trials if r['stop_reason']!='eos'],'inactive_prompt_has_zero_edits_and_identical_output':True,'limits':'Audits recorded scalar norms and source/scope. Does not recompute full intervention vectors or assess clinical correctness.'}
(ROOT/'local-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
