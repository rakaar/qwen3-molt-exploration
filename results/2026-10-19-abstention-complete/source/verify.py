"""Independent source/trace/control/content-selection audit for all sixty cells."""
import hashlib,json,math
from pathlib import Path
S=Path(__file__).resolve().parent;W=S.parents[2];R=W/'results/2026-10-19-abstention-complete'
read=lambda p:json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
P=read(S/'protocol.json');cases={c['id']:c for c in read(S/'cases.json')};assert len(cases)==20 and sum(c['cohort']=='correct' for c in cases.values())==10
for source in read(R/'import-manifest.json'):
 assert sha(W/source['raw_file'])==source['raw_sha256'];assert sha(W/source['review_file'])==source['review_sha256']
controls=0;new=0;prompt_edits=0;response_edits=0;prefills={};firsts={};part_receipts=[]
for path in sorted((R/'artifacts').glob('*.json')):
 d=read(path);assert d['status']=='complete' and d['protocol']==P and d['base_files_hash_verified']
 for k,v in P['runtime'].items():assert d['runtime'][k]==v
 for name,h in d['source_sha256'].items():assert sha(S/name)==h
 assert sha(S/(d['job']['name']+'.json'))==d['job_sha256'] and len(d['rows'])==len(d['job']['trials'])
 for r,t in zip(d['rows'],d['job']['trials']):
  c=cases[r['case_id']];assert r['trial_id']==t['trial_id'] and r['scope']==t['scope'] and r['added_gate']==t['strength']
  assert r['question']==c['question'] and r['context']==c['context'] and r['chat']==c['chat']
  if 'prompt_token_ids' in c:assert r['prompt_token_ids']==c['prompt_token_ids']
  a=r['added_gate'];n=len(r['prompt_token_ids']);m=len(r['output_token_ids']);trace=r['traces'];scope=r['scope']
  if a is None:assert not trace
  else:
   assert len(trace)==n+m-1 and [x['full_chat_position'] for x in trace]==list(range(n+m-1))
   assert [x['token_id'] for x in trace]==r['prompt_token_ids']+r['output_token_ids'][:-1]
   for i,x in enumerate(trace):
    segment='prompt' if i<n else 'response';assert x['segment']==segment
    eligible=segment=='prompt' and scope in ['input','both'] or segment=='response' and scope in ['output','both'];expected=a if eligible else 0.
    assert x['eligible']==eligible and x['gate_delta']==expected and x['output_hook_returned_none']==(not eligible)
    assert math.isclose(x['intended_delta_norm'],expected*x['lowrank_norm'],rel_tol=3e-6,abs_tol=1e-5)
    assert math.isclose(x['effective_estimated_gate'],expected+x['natural_gate'],rel_tol=1e-6,abs_tol=1e-6)
    assert (x['actual_bf16_delta_norm']>0) if expected>0 else (x['actual_bf16_delta_norm']==0)
    if expected>0:
     if segment=='prompt':prompt_edits+=1
     else:response_edits+=1
   if a>0 and scope in ['input','both']:
    prefills[r['case_id'],a,scope]=[{k:x[k] for k in ['natural_raw','natural_gate','lowrank_norm','actual_bf16_delta_norm']} for x in trace[:n]]
    firsts[r['case_id'],a,scope]=r['output_token_ids'][0]
  if a is None or a==0:assert r['output_token_ids']==c['output_token_ids'] and r['native_or_zero_matches_baseline'];controls+=1
  else:
   new+=1
   if scope=='output':assert r['output_token_ids'][0]==c['output_token_ids'][0]
 part_receipts.append({'file':str(path.relative_to(R)),'sha256':sha(path),'rows':len(d['rows']),'elapsed_seconds':d['elapsed_seconds']})
assert controls==40
matched=0
for key,x in prefills.items():
 cid,a,scope=key
 if scope=='input' and (cid,a,'both') in prefills:
  assert firsts[key]==firsts[cid,a,'both'] and x==prefills[cid,a,'both'];matched+=1
assert matched>=140
rows=read(R/'measurements.json');assert len({r['id'] for r in rows})==len(rows)
by_source={}
for r in rows:
 assert hashlib.sha256(r['answer'].encode()).hexdigest()==r['answer_sha256'];by_source.setdefault(r['source_file'],[]).append(r)
for filename,rs in by_source.items():
 d=read(W/filename)['rows']
 for r in rs:
  src=d[r['source_row_index']];assert src['trial_id']==r['source_trial_id'] and src['case_id']==r['case_id'] and src['added_gate']==r['strength'] and src['answer']==r['answer'] and src['output_token_ids']==r['output_token_ids']
content={}
for r in rows:
 k=(r['case_id'],r['answer_sha256']);rating=tuple(r['review'][x] for x in ['coherent','success','answer_present','question_drift'])
 if k in content:assert content[k]==rating,k
 content[k]=rating
summary=read(R/'summary.json');assert summary['all_60_cells_common_grid_complete'] and len(summary['cells'])==60
for cell in summary['cells']:
 xs=[r for r in rows if r['case_id']==cell['case_id'] and r['scope']==cell['scope']];assert set(P['common_strengths'])<=set(r['strength'] for r in xs)
 good=[r for r in xs if r['review']['success']];assert bool(good)==cell['success']
 if good:assert cell['selected_strength']==min(r['strength'] for r in good)
 else:
  bad=[r['strength'] for r in xs if not r['review']['coherent']];first=min(bad) if bad else None;pre=[r['strength'] for r in xs if r['review']['coherent'] and (first is None or r['strength']<first)]
  assert cell['selected_strength']==max(pre,default=0) and cell['first_incoherent_tested_strength']==first
out={'status':'passed','fresh_native_zero_controls':controls,'new_positive_trials':new,'combined_unique_positive_trials':len(rows),'complete_case_scope_cells':60,'common_grid_points_per_cell':len(P['common_strengths']),'new_actual_prompt_edits':prompt_edits,'new_actual_output_edits':response_edits,'matched_input_both_prefills_and_first_predictions':matched,'source_answer_review_and_selection_checks':True,'parts':part_receipts}
(R/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
