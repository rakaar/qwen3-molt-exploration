"""Verify the predeclared hidden-before-L comparison without changing wording."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/2026-11-10-before-l-description'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
plan=json.loads((OUT/'heldout-plan.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
freeze=json.loads((HERE/'protocol.json').read_text())
assert sha(HERE/'protocol.json')==protocol['explanation_freeze_sha256']
assert sha(OUT/'heldout-plan.json')==protocol['heldout_plan_sha256']
for arm in ['baseline','candidate']:
 assert hashlib.sha256(plan[arm+'_explanation'].encode()).hexdigest()==freeze[arm+'_sha256']
assert json.loads((OUT/'judge-settings-comparison.json').read_text())['all_match']
audit=[json.loads(s) for s in (OUT/'audit.jsonl').read_text().splitlines()]
assert len(audit)==448
resume=json.loads((OUT/'resume-provenance.json').read_text())
assert resume['prior_rows']==230 and resume['new_unique_requests']==110
assert sha(OUT/'prior-audit.jsonl')==resume['prior_audit_sha256']
assert sha(OUT/'prior-prepared-inputs.json')==resume['prior_prepared_sha256']
assert sha(OUT/'prior-audit.jsonl')==sha(OUT/'partial-before-resume/audit.jsonl')
assert sha(OUT/'prior-prepared-inputs.json')==sha(OUT/'partial-before-resume/prepared-inputs.json')
assert json.loads((OUT/'partial-before-resume/judge-settings-comparison.json').read_text())['all_match']
prior=[json.loads(s) for s in (OUT/'prior-audit.jsonl').read_text().splitlines()]
new=[json.loads(s) for s in (OUT/'new-unique-audit.jsonl').read_text().splitlines()]
assert len(prior)==230 and len(new)==110
cache={r['request_sha256']:r for r in prior}
assert len(cache)==resume['prior_unique']==226
for row in new:
 assert row['request_sha256'] not in cache
 cache[row['request_sha256']]=row
assert len(cache)==336
for row in audit:
 recorded=cache[row['request_sha256']]
 for key in ['messages','prompt_token_ids','response','response_token_ids','finish_reason','prediction','at_utc']:
  assert row[key]==recorded[key],key
index={(r['method'],r['row_index'],r['arm']):r for r in audit};assert len(index)==448
results={};witnesses=[]
feature_path=ROOT/'comparisons/2026-11-10-next-anticipatory-feature/feature-1440.json'
feature=json.loads(feature_path.read_text());assert sha(feature_path)==protocol['feature_source_sha256']
recovered=json.loads((OUT/'recovered-windows.json').read_text())
selection=json.loads((OUT/'selection-before-scoring.json').read_text())
chosen={r['fingerprint'] for r in selection['selected'] if not r['earlier']}
assert len(chosen)==28
assert {c['source_fingerprint'] for c in plan['cases'] if c['method']=='fuzz' and c['activating']}==chosen
for method in protocol['methods']:
 details=[]
 for case in [c for c in plan['cases'] if c['method']==method]:
  original=feature['methods'][method]['rows'][case['original_row_index']]
  assert original['activating']==case['activating']
  if method!='detection':
   assert all((original['activations'][i]>0)==case['activating'] for i in original['highlighted_positions'])
  entry={k:case[k] for k in ['row_index','original_row_index','context','activating','source_fingerprint']}
  for arm in ['baseline','candidate']:
   row=index[(method,case['row_index'],arm)]
   assert row['messages']==case[arm+'_messages'] and row['activating']==case['activating']
   assert row['finish_reason']=='stop' and row['response'].strip() in ['[0]','[1]']
   assert row['prediction']==(row['response'].strip()=='[1]')
   entry[arm]=row['prediction']
  details.append(entry)
 contexts={}
 for context in protocol['contexts']:
  positive=[r for r in details if r['context']==context and r['activating']]
  negative=[r for r in details if r['context']==context and not r['activating']]
  assert len(positive)==28
  contexts[context]=dict(positive_total=28,control_total=len(negative),
   **{a:dict(positive_correct=sum(r[a] for r in positive),control_false_positives=sum(r[a] for r in negative)) for a in ['baseline','candidate']})
 for before in [r for r in details if r['context']=='prefix_only' and r['activating']]:
  after=next(r for r in details if r['original_row_index']==before['original_row_index'] and r['context']!='prefix_only')
  if not before['candidate'] and after['candidate']:
   window=recovered['windows'][before['source_fingerprint']]
   last=next(a for a in window['matches'][0]['active_positions'] if a['window_position']==len(window['str_tokens'])-1)
   witnesses.append(dict(method=method,original_row_index=before['original_row_index'],peak_token=last['token'],gate=last['gate'],
    prefix=last['prefix'],following=last['following_text'],old_prefix=before['baseline'],old_extended=after['baseline'],new_prefix=False,new_extended=True,
    clean_four_way_pattern=not before['baseline'] and not after['baseline']))
 results[method]=dict(contexts=contexts,per_case=details)
duplicate={}
for r in audit:
 if r['request_sha256'] in duplicate:assert duplicate[r['request_sha256']]==r['prediction']
 duplicate[r['request_sha256']]=r['prediction']
assert len(duplicate)==336
selection=json.loads((OUT/'selection-before-scoring.json').read_text())
report=dict(feature='L34-T1440',verified_at_utc=datetime.now(timezone.utc).isoformat(),responses=448,unique_request_bodies=336,
 resumed_from_preserved_partial=True,prior_logical_decisions=230,new_unique_queries=110,actual_calls_across_attempts=340,
 descriptions_frozen=True,exact_planned_messages_and_labels_verified=True,no_invalid_or_truncated_responses=True,
 original_positive_contexts=50,strict_before_L_contexts=39,fully_hidden_before_L_contexts_scored=28,
 methods=results,context_rescue_witnesses=witnesses)
save(OUT/'analysis.json',report)
lines=['# L34–T1440: a second anticipatory-feature evaluation diagnostic','',
 'The candidate description was frozen from training-partition dashboard windows before its evaluation passages were inspected. This is an exploratory follow-up, not a wholly unseen or preregistered feature evaluation.','',
 '## Coverage','',
 '- 39/50 original positive evaluation windows end immediately before an L/l-leading continuation.','- 28/39 have no earlier active-before-L example visible in the prefix. All 28 have unambiguous recovered source continuations, and all originally highlighted tokens match the before-L pattern.','- Those 28 were selected by the visibility rule before any new judge calls, not by which responses improved. Both descriptions are unchanged.','',
 '## Fresh results: positive cases correct','',
 '| Test | Old prefix | New prefix | Old +16 source tokens | New +16 source tokens |','|---|---:|---:|---:|---:|']
for method,m in results.items():
 a=m['contexts']['prefix_only'];b=m['contexts']['plus_16_source_tokens']
 lines.append(f"| {method} | {a['baseline']['positive_correct']}/28 | {a['candidate']['positive_correct']}/28 | {b['baseline']['positive_correct']}/28 | {b['candidate']['positive_correct']}/28 |")
lines+=['','## Active Fuzz controls','']
for context in protocol['contexts']:
 g=results['active_fuzz']['contexts'][context]
 lines.append(f"- {context}: old {g['baseline']['control_false_positives']}/28 false positives; new {g['candidate']['control_false_positives']}/28.")
lines+=['','Fuzz and Active Fuzz have the same positive requests; these are not independent replications. Counts concern the selected positive contexts and the matching Active Fuzz controls, not the original full 100-case benchmark.','',
 '## Exact original description','',plan['baseline_explanation'],'',
 '## Exact revised description, frozen before evaluation inspection','',plan['candidate_explanation'],'',
 '## Rescued cases','']
for w in witnesses:
 lines.append(f"- {w['method']} row {w['original_row_index']}: {w['peak_token']!r} -> {w['following'][:70]!r}; old prefix/extended {int(w['old_prefix'])}/{int(w['old_extended'])}, new prefix/extended 0/1.")
lines+=['','## Scope and verification','',
 '- The same pinned fresh Qwen3.8-27B-NVFP4 local judge and inference arithmetic as the preceding T2375 study were used. Both descriptions and both context conditions were newly evaluated. Historical original-dashboard scores remain separate.','- Original authenticated feature data supply actual gate values, ground-truth labels and highlight positions. Source windows are matched to the original held-out UltraChat conversations; the candidate wording was not tuned to these passages.','- Shared verified Delphi system/few-shot messages are reused. Final user messages follow the same original scoring template; this is not claimed as a byte-verified historical prompt-export replay.','- Sixteen following original source tokens are appended. Gate labels and original highlight positions stay unchanged. Future context cannot change the already recorded gate at an earlier token of a causal language model.','- Native subject-model generation or contribution ablation was not run. This tests explanation recognition by the judge, not the causal function of the MOLT.','- All 448 logical responses are valid and untruncated; there are 336 unique prompt bodies. Identical prompt bodies have identical predictions. Downloaded result files are hash-verified.','',
 'The original run exited after 230 decisions; its cause was not established. Those decisions and its full prepared query plan were separately downloaded and hash-verified. The resumed attempt made only 110 unfinished unique queries. Identical completed requests, including shared positive Fuzz/Active Fuzz inputs, were reused with their original responses. There were 340 actual calls across the two attempts, covering 336 unique requests and all 448 planned logical decisions. The original partial audit is preserved unchanged.','',
 '[Original feature](https://amsterdam.taild80f17.ts.net/layers/layer-34/features/1440.html) · [User-linked activation report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-03-eleven-results-index/experiments/downstream-activation-examples/)','']
(OUT/'README.md').write_text('\n'.join(lines))
save(OUT/'local-manifest.json',dict(files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='local-manifest.json']))
print(json.dumps(dict(methods={m:v['contexts'] for m,v in results.items()},rescued_positive_decisions=len(witnesses)),indent=2))
