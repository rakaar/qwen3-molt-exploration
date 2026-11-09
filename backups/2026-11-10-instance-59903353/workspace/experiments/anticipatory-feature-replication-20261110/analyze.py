"""Verify paired outputs and separate before-K cases from other activations."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'results/2026-11-10-before-k-description'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
plan=json.loads((OUT/'heldout-plan.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
freeze=json.loads((HERE/'protocol.json').read_text())
assert sha(HERE/'protocol.json')==protocol['explanation_freeze_sha256']
assert sha(OUT/'heldout-plan.json')==protocol['heldout_plan_sha256']
for arm in ['baseline','candidate']:
 assert hashlib.sha256(plan[arm+'_explanation'].encode()).hexdigest()==freeze[arm+'_sha256']
assert json.loads((OUT/'judge-settings-comparison.json').read_text())['all_match']
audit=[json.loads(s) for s in (OUT/'audit.jsonl').read_text().splitlines()]
assert len(audit)==192
index={(r['method'],r['row_index'],r['arm']):r for r in audit};assert len(index)==192
result={};witnesses=[]
coverage=json.loads((OUT/'coverage.json').read_text())
cov={(r['method'],r['row_index']):r for r in coverage['rows']}
feature=json.loads((ROOT/'comparisons/2026-11-10-next-anticipatory-feature/feature-965.json').read_text())
for method in protocol['methods']:
 summary={};details=[]
 for case in [c for c in plan['cases'] if c['method']==method]:
  original=feature['methods'][method]['rows'][case['original_row_index']]
  assert original['activating']==case['activating']
  if method!='detection':
   assert all((original['activations'][i]>0)==case['activating'] for i in original['highlighted_positions'])
  entry={k:case[k] for k in ['method','row_index','original_row_index','context','activating','role','strict_latin_k_peak','cyrillic_k_peak']}
  for arm in ['baseline','candidate']:
   row=index[(method,case['row_index'],arm)]
   assert row['messages']==case[arm+'_messages'] and row['activating']==case['activating']
   assert row['finish_reason']=='stop' and row['response'].strip() in ['[0]','[1]']
   assert row['prediction']==(row['response'].strip()=='[1]')
   entry[arm]=row['prediction']
  details.append(entry)
 for context in protocol['contexts']:
  selected=[r for r in details if r['context']==context]
  groups={}
  for group,predicate in [('all_positives',lambda r:r['activating']),
       ('strict_before_latin_K',lambda r:r['activating'] and r['strict_latin_k_peak']),
       ('before_cyrillic_K',lambda r:r['activating'] and r['cyrillic_k_peak']),
       ('no_earlier_active_K_clue',lambda r:r['activating'] and r['strict_latin_k_peak'] and not cov[(method,r['original_row_index'])]['pre_k_already_visible_in_prefix']),
       ('inactive_highlight_controls',lambda r:not r['activating'])]:
   rows=[r for r in selected if predicate(r)]
   groups[group]=dict(n=len(rows),**{arm:dict(correct=sum(r[arm]==r['activating'] for r in rows),false_positives=sum(not r['activating'] and r[arm] for r in rows)) for arm in ['baseline','candidate']})
  summary[context]=groups
 for r in [x for x in details if x['context']=='prefix_only' and x['activating'] and x['strict_latin_k_peak']]:
  after=next(x for x in details if x['original_row_index']==r['original_row_index'] and x['context']!='prefix_only')
  if not r['candidate'] and after['candidate']:
   c=cov[(method,r['original_row_index'])]
   witnesses.append(dict(method=method,original_row_index=r['original_row_index'],old_prefix=r['baseline'],old_extended=after['baseline'],new_prefix=r['candidate'],new_extended=after['candidate'],peak_token=c['peak_token'],following=c['following_text'],clean_four_way_pattern=not r['baseline'] and not after['baseline']))
 result[method]=dict(summary=summary,per_case=details)
by_request={}
for row in audit:
 key=row['request_sha256']
 if key in by_request:assert by_request[key]==row['prediction']
 by_request[key]=row['prediction']
assert len(by_request)==144
report=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),feature='L34-T965',descriptions_frozen=True,
 exact_planned_messages_and_labels_verified=True,valid_untruncated_responses=192,unique_request_bodies=144,
 same_runtime_as_previous_local_judge=True,methods=result,context_rescue_witnesses=witnesses,
 rule='Strict Latin K/k classification uses the first non-whitespace character after the peak; Cyrillic к is reported separately. All 12 positive cases are retained, with this subset reported separately.')
save(OUT/'analysis.json',report)
lines=['# L34–T965: before-K explanation and truncation diagnostic','',
 'The candidate wording was frozen from training-partition top activation examples before reading this feature’s evaluation passages. This is an exploratory candidate selected from the user-linked downstream report; it is not a new preregistered or wholly unseen feature evaluation.','',
 'The original feature has 12 positive cases per method. Seven peaks immediately precede Latin K/k; one precedes Cyrillic к; four have other continuations. Five of the seven Latin-K passages already show an earlier active-before-k event inside the prefix.','',
 '## Fresh paired results: seven strict before-K cases','',
 '| Test | Old prefix | New prefix | Old +16 | New +16 |','|---|---:|---:|---:|---:|']
for method,m in result.items():
 a=m['summary']['prefix_only']['strict_before_latin_K'];b=m['summary']['plus_16_source_tokens']['strict_before_latin_K']
 lines.append(f"| {method} | {a['baseline']['correct']}/{a['n']} | {a['candidate']['correct']}/{a['n']} | {b['baseline']['correct']}/{b['n']} | {b['candidate']['correct']}/{b['n']} |")
lines+=['','## All 12 original positive cases','', '| Test | Old prefix | New prefix | Old +16 | New +16 |','|---|---:|---:|---:|---:|']
for method,m in result.items():
 a=m['summary']['prefix_only']['all_positives'];b=m['summary']['plus_16_source_tokens']['all_positives']
 lines.append(f"| {method} | {a['baseline']['correct']}/12 | {a['candidate']['correct']}/12 | {b['baseline']['correct']}/12 | {b['candidate']['correct']}/12 |")
lines+=['','## Active Fuzz controls','']
for context in protocol['contexts']:
 g=result['active_fuzz']['summary'][context]['inactive_highlight_controls']
 lines.append(f"- {context}: old {g['baseline']['false_positives']}/12 false positives; new {g['candidate']['false_positives']}/12.")
lines+=['','## Cases rescued by showing the continuation','']
for w in witnesses:
 lines.append(f"- {w['method']} row {w['original_row_index']}: token {w['peak_token']!r}, next text {w['following'][:65]!r}. Old prefix/extended: {int(w['old_prefix'])}/{int(w['old_extended'])}; new prefix/extended: {int(w['new_prefix'])}/{int(w['new_extended'])}.")
if not witnesses:lines.append('No strict before-K positive case changed from new-description incorrect to correct after adding context.')
lines+=['','## Exact descriptions','', '**Old**', '', plan['baseline_explanation'], '', '**New, frozen**','',plan['candidate_explanation'],'',
 '## Method','',
 '- Same fresh local judge and arithmetic settings as the T2375 comparison. Both descriptions and both context conditions were newly run, with temperature 0 and thinking disabled. Historical dashboard metrics are separate.','- Original feature JSON supplies exact labels, token gates and highlighted positions. Shared verified Delphi system/few-shot prefixes are reused; final user messages are reconstructed with the same template. This is not claimed as a byte-verified historical prompt-export replay.','- All 12 positives are scored in all three methods; Active Fuzz also includes all 12 matching zero-gate highlighted controls. There are 192 logical decisions and 144 unique request bodies. Positive Fuzz and Active Fuzz inputs are identical and are not independent replications.','- Sixteen original source tokens are appended; existing highlights and labels are unchanged. Future context cannot alter the previously measured gate at an earlier causal-model token.','- Original source windows are matched to the same held-out UltraChat conversations. The native subject model was not rerun, and dashboard descriptions/scores were not overwritten.','',
 '[Original dashboard](https://amsterdam.taild80f17.ts.net/layers/layer-34/features/965.html) · [User-linked report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-03-eleven-results-index/experiments/downstream-activation-examples/)','']
(OUT/'README.md').write_text('\n'.join(lines))
save(OUT/'local-manifest.json',dict(files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='local-manifest.json']))
print(json.dumps(dict(methods={m:v['summary'] for m,v in result.items()},witnesses=witnesses),indent=2))
