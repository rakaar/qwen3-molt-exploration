"""Audit subtraction vectors, edit coverage and previous-run reproduction on CPU."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'remote-export'
D = json.loads((RUN / 'artifacts/results.json').read_text())
P = json.loads((RUN / 'protocol.json').read_text())
cases = {c['id']: c for c in json.loads((RUN / 'cases.json').read_text())}
previous = json.loads((ROOT.parent / '2026-10-19-ch-child-input-ablation/remote-export/artifacts/results.json').read_text())
prior = {(r['case_id'],r['condition_id']):r for r in previous['rows']}
assert D['status'] == 'complete' and len(D['rows']) == 8
assert D['ablation_confined_to_requested_child_positions'] and D['all_generated_child_word_occurrences_covered']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, digest in D['source_sha256'].items():
    assert sha(RUN / name) == digest
for item in json.loads((ROOT / 'source-manifest-before-execution.json').read_text()):
    assert sha(RUN / item['file']) == item['sha256']
for item in json.loads((RUN / 'download-manifest.json').read_text()):
    path = RUN / item['file']
    assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
assets = {a['transform']: a for a in D['assets']['features']}
for asset in assets.values():
    assert sha(RUN / 'weights' / asset['file']) == asset['sha256']
with np.load(RUN / 'weights' / assets[1372]['file']) as arrays:
    w = {k: arrays[k].copy() for k in arrays.files}


def bf16_round(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


summary, audit, seen = [], [], set()
for row in D['rows']:
    key = row['case_id'], row['condition_id']
    assert key not in seen
    seen.add(key)
    case = cases[row['case_id']]
    g, s = row['generation'], row['baseline_continuation_scores']
    cond = g['condition']
    assert g['prompt_token_ids'] == s['prompt_token_ids'] == case['prompt_token_ids']
    assert g['input_target_position'] == s['input_target_position'] == 27
    assert s['output_token_ids'] == case['output_token_ids']
    for trial in [g,s]:
        assert trial['attention_functions_unmodified']
        assert trial['all_generated_child_word_occurrences_covered']
        expected = [] if cond['feature'] is None else [27]
        if cond['edit_scope'] == 'input_and_generated':
            expected.extend(x['full_position'] for x in trial['generated_child_word_occurrences'])
        assert [e['position'] for e in trial['edits']] == expected
        assert trial['processed_response_positions_edited'] == sum(e['segment']=='response' and e['actual_bf16_delta_norm']>0 for e in trial['edits'])
        for edit in trial['edits']:
            assert edit['feature'] == 1372 and edit['token'].strip().casefold() == 'child'
            assert edit['other_positions_at_intervention_boundary_identical']
            if cond['removal_fraction'] == 1:
                assert edit['effective_estimated_gate'] == 0
                # All actual matching child positions in these trajectories must be active.
                assert edit['natural_gate'] > 0 and edit['actual_bf16_delta_norm'] > 0
            else:
                assert edit['actual_bf16_delta_norm'] == 0
    assert len(g['vector_files']) == len(g['edits'])
    for vector, edit in zip(g['vector_files'],g['edits']):
        path = RUN / 'artifacts' / vector['file']
        assert sha(path) == vector['sha256'] and vector['position'] == edit['position']
        with np.load(path) as arrays:
            v = {k: arrays[k].copy() for k in arrays.files}
        raw = float(np.dot(v['z'],w['e']) + w['bias'][0])
        gate = raw if raw > float(w['theta'][0]) and raw > 0 else 0.
        assert math.isclose(gate,float(v['gate'][0]),abs_tol=3e-5,rel_tol=3e-5)
        low = ((v['z'] @ w['V']) @ w['U']) * w['output_std']
        assert np.allclose(low,v['lowrank'],atol=3e-5,rtol=1e-4)
        delta = -np.float32(cond['removal_fraction']) * v['gate'][0] * v['lowrank']
        assert np.array_equal(delta,v['intended_delta'])
        assert np.array_equal(v['edited_mlp'],bf16_round(v['native_mlp']+delta))
        assert np.array_equal(v['actual_delta'],v['edited_mlp']-v['native_mlp'])
        assert math.isclose(float(np.linalg.norm(v['actual_delta'])),edit['actual_bf16_delta_norm'],rel_tol=1e-5,abs_tol=1e-5)
        audit.append({'case_id':row['case_id'],'condition':row['condition_id'],
            'independent_cpu_gate':gate,'independent_cpu_lowrank_max_abs_error':float(np.max(np.abs(low-v['lowrank']))),
            'bf16_rounded_subtraction_exact':True,**edit})
    if row['condition_id'] == 'native':
        assert g['native_generation_unhooked'] and not g['edits']
    if row['condition_id'] in ['native','zero_control','input_only']:
        prior_cond = 'ablate_B' if row['condition_id']=='input_only' else row['condition_id']
        prev = prior[row['case_id'],prior_cond]
        assert g['output_token_ids'] == prev['generation']['output_token_ids']
        assert g['token_scores'] == prev['generation']['token_scores']
        assert s['token_scores'] == prev['baseline_continuation_scores']['token_scores']
    letters = [x for x in s['token_scores'] if x['token'].strip().lower()=='ch' and not x['eos']]
    assert len(letters)==1
    letter = letters[0]
    summary.append({'case_id':row['case_id'],'condition':row['condition_id'],
        'baseline_answer':case['answer'],'answer':g['answer'],
        'same_answer_tokens_as_baseline':g['output_token_ids']==case['output_token_ids'],
        'conditional_gold_letter_token':letter['token'],
        'conditional_gold_letter_probability':letter['probability'],
        'gold_letter_probability_prefix':[x['token'] for x in s['token_scores'][:letter['output_position']]],
        'nonzero_edits':[e for e in g['edits'] if e['actual_bf16_delta_norm']>0],
        'processed_response_positions_edited':g['processed_response_positions_edited'],
        'generated_child_word_occurrences':g['generated_child_word_occurrences'],
        'sum_baseline_continuation_log_probability_without_eos':s['sum_log_probability_without_eos']})

for case_id in cases:
    rows = {r['condition_id']:r for r in D['rows'] if r['case_id']==case_id}
    assert rows['native']['baseline_continuation_scores']['token_scores'] == rows['zero_control']['baseline_continuation_scores']['token_scores']
    if not rows['input_only']['generation']['generated_child_word_occurrences']:
        assert rows['input_only']['generation']['token_scores'] == rows['input_and_generated']['generation']['token_scores']
        assert rows['input_only']['baseline_continuation_scores']['token_scores'] == rows['input_and_generated']['baseline_continuation_scores']['token_scores']
    else:
        # The first emitted child precedes the first generated-token edit.
        assert rows['input_only']['generation']['token_scores'][0] == rows['input_and_generated']['generation']['token_scores'][0]

proof = {'status':'passed','cases':2,'fixed_conditions_per_case':4,'generations':8,
    'conditional_scoring_replays':8,'previous_native_zero_and_input_only_exactly_reproduced':True,
    'source_and_weight_hashes_verified':True,'download_manifest_hashes_verified':True,
    'independent_cpu_gate_and_lowrank_checks_passed':True,
    'bf16_actual_subtraction_verified_from_saved_vectors':True,
    'all_generated_child_word_occurrences_covered':True,
    'only_requested_child_positions_edited':True,
    'uppercase_two_ablation_scopes_exactly_agree':True,
    'vector_audits':audit}
(ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'summary.csv').open('w') as stream:
    fields=['case_id','condition','baseline_answer','answer','conditional_gold_letter_probability','same_answer_tokens_as_baseline','processed_response_positions_edited']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in summary)


def esc(value):
    return html.escape(str(value))


trs,details=[],[]
for row,original in zip(summary,D['rows']):
    changes='; '.join(f"{e['segment']} {e['position']}: gate {e['natural_gate']:.6f}, edit norm {e['actual_bf16_delta_norm']:.6f}" for e in row['nonzero_edits']) or 'none'
    values=[row['case_id'],row['condition'],row['answer'],f"{100*row['conditional_gold_letter_probability']:.6f}%",changes]
    trs.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in values)+'</tr>')
    g=original['generation'];traces=g['traces'] or original['baseline_continuation_scores']['traces']
    trace_rows=['<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [t['segment'],t['position'],repr(t['token']),t['features']['L3-T2334']['gate'],t['features']['L3-T1372']['gate'],t['edit_eligible']])+'</tr>' for t in traces]
    details.append('<details><summary>'+esc(row['case_id']+' / '+row['condition'])+'</summary><pre>'+esc(g['user'])+'</pre><p>Actual greedy answer:</p><pre>'+esc(g['answer'])+'</pre><p>Edits, including zero-strength attempts:</p><pre>'+esc(json.dumps(g['edits'],indent=2))+'</pre><table><thead><tr><th>Segment</th><th>Position</th><th>Token</th><th>CH gate</th><th>Child gate</th><th>Selected for edit</th></tr></thead><tbody>'+''.join(trace_rows)+'</tbody></table></details>')
unchanged=all(r['same_answer_tokens_as_baseline'] for r in summary if r['condition']=='input_and_generated')
finding='Neither answer changed after ablating at the input and every generated child token.' if unchanged else 'At least one answer changed after ablating at the input and every generated child token; compare the exact generations below.'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Child MOLT ablation at input and generated tokens</title>
<style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.55 system-ui,sans-serif;color:#172f3b;background:#fbfcfd}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4dfe6;text-align:left;padding:9px;white-space:pre-wrap}th{background:#eaf1f5}pre{white-space:pre-wrap;background:#eef3f6;padding:12px}details{border:1px solid #d4dfe6;background:white;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:650}.tablewrap{overflow-x:auto}</style>
<h1>Child MOLT ablation at input and generated tokens</h1><p>Qwen3-4B · L3–T1372 · 19 October 2026</p><p><strong>'''+esc(finding)+'''</strong></p>
<p>Two unchanged few-shot spelling prompts, four fixed conditions, greedy decoding. Full ablation subtracts the current natural estimated Child MOLT contribution from the native layer-3 MLP at input child (position 27) and every processed generated child token. The actual edited BF16 vectors and token coverage are independently checked. CH/ch answer tokens are unedited.</p>
<h2>Answers and conditional letter scores</h2><div class="tablewrap"><table><thead><tr><th>Prompt</th><th>Condition</th><th>Actual greedy answer</th><th>P(correct letter token given baseline prefix)</th><th>Nonzero edits</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table></div>
<p>Uppercase probability is P(CH) at the first output position. Lowercase probability is P( ch | child:), evaluated with the same baseline prefix in separate scoring replays. In the full-ablation condition the generated child is also edited during that replay. Conditional probabilities are not overall task success probabilities.</p>
<h2>Timing and interpretation</h2><p>The uppercase baseline emits only CH, so both nonzero ablation scopes should be identical; the audit confirms this. The lowercase baseline first emits child, followed by : ch. Editing the generated child cannot undo that emitted word, but it can change subsequent predictions.</p>
<p>The intervention removes an estimated MOLT component while preserving the rest of the native MLP, including its reconstruction remainder. Unchanged answers show this component was not necessary for these two answers under this intervention. They neither establish nor rule out absorbed CH information: redundancy and the native remainder may retain it. A changed answer would also require specificity controls before attributing it to CH information. This pilot has no active norm-matched controls or probe-aligned component interventions.</p>
<h2>Exact prompts and token traces</h2><p>Native generation is unhooked; its displayed observations come from the separate baseline-continuation replay. All other displayed traces are from actual greedy generation.</p>'''+''.join(details)+'''
<p><a href="summary.json">Summary JSON</a> · <a href="summary.csv">Summary CSV</a> · <a href="verification.json">Independent verification</a> · <a href="remote-export/artifacts/results.json">Complete results</a></p></html>'''
(ROOT/'index.html').write_text(page)
print(json.dumps({'verification':proof['status'],'cases':2,'generations':8,'conditional_scoring_replays':8,'audited_vectors':len(audit)},indent=2))
for row in summary:
    print(row['case_id'],row['condition'],repr(row['answer']),row['conditional_gold_letter_probability'],[(e['segment'],e['position'],e['natural_gate'],e['actual_bf16_delta_norm']) for e in row['nonzero_edits']])
