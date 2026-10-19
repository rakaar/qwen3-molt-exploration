"""Independent CPU audit of input-only subtraction and local result report."""
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
assert D['status'] == 'complete' and len(D['rows']) == 8
assert D['all_sanity_controls_match_prior_native_tokens'] and D['ablation_confined_to_input_child']
for name, digest in D['source_sha256'].items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == digest
for item in json.loads((RUN / 'download-manifest.json').read_text()):
    p = RUN / item['file']
    assert p.stat().st_size == item['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
assets = {a['transform']: a for a in D['assets']['features']}
for asset in assets.values():
    assert hashlib.sha256((RUN / 'weights' / asset['file']).read_bytes()).hexdigest() == asset['sha256']


def bf16_round(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


summary, audit, seen = [], [], set()
for row in D['rows']:
    key = (row['case_id'], row['condition_id'])
    assert key not in seen
    seen.add(key)
    case = cases[row['case_id']]
    g, s = row['generation'], row['baseline_continuation_scores']
    assert g['prompt_token_ids'] == s['prompt_token_ids'] == case['prompt_token_ids']
    assert g['input_target_position'] == s['input_target_position'] == 27
    assert g['processed_response_positions_edited'] == s['processed_response_positions_edited'] == 0
    assert g['attention_functions_unmodified'] and s['attention_functions_unmodified']
    assert s['output_token_ids'] == case['output_token_ids']
    if row['condition_id'] == 'native':
        assert g['native_generation_unhooked'] and not g['edits']
    if row['condition_id'] in ['native', 'zero_control', 'ablate_A_inactive_sanity']:
        assert g['output_token_ids'] == case['output_token_ids']
    vector_audit = None
    if g['vector_file']:
        path = RUN / 'artifacts' / g['vector_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == g['vector_sha256']
        with np.load(path) as arrays:
            v = {k: arrays[k].copy() for k in arrays.files}
        condition = g['condition']
        asset = assets[condition['feature']]
        with np.load(RUN / 'weights' / asset['file']) as arrays:
            w = {k: arrays[k].copy() for k in arrays.files}
        raw = float(np.dot(v['z'], w['e']) + w['bias'][0])
        gate = raw if raw > float(w['theta'][0]) and raw > 0 else 0.
        assert math.isclose(gate, float(v['gate'][0]), abs_tol=3e-5, rel_tol=3e-5)
        low = ((v['z'] @ w['V']) @ w['U']) * w['output_std']
        assert np.allclose(low, v['lowrank'], atol=3e-5, rtol=1e-4)
        expected_delta = -np.float32(condition['removal_fraction']) * v['gate'][0] * v['lowrank']
        assert np.array_equal(expected_delta, v['intended_delta'])
        assert np.array_equal(v['edited_mlp'], bf16_round(v['native_mlp'] + expected_delta))
        assert np.array_equal(v['actual_delta'], v['edited_mlp'] - v['native_mlp'])
        assert len(g['edits']) == len(s['edits']) == 1
        edit = g['edits'][0]
        assert edit['other_positions_at_intervention_boundary_identical']
        assert edit['position'] == 27 and edit['token'] == 'child'
        if row['condition_id'] == 'ablate_B':
            assert edit['actual_bf16_delta_norm'] > 0 and edit['effective_estimated_gate'] == 0
        else:
            assert np.count_nonzero(v['actual_delta']) == 0
        vector_audit = {'case_id': row['case_id'], 'condition_id': row['condition_id'],
            'independent_cpu_gate': gate, 'saved_gate': float(v['gate'][0]),
            'independent_cpu_lowrank_max_abs_error': float(np.max(np.abs(low-v['lowrank']))),
            'bf16_rounded_subtraction_exact': True, **edit}
        audit.append(vector_audit)
    letters = [x for x in s['token_scores'] if x['token'].strip().lower() == 'ch' and not x['eos']]
    assert len(letters) == 1
    letter_score = letters[0]
    current = {'case_id': row['case_id'], 'condition': row['condition_id'],
        'baseline_answer': case['answer'], 'answer': g['answer'],
        'same_answer_tokens_as_baseline': g['output_token_ids'] == case['output_token_ids'],
        'stop_reason': g['stop_reason'],
        'conditional_gold_letter_token': letter_score['token'],
        'conditional_gold_letter_probability': letter_score['probability'],
        'gold_letter_probability_prefix': [x['token'] for x in s['token_scores'][:letter_score['output_position']]],
        'sum_baseline_continuation_log_probability_without_eos': s['sum_log_probability_without_eos'],
        'input_edit': g['edits'][0] if g['edits'] else None,
        'response_Child_gate_traces': [{'token': x['token'], 'position': x['position'],
            'gate': x['features']['L3-T1372']['gate']} for x in g['traces'] if x['segment']=='response']}
    summary.append(current)

for case_id in cases:
    rows = {r['condition']:r for r in summary if r['case_id']==case_id}
    for c in ['zero_control', 'ablate_A_inactive_sanity']:
        assert rows[c]['conditional_gold_letter_probability'] == rows['native']['conditional_gold_letter_probability']
proof = {'status':'passed','cases':2,'fixed_conditions_per_case':4,'generations':8,
    'conditional_scoring_replays':8,'native_and_noop_controls_match_prior_baseline':True,
    'source_and_weight_hashes_verified':True,'download_manifest_hashes_verified':True,
    'independent_cpu_gate_and_lowrank_checks_passed':True,
    'bf16_actual_subtraction_verified_from_saved_vectors':True,
    'only_input_position_27_edited':True,'response_tokens_edited':0,
    'B_ablation_actual_delta_norms':[r['input_edit']['actual_bf16_delta_norm'] for r in summary if r['condition']=='ablate_B'],
    'vector_audits':audit}
(ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'summary.csv').open('w') as stream:
    fields=['case_id','condition','baseline_answer','answer','conditional_gold_letter_probability','same_answer_tokens_as_baseline']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in summary)

def esc(value):return html.escape(str(value))

trs=[];details=[]
for row, original in zip(summary,D['rows']):
    edit=row['input_edit'];delta=edit['actual_bf16_delta_norm'] if edit else 0
    values=[row['case_id'],row['condition'],row['answer'],f"{row['conditional_gold_letter_probability']:.6f}",f'{delta:.6f}']
    trs.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in values)+'</tr>')
    g=original['generation']
    trace_rows=['<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [t['segment'],t['position'],repr(t['token']),t['features']['L3-T2334']['gate'],t['features']['L3-T1372']['gate']])+'</tr>' for t in g['traces']]
    details.append('<details><summary>'+esc(row['case_id']+' / '+row['condition'])+'</summary><pre>'+esc(g['user'])+'</pre><p>Actual answer:</p><pre>'+esc(g['answer'])+'</pre><p>Input edit:</p><pre>'+esc(json.dumps(edit,indent=2))+'</pre><table><thead><tr><th>Segment</th><th>Position</th><th>Token</th><th>CH gate</th><th>Child gate</th></tr></thead><tbody>'+''.join(trace_rows)+'</tbody></table></details>')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Input Child MOLT ablation</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.55 system-ui,sans-serif;color:#172f3b;background:#fbfcfd}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4dfe6;text-align:left;padding:9px;white-space:pre-wrap}th{background:#eaf1f5}pre{white-space:pre-wrap;background:#eef3f6;padding:12px}details{border:1px solid #d4dfe6;background:white;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:650}.tablewrap{overflow-x:auto}</style>
<h1>Input Child MOLT ablation</h1><p>Qwen3-4B · layer 3 · input token child at zero-based position 27 · 19 October 2026</p>
<p>Two unchanged few-shot prompts requesting uppercase or lowercase first-two-letter answers. One fixed removal: subtract the natural L3-T1372 contribution at the input child token from native MLP output. All response tokens remain unedited; attention and other native computation continue normally. No strength search.</p>
<h2>Answers and conditional letter scores</h2><div class="tablewrap"><table><thead><tr><th>Prompt</th><th>Condition</th><th>Actual greedy answer</th><th>P(correct letter token given baseline prefix)</th><th>Actual input edit norm</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table></div>
<p>For uppercase, the probability is P(CH) at the first output position. For lowercase, it is P( ch | child:), with the saved baseline prefix teacher-forced. That lowercase conditional score can benefit from the unedited generated child token; it is not the probability of answering correctly from the prompt alone. Actual greedy answers are shown separately.</p>
<h2>Scope and interpretation</h2><p>The native MLP is retained; the intervention subtracts an estimated MOLT component while preserving the reconstruction remainder. Zero-strength and inactive-CH-feature ablations are implementation sanity checks, not norm-matched active specificity controls. A behavioral effect alone does not establish selective removal of CH information.</p>
<h2>Exact prompts, edits and observed token traces</h2>'''+''.join(details)+'''
<p><a href="summary.json">Summary JSON</a> · <a href="summary.csv">Summary CSV</a> · <a href="verification.json">Independent verification</a> · <a href="remote-export/artifacts/results.json">Complete results</a></p></html>'''
(ROOT/'index.html').write_text(page)
print(json.dumps({'verification':proof['status'],'cases':2,'generations':8,'conditional_scoring_replays':8},indent=2))
for row in summary:print(row['case_id'],row['condition'],repr(row['answer']),row['conditional_gold_letter_probability'],row['input_edit'])
