"""Curate and render the later coactivation-selected joint-ablation study."""
import csv
import html
import json
import shutil
from pathlib import Path


def collect(workspace):
    source = workspace / 'results/2026-10-28-ch-coactivation'
    activation = json.loads((source / 'activation-summary.json').read_text())
    ablations = json.loads((source / 'ablation-summary.json').read_text())
    validation = json.loads((source / 'validation.json').read_text())
    assert validation['status'] == 'passed' and validation['eligible_CH_prompts'] == 9
    scan_keys = ['case_id', 'word', 'letter_case', 'group', 'active_L3', 'active_L29', 'both_active',
                 'both_active_same_token', 'eligible_for_ablation', 'native_answer', 'gates',
                 'expected_answer', 'native_final_answer_atom', 'native_final_answer_correct']
    common = ['case_id', 'word', 'letter_case', 'native_answer', 'matched_prefix', 'native_p_answer_slot',
              'probability_scale_interaction_pp', 'joint_vs_stronger_single_pp', 'L29_gates_native',
              'L29_gates_after_L3_off', 'L29_remains_active_after_L3_off']
    condition_keys = ['condition', 'native_answer', 'answer', 'greedy_response_changed', 'answer_slot',
                      'native_p_answer_slot', 'p_answer_slot', 'answer_slot_change_pp',
                      'native_p_first_output', 'p_first_output', 'native_log_odds_answer_slot',
                      'log_odds_answer_slot', 'gates', 'expected_answer', 'native_final_answer_atom',
                      'edited_final_answer_atom', 'final_answer_letters_changed']
    rows = []
    for original in ablations:
        row = {k: original[k] for k in common}
        for condition in ['L3_off', 'L29_off', 'both_off']:
            row[condition] = {k: original[condition][k] for k in condition_keys}
        rows.append(row)
    cases = json.loads((workspace / 'experiments/ch-coactivation-20261028/cases.json').read_text())
    assert len(rows) == 9 and len(cases) == 64
    assert all(r['L29_remains_active_after_L3_off'] for r in rows)
    return {'counts': activation['counts'], 'scan': [{k: r[k] for k in scan_keys} for r in activation['cases']],
            'ablations': rows, 'cases': cases, 'validation': validation,
            'runtime': {'python': '3.11.14', 'torch': '2.14.0+cu126', 'transformers': '4.57.6',
                        'gpu': 'RTX 3090', 'native_dtype': 'BF16', 'molt_dtype': 'FP32', 'attention': 'eager'}}


def render(data, output, figure_source):
    e = html.escape
    (output / 'joint-data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'joint-validation.json').write_text(json.dumps(data['validation'], indent=2) + '\n')
    (output / 'joint-prompts.json').write_text(json.dumps(data['cases'], indent=2) + '\n')
    if figure_source.resolve() != (output / 'joint-probabilities.svg').resolve():
        shutil.copy2(figure_source, output / 'joint-probabilities.svg')
    for file, rows in [('joint-activation.csv', data['scan']),
                       ('joint-comparisons.csv', [dict(case_id=r['case_id'], word=r['word'],
                                                      letter_case=r['letter_case'], **r[c])
                                                  for r in data['ablations'] for c in ['L3_off', 'L29_off', 'both_off']])]:
        with (output / file).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})
    probability_rows, response_rows = [], []
    for r in data['ablations']:
        cells = [r['native_p_answer_slot']] + [r[c]['p_answer_slot'] for c in ['L3_off', 'L29_off', 'both_off']]
        probability_rows.append(f'<tr data-case="{e(r["case_id"])}"><td><code>{e(r["case_id"])}</code></td>'
                                f'<td class="joint-prefix"><code>{e(r["matched_prefix"] or "prompt alone")}</code></td>'
                                + ''.join(f'<td class="joint-p">{100*v:.6f}</td>' for v in cells)
                                + f'<td><code>{e(r["both_off"]["answer"])}</code></td></tr>')
        for c in ['L3_off', 'L29_off', 'both_off']:
            response_rows.append(f'<tr><td><code>{e(r["case_id"])}</code></td><td>{e(c)}</td>'
                                 f'<td><code>{e(r["native_answer"])}</code></td><td><code>{e(r[c]["answer"])}</code></td></tr>')
    activation_rows = []
    for word in dict.fromkeys(r['word'] for r in data['scan']):
        cells = []
        for case in ['upper', 'lower']:
            row = next(r for r in data['scan'] if r['word'] == word and r['letter_case'] == case)
            for layer in [3, 29]:
                gates = [g for g in row['gates'] if g['layer'] == layer]
                cells.append(e(' · '.join(f"{g['token']}:{g['gate']:.4f}" for g in gates)))
        activation_rows.append(f'<tr><td><code>{e(word)}</code></td>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    return '''<section class="chapter" id="joint-followup" aria-labelledby="joint-title">
<p class="eyebrow">Second study · Coactivation checked before joint intervention</p>
<h2 id="joint-title">Both gates active.<br>What changes when both are removed?</h2>
<p>We preserved the original six words and expanded to 24 CH words plus eight non-CH controls, each in the same uppercase/lowercase few-shot task. Native coactivation selected the eligible prompts before any ablation outcomes were inspected.</p>
<div class="metrics" aria-label="Joint follow-up study counts"><div><strong>64</strong><span>scanned prompts<br>48 CH + 16 non-CH controls</span></div><div><strong>9</strong><span>CH prompts with both gates active<br>all at a shared input token</span></div><div><strong>27</strong><span>nonzero intervention comparisons<br>L3 only · L29 only · both</span></div><div><strong>0</strong><span>changed final two-letter answers<br>2 full responses changed format</span></div></div>
<p>Across the 48 CH prompts, <strong>L3–T2334 was active on 11</strong>, <strong>L29–T2334 on 28</strong>, and both on nine. Neither fired on the 16 non-CH control prompts. Both were active for uppercase chin, chair, cheek, cheese, cheat and chimney; and lowercase chin, chair and chimney. Both remained inactive on child.</p>
<div class="note"><h3>Joint removal did not deepen the strongest earlier drop.</h3><p>Lowercase chair: P(ch) was <strong>70.646% native → 45.779% with L3 off → 48.701% with both off</strong>. Every joint condition removed a nonzero contribution at both layers. L29 remained active downstream of the L3 edit in all nine cases.</p></div>
<figure><a href="joint-probabilities.svg" target="_blank" rel="noopener"><img src="joint-probabilities.svg" width="1050" height="650" alt="Nine coactive CH prompts, comparing native CH probability with L3-only, L29-only and joint ablations. Lowercase chair has the largest decrease; several other prompts increase."></a><figcaption>Only the nine prompts selected by native coactivation. Conditional next-token mass at the identical native answer prefix; each dot is a deterministic measurement.</figcaption></figure>
<form class="filters" onsubmit="return false"><label>Joint-study probability position<select id="joint-metric" aria-label="Joint-study probability position"><option value="answer_slot">Answer slot · same native prefix</option><option value="first_output">First output · prompt alone</option></select></label></form>
<div class="table-wrap" tabindex="0" aria-label="Joint ablation probability comparisons"><table class="joint-table"><thead><tr><th>Prompt</th><th>Probability prefix</th><th>Native P(%)</th><th>L3 off P(%)</th><th>L29 off P(%)</th><th>Both off P(%)</th><th>Joint response</th></tr></thead><tbody id="joint-rows">''' + ''.join(probability_rows) + '''</tbody></table></div>
<p class="caption">P sums plain and one-leading-space CH/ch single-token variants. Answer-slot scoring uses the exact same native continuation prefix in every condition. It does not measure the full probability of every correct response. Select first output to inspect format changes separately.</p>
<p><strong>Responses:</strong> no final two-letter answer changed in the 27 comparisons. Uppercase cheat changed from <code>CH</code> to <code>cheat: CH</code> with L3 off and with both off, so its immediate-CH probability decrease coincides with an output-format change. The original model already answered uppercase cheese as <code>CE</code> and lowercase chip as <code>ci</code>; these baseline failures were retained. Baseline final-letter correctness was 46/48 CH prompts and 16/16 controls.</p>
<details><summary>All 27 full native and edited responses</summary><div class="table-wrap"><table class="joint-table"><thead><tr><th>Prompt</th><th>Condition</th><th>Native response</th><th>Edited response</th></tr></thead><tbody>''' + ''.join(response_rows) + '''</tbody></table></div></details>
<details><summary>Native gate values for every scanned word, including inactive cases and controls</summary><div class="table-wrap"><table class="joint-table"><thead><tr><th>Word</th><th>Upper L3</th><th>Upper L29</th><th>Lower L3</th><th>Lower L29</th></tr></thead><tbody>''' + ''.join(activation_rows) + '''</tbody></table></div><p class="caption">Token:value pairs refer to input-word positions. The shared initial che token has identical gates in uppercase cheek, cheese and cheat. In chimney, L3 activates on im rather than on the initial ch token.</p></details>
<details><summary>Joint-intervention method, verification and limits</summary><p>Each layer subtracts its estimated current naturally gated contribution at every input token overlapping the target word. In the joint condition, L29’s gate and low-rank operator are recomputed downstream of the L3 edit. The L29 contribution is not frozen to its native value. Other positions at each edit boundary, native attention functions, generated tokens, other features and the native MLP remainder are preserved.</p><p>All 64 zero controls and all 12 earlier native baselines matched exactly. An independent CPU audit reconstructed 182 saved gate/operator/edit vector files; all BF16 edits matched exactly, all hooks were removed, and no Child-feature intervention was installed. The pinned model/MOLT revisions match the first study. This run used Python 3.11.14, PyTorch 2.14.0+cu126, Transformers 4.57.6 and an RTX 3090.</p><p>Effects vary by prompt and format. Coactivation is not sufficient to predict the direction of a contribution’s output effect. This selected-subset experiment does not establish absorption or general feature redundancy. Non-CH controls were scanned for activation only; no active norm-matched specificity intervention or full-layer reconstruction-fidelity study was performed.</p></details>
<div class="download-links"><a href="joint-activation.csv" download>All 64 activation scans · CSV ↓</a><a href="joint-comparisons.csv" download>All 27 ablations · CSV ↓</a><a href="joint-data.json">Curated follow-up data ↗</a><a href="joint-prompts.json">Exact 64 prompts ↗</a><a href="joint-validation.json">Follow-up validation ↗</a><a href="../../experiments/ch-coactivation-20261028/run.py">Scientific run source ↗</a></div>
</section>'''
