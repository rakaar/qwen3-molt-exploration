"""Independently audit saved intervention vectors and export readable results."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
EXPORT = WORKSPACE / 'comparisons/2026-10-28-ch-coactivation/remote-export/ch-coactivation-20261028'
OUT = WORKSPACE / 'results/2026-10-28-ch-coactivation'


def bf16_round(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def final_answer_atom(text):
    match = re.search(r'''(?<![A-Za-z])([A-Za-z]{2})[ \t\r\n.!?'"`]*$''', text)
    return match.group(1) if match else None


def main():
    OUT.mkdir(exist_ok=True)
    data = json.loads((EXPORT / 'artifacts/results.json').read_text())
    assert data['status'] == 'complete' and data['scan_complete']
    assert data['hooks_remaining'] == 0 and data['generated_positions_edited'] == 0
    assert data['child_feature_manipulated'] is False
    assert data['scan_prompts'] == 64 and len(data['scan']) == 64
    for name, expected in data['source_sha256'].items():
        assert hashlib.sha256((EXPORT / name).read_bytes()).hexdigest() == expected, name
    weights = {}
    for feature in data['protocol']['features']:
        path = EXPORT / 'weights' / feature['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == feature['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            weights[feature['layer']] = {k: arrays[k].astype(np.float64) for k in arrays.files}
    errors = {'max_raw_gate_abs_error': 0., 'max_operator_relative_l2_error': 0.,
              'max_probability_reconstruction_error': 0.}
    vector_count, run_count, previous_count = 0, 0, 0

    def audit_run(run):
        nonlocal vector_count, run_count
        run_count += 1
        assert run['generated_positions_edited'] == 0 and run['attention_unmodified']
        for step in run['steps']:
            p = sum(math.exp(x['logit'] - step['logsumexp']) for x in step['target_tokens'])
            err = abs(p - step['target_probability_mass'])
            errors['max_probability_reconstruction_error'] = max(errors['max_probability_reconstruction_error'], err)
            assert err < 4e-6
            assert 0 <= step['expected_prefix_probability_mass'] <= 1.000001
        for edit in run['edits']:
            assert edit['transform'] == 2334 and edit['layer'] in {3, 29}
            assert edit['position'] in run['target_positions']
            assert edit['all_other_positions_identical_at_edit_boundary']
        for vector in run['vectors']:
            path = EXPORT / 'artifacts' / vector['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == vector['sha256']
            with np.load(path, allow_pickle=False) as arrays:
                x = {k: arrays[k] for k in arrays.files}
            w = weights[vector['layer']]
            raw = x['z'].astype(np.float64) @ w['e'] + w['bias'][0]
            err = float(np.max(np.abs(raw - x['raw'])))
            errors['max_raw_gate_abs_error'] = max(errors['max_raw_gate_abs_error'], err)
            assert np.allclose(raw, x['raw'], atol=3e-5, rtol=3e-5)
            gate = np.where((raw > w['theta'][0]) & (raw > 0), raw, 0.)
            assert np.allclose(gate, x['gate'], atol=3e-5, rtol=3e-5)
            operator = (x['z'].astype(np.float64) @ w['V'] @ w['U']) * w['output_std']
            err = float(np.linalg.norm(operator - x['operator']) / max(np.linalg.norm(operator), 1e-12))
            errors['max_operator_relative_l2_error'] = max(errors['max_operator_relative_l2_error'], err)
            assert err < 1e-4
            delta = -np.float32(vector['removed_fraction']) * x['gate'][:, None] * x['operator']
            assert np.array_equal(delta, x['intended_delta'])
            assert np.array_equal(bf16_round(x['native_mlp'] + delta), x['edited_mlp'])
            assert np.array_equal(x['edited_mlp'] - x['native_mlp'], x['actual_delta'])
            if vector['removed_fraction'] == 0:
                assert np.array_equal(x['native_mlp'], x['edited_mlp'])
            vector_count += 1

    scan = []
    for row in data['scan']:
        native, zero, s = row['native'], row['zero_control'], row['summary']
        assert native['steps'] == zero['steps']
        assert native['output_token_ids'] == zero['output_token_ids']
        audit_run(native)
        audit_run(zero)
        if s['previous_native_baseline_exact'] is not None:
            assert s['previous_native_baseline_exact']
            previous_count += 1
        active = {layer: any(e['gate'] > 0 for e in zero['edits'] if e['layer'] == layer) for layer in (3, 29)}
        assert s['both_active'] == all(active.values())
        assert s['eligible_for_ablation'] == (all(active.values()) and row['case']['group'] == 'ch_words')
        s['expected_answer'] = row['case']['expected']
        s['native_final_answer_atom'] = final_answer_atom(native['answer'])
        s['native_final_answer_correct'] = s['native_final_answer_atom'] == s['expected_answer']
        scan.append(s)
    assert previous_count == 12
    eligible_ids = [s['case_id'] for s in scan if s['eligible_for_ablation']]
    assert eligible_ids == data['eligible_case_ids'] == [a['case']['id'] for a in data['ablations']]
    summaries, comparisons = [], []
    for row in data['ablations']:
        s = row['native_summary']
        native = next(x['native'] for x in data['scan'] if x['case']['id'] == row['case']['id'])
        record = {'case_id': row['case']['id'], 'word': row['case']['word'],
                  'letter_case': row['case']['letter_case'], 'native_answer': native['answer'],
                  'matched_prefix': s['answer_slot']['prefix'],
                  'native_p_answer_slot': native['steps'][s['answer_slot']['index']]['target_probability_mass']}
        for condition, runs in row['conditions'].items():
            free, forced, summary = runs['generation'], runs['same_native_prefix'], runs['summary']
            audit_run(free)
            audit_run(forced)
            assert forced['output_token_ids'] == native['output_token_ids']
            assert free['edits'] == forced['edits']
            summary['expected_answer'] = row['case']['expected']
            summary['native_final_answer_atom'] = final_answer_atom(native['answer'])
            summary['edited_final_answer_atom'] = final_answer_atom(free['answer'])
            summary['final_answer_letters_changed'] = summary['native_final_answer_atom'] != summary['edited_final_answer_atom']
            record[condition] = summary
            comparisons.append({'case_id': record['case_id'], 'word': record['word'],
                                'letter_case': record['letter_case'], **summary})
        p = record['native_p_answer_slot']
        a = record['L3_off']['p_answer_slot']
        b = record['L29_off']['p_answer_slot']
        joint = record['both_off']['p_answer_slot']
        record['probability_scale_interaction_pp'] = 100 * (joint - a - b + p)
        record['joint_vs_stronger_single_pp'] = 100 * (joint - min(a, b))
        natural = [g for g in s['gates'] if g['layer'] == 29]
        after = [g for g in record['both_off']['gates'] if g['layer'] == 29]
        record['L29_gates_native'] = [x['gate'] for x in natural]
        record['L29_gates_after_L3_off'] = [x['gate'] for x in after]
        record['L29_remains_active_after_L3_off'] = any(x['gate'] > 0 for x in after)
        summaries.append(record)
    assert len(comparisons) == data['intervention_comparisons'] == 3 * len(eligible_ids)
    expected_vectors = 128 + 6 * len(eligible_ids)
    assert vector_count == expected_vectors
    counts = {}
    for group in ['ch_words', 'non_ch_control']:
        rows = [s for s in scan if s['group'] == group]
        counts[group] = {'prompts': len(rows), 'active_L3': sum(s['active_L3'] for s in rows),
                         'active_L29': sum(s['active_L29'] for s in rows),
                         'both_active': sum(s['both_active'] for s in rows),
                         'both_active_same_token': sum(s['both_active_same_token'] for s in rows),
                         'native_final_answer_correct': sum(s['native_final_answer_correct'] for s in rows)}
    validation = {'status': 'passed', 'scan_prompts': 64, 'eligible_CH_prompts': len(eligible_ids),
                  'intervention_comparisons': len(comparisons), 'cpu_vector_files_audited': vector_count,
                  'runs_audited': run_count, 'previous_native_baselines_exact': previous_count,
                  'zero_controls_exact': 64, 'all_BF16_edits_exact': True,
                  'full_greedy_response_changes': sum(s['greedy_response_changed'] for s in comparisons),
                  'final_answer_letter_changes': sum(s['final_answer_letters_changed'] for s in comparisons),
                  'hooks_remaining': 0, 'child_feature_manipulated': False, **errors}
    write_json('validation.json', validation)
    write_json('activation-summary.json', {'counts': counts, 'cases': scan})
    write_json('ablation-summary.json', summaries)
    with (OUT / 'activation.csv').open('w', newline='') as stream:
        cols = ['case_id', 'word', 'letter_case', 'group', 'active_L3', 'active_L29', 'both_active',
                'both_active_same_token', 'eligible_for_ablation', 'native_answer', 'gates']
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        for s in scan:
            writer.writerow({k: json.dumps(s[k]) if isinstance(s[k], list) else s[k] for k in cols})
    with (OUT / 'ablations.csv').open('w', newline='') as stream:
        cols = ['case_id', 'word', 'letter_case', 'condition', 'native_answer', 'answer', 'greedy_response_changed',
                'native_p_first_output', 'p_first_output', 'native_p_answer_slot', 'p_answer_slot',
                'answer_slot_change_pp', 'native_log_odds_answer_slot', 'log_odds_answer_slot',
                'native_final_answer_atom', 'edited_final_answer_atom', 'final_answer_letters_changed']
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparisons)
    lines = ['# CH coactivation and conditional joint ablation — 28 October 2026', '',
             'The activation scan preceded ablation selection: 24 fixed CH words and eight non-CH controls, each in the original uppercase/lowercase few-shot task (64 prompts). The original 12 prompts were retained exactly. Only CH prompts with both selected gates naturally active on the input target word were eligible for the three intervention conditions.', '',
             'Only L3–T2334 and L29–T2334 were manipulated. Each intervention subtracts the estimated current naturally gated contribution from the native MLP at input target-word tokens. In the joint condition, the L29 gate and operator are recomputed downstream of the L3 edit. Native attention, other positions at each edit boundary, generated tokens, and the native MLP reconstruction remainder are preserved.', '',
             '| Group | Prompts | L3 active | L29 active | Both active | Both active at same token |',
             '|---|---:|---:|---:|---:|---:|']
    for group, count in counts.items():
        lines.append(f"| {group} | {count['prompts']} | {count['active_L3']} | {count['active_L29']} | {count['both_active']} | {count['both_active_same_token']} |")
    lines += ['', '## Eligible prompts: matched-prefix answer-slot P(CH/ch)', '',
              'Probability percentages sum the plain and one-leading-space CH/ch single-token variants. For words echoed by the native model, every condition uses the identical native continuation prefix before scoring the final two-letter answer. These are conditional next-token masses, not full-sequence correctness probabilities. First-output measurements and log odds are preserved in the CSV.', '',
              '| Prompt | Native response | Same prefix | Native % | L3 off % | L29 off % | Both off % | Joint vs stronger single, pp | Joint response |',
              '|---|---|---|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        lines.append(f"| {s['case_id']} | `{s['native_answer']}` | `{s['matched_prefix'] or '(empty)'}` | {100*s['native_p_answer_slot']:.6f} | {100*s['L3_off']['p_answer_slot']:.6f} | {100*s['L29_off']['p_answer_slot']:.6f} | {100*s['both_off']['p_answer_slot']:.6f} | {s['joint_vs_stronger_single_pp']:+.6f} | `{s['both_off']['answer']}` |")
    changes = [s for s in comparisons if s['greedy_response_changed']]
    lines += ['', f"Full greedy text changed in {len(changes)} of {len(comparisons)} comparisons; final extracted two-letter answers changed in {validation['final_answer_letter_changes']} comparisons.", '']
    for s in changes:
        lines.append(f"- {s['case_id']}, {s['condition']}: `{s['native_answer']}` → `{s['answer']}`.")
    incorrect = [s for s in scan if not s['native_final_answer_correct']]
    lines += ['', 'Baseline failures are retained, rather than filtered out. Final two-letter correctness by scan group: '
              + '; '.join(f"{g}: {c['native_final_answer_correct']}/{c['prompts']}" for g, c in counts.items()) + '.', '']
    for s in incorrect:
        lines.append(f"- {s['case_id']}: native `{s['native_answer']}`; expected `{s['expected_answer']}`.")
    lines += ['', '## Native activation for every word', '',
              '| Word | Upper L3 / L29 gates | Lower L3 / L29 gates |', '|---|---|---|']
    for word in dict.fromkeys(s['word'] for s in scan):
        parts = []
        for case in ['upper', 'lower']:
            s = next(x for x in scan if x['word'] == word and x['letter_case'] == case)
            parts.append(' / '.join(', '.join(f"{g['token']}:{g['gate']:.4f}" for g in s['gates'] if g['layer'] == layer)
                                    for layer in [3, 29]))
        lines.append(f'| {word} | {parts[0]} | {parts[1]} |')
    lines += ['', '## What this follow-up shows', '',
              'The strongest prior decrease replicated: lowercase chair falls from 70.646262% to 45.779493% after L3 removal. Joint removal reaches 48.700592%, so it does not deepen that decrease. Uppercase cheat also loses immediate CH mass, but the response changes to an echoed word followed by the same final CH answer. Uppercase cheek gains CH probability after removal. No final extracted two-letter answer changes in the 27 intervention comparisons.', '',
              'Both gates coactivate on six distinct CH words in at least one output format: chin, chair, cheek, cheese, cheat, and chimney. The counts are prompt-level, not six independent replications of a uniform effect. Gates also depend on tokenization: the shared initial che token has identical activation in uppercase cheek, cheese, and cheat; in chimney the L3 gate activates on im rather than the initial ch token.', '',
              '## Interpretation limits', '',
              'The joint intervention measures the effect of removing current estimated contributions, including changes to the downstream L29 gate/operator induced by L3 removal. It does not hold the L29 contribution frozen to its baseline value. Probability-scale interaction is descriptive and depends on the metric scale; it is not by itself evidence of redundancy or feature absorption.', '',
              'Selection used native coactivation, so these ablation results describe that selected subset. Inactive cases remain visible in the activation table. Non-CH controls were scanned for activation only; no active norm-matched specificity intervention was run. Full-layer MOLT reconstruction fidelity was not evaluated. No Child-feature intervention was performed.', '',
              f"Validation: {vector_count} saved vector files independently reconstructed on CPU; all BF16 edits exact, 64 exact zero controls, and all 12 previous native baselines exact. Hooks removed.", '',
              f"[Probability figure](<{OUT/'probability-comparison.png'}>) · [Activation CSV](<{OUT/'activation.csv'}>) · [Ablation CSV](<{OUT/'ablations.csv'}>) · [Validation](<{OUT/'validation.json'}>) · [Raw export](<{EXPORT}>)", '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'validation': validation, 'counts': counts}, indent=2))
    for s in summaries:
        print(s['case_id'], 'P_SLOT', *(round(100 * p, 6) for p in [s['native_p_answer_slot'], s['L3_off']['p_answer_slot'], s['L29_off']['p_answer_slot'], s['both_off']['p_answer_slot']]),
              'RESPONSE', repr(s['both_off']['answer']), 'L29_GATES', s['L29_gates_native'], '=>', s['L29_gates_after_L3_off'])


if __name__ == '__main__':
    main()
