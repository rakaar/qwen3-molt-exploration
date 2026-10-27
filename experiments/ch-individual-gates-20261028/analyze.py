"""Audit the four separate ablations and export all responses and probabilities."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
EXPORT = WORKSPACE / 'comparisons/2026-10-28-ch-individual-gates/remote-export/ch-individual-gates-20261028'
OUT = WORKSPACE / 'results/2026-10-28-ch-individual-gates'


def bf16_round(values):
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    combined = json.loads((EXPORT / 'combined-results.json').read_text())
    reference = json.loads((EXPORT / 'baseline-reference.json').read_text())
    assert combined['status'] == 'complete' and len(combined['gate_runs']) == 4
    assert combined['simultaneous_gate_ablations'] is False and combined['hooks_remaining'] == 0
    summaries, gate_summaries = [], []
    errors = {'max_raw_gate_abs_error': 0.0, 'max_operator_relative_l2_error': 0.0,
              'max_probability_reconstruction_error': 0.0}
    vectors = 0
    for gate in combined['gate_runs']:
        name, feature = gate['name'], gate['feature']
        folder = EXPORT / name
        result = json.loads((EXPORT / gate['result_file']).read_text())
        assert result['status'] == 'complete' and result['hooks_remaining'] == 0
        assert result['only_edited_feature'] == {'layer': feature['layer'], 'transform': feature['transform']}
        assert len(result['cases']) == 12
        assert hashlib.sha256((folder / feature['file']).read_bytes()).hexdigest() == feature['sha256']
        with np.load(folder / feature['file'], allow_pickle=False) as data:
            w = {k: data[k].astype(np.float64) for k in data.files}
        current = []
        for row in result['cases']:
            native, zero = row['native'], row['zero_control']
            off, forced = row['gate_off_generation'], row['gate_off_baseline_prefix']
            case_id = row['summary']['case_id']
            assert native['output_token_ids'] == zero['output_token_ids'] == forced['output_token_ids']
            assert native['steps'] == zero['steps'] == reference[case_id]['steps']
            assert native['prompt_token_ids'] == reference[case_id]['prompt_token_ids']
            assert native['output_token_ids'] == reference[case_id]['output_token_ids']
            for run in [native, zero, off, forced]:
                assert run['generated_positions_edited'] == 0 and run['attention_functions_unmodified']
                for step in run['steps']:
                    mass = sum(math.exp(x['logit'] - step['logsumexp']) for x in step['target_tokens'])
                    error = abs(mass - step['target_probability_mass'])
                    errors['max_probability_reconstruction_error'] = max(errors['max_probability_reconstruction_error'], error)
                    assert error < 4e-6
                for edit in run['edits']:
                    assert edit['layer'] == feature['layer'] and edit['transform'] == feature['transform']
                    assert edit['position'] in run['target_positions']
                    assert edit['all_other_positions_identical_at_edit_boundary']
            for run in [zero, off]:
                path = folder / 'artifacts' / run['vector_file']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == run['vector_sha256']
                with np.load(path, allow_pickle=False) as data:
                    x = {k: data[k] for k in data.files}
                raw = x['z'].astype(np.float64) @ w['e'] + w['bias'][0]
                error = float(np.max(np.abs(raw - x['raw'])))
                errors['max_raw_gate_abs_error'] = max(errors['max_raw_gate_abs_error'], error)
                assert np.allclose(raw, x['raw'], atol=3e-5, rtol=3e-5)
                expected_gate = np.where((raw > w['theta'][0]) & (raw > 0), raw, 0)
                assert np.allclose(expected_gate, x['gate'], atol=3e-5, rtol=3e-5)
                operator = (x['z'].astype(np.float64) @ w['V'] @ w['U']) * w['output_std']
                error = float(np.linalg.norm(operator - x['operator']) / max(np.linalg.norm(operator), 1e-12))
                errors['max_operator_relative_l2_error'] = max(errors['max_operator_relative_l2_error'], error)
                assert error < 1e-4
                fraction = 0.0 if run['condition'] == 'zero_control' else 1.0
                delta = -np.float32(fraction) * x['gate'][:, None] * x['operator']
                assert np.array_equal(delta, x['intended_delta'])
                assert np.array_equal(bf16_round(x['native_mlp'] + delta), x['edited_mlp'])
                assert np.array_equal(x['edited_mlp'] - x['native_mlp'], x['actual_delta'])
                vectors += 1
            if all(g == 0 for g in row['summary']['natural_gates']):
                assert forced['steps'] == native['steps'] and off['output_token_ids'] == native['output_token_ids']
            summary = {'feature': name, 'layer': feature['layer'], 'transform': feature['transform'],
                       **row['summary']}
            summary['first_output_change_pp'] = 100 * (summary['gate_off_p_first_output'] - summary['native_p_first_output'])
            summaries.append(summary)
            current.append(summary)
        worst = min(current, key=lambda x: x['answer_slot_change_pp'])
        best = max(current, key=lambda x: x['answer_slot_change_pp'])
        first_worst = min(current, key=lambda x: x['first_output_change_pp'])
        changes = [x for x in current if x['native_answer'] != x['gate_off_answer']]
        gate_summaries.append({'feature': name, 'label': feature['label'],
                               'active_gate_cases': sum(any(g > 0 for g in x['natural_gates']) for x in current),
                               'nonzero_edit_cases': sum(any(g > 0 for g in x['actual_edit_norms']) for x in current),
                               'greedy_answer_changes': len(changes), 'changed_answers': changes,
                               'largest_answer_slot_drop': worst, 'largest_answer_slot_increase': best,
                               'largest_first_output_drop': first_worst,
                               'child_cases': [x for x in current if x['group'] == 'child']})
    assert len(summaries) == 48 and vectors == 96
    validation = {'status': 'passed', 'gates_tested_separately': [x['feature'] for x in gate_summaries],
                  'gate_prompt_comparisons': 48, 'independent_cpu_vector_checks': vectors,
                  'native_baselines_match_previous_study_exactly': True, 'all_zero_controls_exact': True,
                  'all_bf16_rounding_exact': True, 'no_generated_token_edits': True,
                  'simultaneous_gate_ablations': False, 'hooks_remaining': 0, **errors}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    (OUT / 'gate-summary.json').write_text(json.dumps(gate_summaries, indent=2) + '\n')
    columns = ['feature', 'layer', 'transform', 'case_id', 'word', 'letter_case', 'group',
               'native_answer', 'gate_off_answer', 'native_p_first_output', 'gate_off_p_first_output',
               'first_output_change_pp', 'matched_prefix', 'native_p_answer_slot', 'gate_off_p_answer_slot',
               'answer_slot_change_pp', 'natural_gates', 'actual_edit_norms', 'zero_control_exact']
    with (OUT / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            record = {k: row[k] for k in columns if k in row}
            record['matched_prefix'] = row['answer_slot']['prefix']
            for k in ['natural_gates', 'actual_edit_norms']:
                record[k] = json.dumps(record[k])
            writer.writerow(record)
    lines = ['# Four CH gates, ablated separately — 28 October 2026', '',
             'The same five CH words plus child were tested in uppercase and lowercase: 12 prompts per gate, 48 gate/prompt comparisons. Every condition removed only one selected feature at its own layer. No joint ablations were performed.', '',
             'Gate-off subtracts the estimated natural gated contribution from the native MLP output at every input token overlapping the target word. Edits use FP32 accumulation and one BF16 cast. All other MLP positions at the intervention boundary, native attention, and generated tokens are unedited.', '',
             'Probabilities sum the plain and one-leading-space CH/ch token variants. First output means the immediate next token after the prompt. Answer-slot probabilities use the same native prefix in both conditions, such as chin:. These are conditional next-token probabilities, not the probability of every possible full correct response.', '',
             '| Gate | Active gate cases / 12 | Nonzero edits / 12 | Changed greedy responses / 12 | Largest answer-slot decrease |',
             '|---|---:|---:|---:|---|']
    for g in gate_summaries:
        x = g['largest_answer_slot_drop']
        lines.append(f"| {g['feature']} | {g['active_gate_cases']} | {g['nonzero_edit_cases']} | {g['greedy_answer_changes']} | {x['case_id']}: {x['answer_slot_change_pp']:+.6f} pp |")
    for g in gate_summaries:
        lines += ['', f"**{g['feature']} — {g['label']}**", '',
                  '| Case | Native response | Gate-off response | P(first), native → off (%) | Same native prefix | P(answer slot), native → off (%) | Slot delta (pp) | Natural gate(s) |',
                  '|---|---|---|---:|---|---:|---:|---|']
        for s in summaries:
            if s['feature'] != g['feature']:
                continue
            first = f"{100*s['native_p_first_output']:.6f} → {100*s['gate_off_p_first_output']:.6f}"
            slot = f"{100*s['native_p_answer_slot']:.6f} → {100*s['gate_off_p_answer_slot']:.6f}"
            prefix = s['answer_slot']['prefix'] or '(empty)'
            gates = ', '.join(f'{x:.6f}' for x in s['natural_gates'])
            lines.append(f"| {s['case_id']} | `{s['native_answer']}` | `{s['gate_off_answer']}` | {first} | `{prefix}` | {slot} | {s['answer_slot_change_pp']:+.6f} | {gates} |")
    lines += ['', 'The native baselines and all zero-strength controls exactly match the preceding L3–T2334 study. Independent CPU reconstruction checked every saved gate/operator vector and every BF16 edit (96 vector files). All hooks were removed after the study.', '',
              'Inactive gates produce exact no-ops. A no-op does not establish that an actively contributing feature would be irrelevant. The experiment subtracts estimated components from the native MLP; it does not replace the full layer or establish feature absorption.', '',
              'Prompt example:', '', '```text', json.loads((EXPORT / 'cases.json').read_text())[0]['user'], '```', '',
              'The lowercase version changes uppercase to lowercase and GA/WI/TA to ga/wi/ta.', '',
              f"[Full raw export](<{EXPORT}>) · [Run source](<{Path(__file__).with_name('driver.py')}>) · [CSV](<{OUT / 'summary.csv'}>) · [Validation](<{OUT / 'validation.json'}>)", '',
              'SPAR instance 58000985 is retained running for follow-up work.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(validation, indent=2))
    for g in gate_summaries:
        print('GATE', g['feature'], 'ACTIVE', g['active_gate_cases'], 'CHANGES', g['greedy_answer_changes'])
        for x in summaries:
            if x['feature'] == g['feature']:
                print(x['case_id'], repr(x['native_answer']), '=>', repr(x['gate_off_answer']),
                      'P_SLOT', 100*x['native_p_answer_slot'], '=>', 100*x['gate_off_p_answer_slot'],
                      'P_FIRST', 100*x['native_p_first_output'], '=>', 100*x['gate_off_p_first_output'],
                      'GATES', x['natural_gates'])


if __name__ == '__main__':
    main()
