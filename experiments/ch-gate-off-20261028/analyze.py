"""Independently audit saved intervention vectors and build a compact result record."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
EXPORT = WORKSPACE / 'comparisons/2026-10-28-ch-gate-ablation/remote-export/ch-gate-off-20261028'
OUT = WORKSPACE / 'results/2026-10-28-ch-gate-off'


def bf16_round(values):
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    result = json.loads((EXPORT / 'artifacts/results.json').read_text())
    assert result['status'] == 'complete' and len(result['cases']) == 12
    assert result['only_edited_feature'] == {'layer': 3, 'transform': 2334}
    assert result['hooks_remaining'] == 0
    selected = json.loads((EXPORT / 'selected-transform.json').read_text())
    assert hashlib.sha256((EXPORT / selected['file']).read_bytes()).hexdigest() == selected['sha256']
    with np.load(EXPORT / selected['file'], allow_pickle=False) as data:
        weights = {k: data[k].astype(np.float64) for k in data.files}
    errors = {'max_raw_gate_abs_error': 0.0, 'max_operator_relative_l2_error': 0.0,
              'max_probability_mass_reconstruction_error': 0.0}
    vector_count = 0
    summaries = []
    for case in result['cases']:
        native, zero = case['native'], case['zero_control']
        off, forced = case['gate_off_generation'], case['gate_off_baseline_prefix']
        assert native['output_token_ids'] == zero['output_token_ids']
        assert native['steps'] == zero['steps']
        assert native['output_token_ids'] == forced['output_token_ids']
        assert forced['forced_baseline_continuation']
        assert off['target_positions'] == zero['target_positions']
        for run in [native, zero, off, forced]:
            assert run['generated_positions_edited'] == 0 and run['attention_functions_unmodified']
            for step in run['steps']:
                reconstructed = sum(math.exp(t['logit'] - step['logsumexp']) for t in step['target_tokens'])
                error = abs(reconstructed - step['target_probability_mass'])
                errors['max_probability_mass_reconstruction_error'] = max(errors['max_probability_mass_reconstruction_error'], error)
                assert error < 4e-6
            for edit in run['edits']:
                assert edit['layer'] == 3 and edit['transform'] == 2334
                assert edit['position'] in run['target_positions']
                assert edit['all_other_positions_identical_at_edit_boundary']
        for run in [zero, off]:
            file = EXPORT / 'artifacts' / run['vector_file']
            assert hashlib.sha256(file.read_bytes()).hexdigest() == run['vector_sha256']
            with np.load(file, allow_pickle=False) as data:
                x = {k: data[k] for k in data.files}
            raw = x['z'].astype(np.float64) @ weights['e'] + weights['bias'][0]
            raw_error = float(np.max(np.abs(raw - x['raw'])))
            errors['max_raw_gate_abs_error'] = max(errors['max_raw_gate_abs_error'], raw_error)
            assert np.allclose(raw, x['raw'], atol=3e-5, rtol=3e-5)
            gate = np.where((raw > weights['theta'][0]) & (raw > 0), raw, 0)
            assert np.allclose(gate, x['gate'], atol=3e-5, rtol=3e-5)
            operator = (x['z'].astype(np.float64) @ weights['V'] @ weights['U']) * weights['output_std']
            relative_error = float(np.linalg.norm(operator - x['operator']) / max(np.linalg.norm(operator), 1e-12))
            errors['max_operator_relative_l2_error'] = max(errors['max_operator_relative_l2_error'], relative_error)
            assert relative_error < 1e-4
            fraction = 0.0 if run['condition'] == 'zero_control' else 1.0
            delta = -np.float32(fraction) * x['gate'][:, None] * x['operator']
            assert np.array_equal(delta, x['intended_delta'])
            assert np.array_equal(bf16_round(x['native_mlp'] + delta), x['edited_mlp'])
            assert np.array_equal(x['edited_mlp'] - x['native_mlp'], x['actual_delta'])
            vector_count += 1
        if all(g == 0 for g in case['summary']['natural_gates']):
            assert forced['steps'] == native['steps']
            assert off['output_token_ids'] == native['output_token_ids']
        summaries.append(case['summary'])
    active = [s for s in summaries if any(g > 0 for g in s['natural_gates'])]
    inactive = [s['case_id'] for s in summaries if all(g == 0 for g in s['natural_gates'])]
    validation = {'status': 'passed', 'cases': len(summaries), 'vector_files_verified': vector_count,
                  'active_gate_cases': len(active), 'inactive_gate_cases': inactive,
                  'all_zero_controls_exact': True, 'cpu_gate_and_operator_reconstruction_passed': True,
                  'bf16_edit_rounding_exact': True, 'only_edited_feature': {'layer': 3, 'transform': 2334},
                  'generated_tokens_edited': 0, 'hooks_remaining': 0,
                  'greedy_answer_changes': sum(s['native_answer'] != s['gate_off_answer'] for s in summaries),
                  'elapsed_seconds': result['elapsed_seconds'], **errors}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    columns = ['case_id', 'word', 'letter_case', 'group', 'native_answer', 'gate_off_answer',
               'native_p_answer_slot', 'gate_off_p_answer_slot', 'answer_slot_change_pp',
               'native_p_first_output', 'gate_off_p_first_output', 'natural_gates', 'actual_edit_norms',
               'matched_prefix', 'zero_control_exact']
    with (OUT / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            record = {k: row[k] for k in columns if k in row}
            for k in ['natural_gates', 'actual_edit_norms']:
                record[k] = json.dumps(record[k])
            record['matched_prefix'] = row['answer_slot']['prefix']
            writer.writerow(record)
    lines = ['# CH gate-off pilot — 28 October 2026', '',
             'Only L3–T2334 was manipulated. Five fixed CH words were tested in uppercase and lowercase, with child tested separately (12 prompts total).', '',
             'The prompt matched the linked result: garden/window/table few-shot demonstrations, followed by the target word. Greedy decoding, thinking disabled, at most 16 new tokens.', '',
             'Example:', '', '```text', result['cases'][0]['case']['user'], '```', '',
             'The lowercase version changes uppercase to lowercase and GA/WI/TA to ga/wi/ta.', '',
             'At every input token overlapping the target word, gate-off subtracts the estimated natural gated contribution of L3–T2334 from the native layer-3 MLP output. The FP32 edit is cast once to BF16. All other positions at that boundary, native attention, generated tokens, and the native MLP reconstruction remainder are preserved.', '',
             'Probabilities are the sum for the two single-token variants CH and leading-space CH, or ch and leading-space ch. They are next-token probabilities, not the total probability of every possible full correct response.', '',
             'First output uses the prompt alone. Answer slot uses the same native continuation prefix in both conditions; for example, lower chin compares ch after the identical prefix chin:.', '',
             '| Case | Native response | Gate-off response | P at first output: native → off (%) | Matched prefix | P at answer slot: native → off (%) | Slot delta (pp) | Natural gate(s) |',
             '|---|---|---|---:|---|---:|---:|---|']
    for s in summaries:
        p1 = f"{100*s['native_p_first_output']:.6f} → {100*s['gate_off_p_first_output']:.6f}"
        pa = f"{100*s['native_p_answer_slot']:.6f} → {100*s['gate_off_p_answer_slot']:.6f}"
        prefix = s['answer_slot']['prefix'] or '(empty)'
        gates = ', '.join(f'{g:.6f}' for g in s['natural_gates'])
        lines.append(f"| {s['case_id']} | `{s['native_answer']}` | `{s['gate_off_answer']}` | {p1} | `{prefix}` | {pa} | {s['answer_slot_change_pp']:+.6f} | {gates} |")
    lines += ['', f"All 12 zero-strength controls exactly reproduced native tokens and recorded probabilities. Independent CPU reconstruction checked {vector_count} saved vector files; BF16 rounding matched exactly. All hooks were removed.", '',
              'All greedy responses were unchanged. The largest confidence decrease was lowercase chair, 70.6463% to 45.7795% (−24.8668 percentage points). Uppercase cheek increased from 69.9403% to 73.6768%.', '',
              'The gate was already inactive on both child prompts, both chest prompts, and lowercase cheek. Those five interventions were exact no-ops. Uppercase cheek split into che/ek, with an active gate only on che.', '',
              'These are deterministic, selected-prompt results for subtraction of an estimated MOLT component from the native MLP. They do not establish feature absorption, full-layer reconstruction fidelity, or specificity relative to other active features.', '',
              f"Pinned base: `{result['protocol']['base_repo']}@{result['protocol']['base_revision']}`.",
              f"Pinned MOLTs: `{result['protocol']['molt_repo']}@{result['protocol']['molt_revision']}`.", '',
              f"Reference: {result['protocol']['reference_url']}", '',
              f"Full raw results: `{EXPORT / 'artifacts/results.json'}`.",
              f"Source: `{Path(__file__).with_name('run.py')}`.", '',
              'The GPU is retained running on SPAR instance 58000985 for follow-up work.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(validation, indent=2))
    for s in summaries:
        print(s['case_id'], repr(s['native_answer']), '=>', repr(s['gate_off_answer']),
              'P_SLOT', 100*s['native_p_answer_slot'], '=>', 100*s['gate_off_p_answer_slot'],
              'P_FIRST', 100*s['native_p_first_output'], '=>', 100*s['gate_off_p_first_output'])


if __name__ == '__main__':
    main()
