"""Independently verify selection, full-word scores and actual contribution edits."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / 'results/2026-11-11-list-retrieval-pilot/all-token-off'
BASE = OUT.parent / 'without-starting-one'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clean(text):
    return text.strip().strip('`*_\"\'“”‘’').rstrip('.!?').strip().casefold()

def main():
    r = json.loads((OUT / 'results.json').read_text())
    p = json.loads((HERE / 'protocol.json').read_text())
    baseline = json.loads((BASE / 'analyzed-cases.json').read_text())
    selected = [c for c in baseline['cases'] if c['ordinal'] == 11 and c['assessment']['strict_correct']]
    assert r['status'] == 'complete' and r['hooks_remaining'] == 0
    assert r['protocol'] == p and r['protocol_sha256'] == sha(HERE / 'protocol.json')
    assert r['source_sha256'] == sha(HERE / 'run.py') == sha(OUT / 'run.py')
    assert p['selection_source_results_sha256'] == sha(BASE / 'results.json')
    assert [c['id'] for c in selected] == [c['id'] for c in p['cases']] == [c['id'] for c in r['cases']]
    assert len(selected) == 7
    with np.load(HERE / 'weights/L29-T2375.npz', allow_pickle=False) as f:
        w = {k: f[k].astype(np.float32) for k in f.files}
    audit_errors = []
    rows = []
    for record, reference in [(r['calibration'], p['calibration'])] + list(zip(r['cases'], p['cases'])):
        assert record['native']['native'] == reference['native']
        assert record['zero']['native'] == record['native']['native']
        assert record['zero']['candidates'] == record['native']['candidates']
        for name in ['native', 'zero', 'off']:
            run = record[name]
            array_path = OUT / run['arrays']['file']
            assert sha(array_path) == run['arrays']['sha256']
            with np.load(array_path, allow_pickle=False) as a:
                logits = a['first_logits'].astype(np.float64)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                for score in run['candidates'].values():
                    assert score['token_ids'] == [x['token_id'] for x in score['conditional']]
                    assert math.isclose(probs[score['token_ids'][0]], score['conditional'][0]['probability'], rel_tol=2e-5, abs_tol=1e-40)
                    for t in score['conditional']:
                        assert math.isclose(math.exp(t['logit'] - t['logsumexp']), t['probability'], rel_tol=3e-5, abs_tol=1e-40)
                    product = math.prod(t['probability'] for t in score['conditional'])
                    assert product == score['word_probability']
                    assert score['exact_word_then_eos_probability'] == product * score['eos_probability_given_word']
                if name == 'off':
                    # Reconstruct the learned contribution independently on CPU.
                    z = (a['residual_input'] - w['input_mean']) / w['input_std']
                    raw = z @ w['e'].reshape(-1) + w['bias'].reshape(-1)[0]
                    gate = np.where((raw > w['theta'].reshape(-1)[0]) & (raw > 0), raw, 0)
                    operator = (z @ w['V'] @ w['U']) * w['output_std']
                    np.testing.assert_allclose(a['gate_before'], gate, rtol=1e-4, atol=1e-4)
                    np.testing.assert_allclose(a['operator'], operator, rtol=5e-4, atol=5e-5)
                    np.testing.assert_allclose(a['intended_delta'], -a['gate_before'][:, None] * a['operator'], rtol=1e-6, atol=1e-7)
                    intended_output = (a['native_mlp'] + a['intended_delta']).astype(np.float32)
                    bits = intended_output.view(np.uint32)
                    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
                    np.testing.assert_array_equal(a['edited_mlp'], rounded)
                    np.testing.assert_array_equal(a['actual_delta'], a['edited_mlp'] - a['native_mlp'])
                    for wanted, actual in zip(a['intended_delta'], a['actual_delta']):
                        audit_errors.append(float(np.linalg.norm(actual - wanted) / np.linalg.norm(wanted)))
            if name == 'zero':
                assert all(t['actual_delta_norm'] == t['intended_delta_norm'] == 0 for t in run['trace'])
            if name == 'off':
                eligible = [t for t in run['trace'] if run['scope'] == 'all' or t['final_input']]
                assert all(t['effective_gate'] == 0 for t in eligible)
                assert all(t['removed'] == (t['natural_gate'] > 0) for t in eligible)
        if reference['id'] == 'calibration-month':
            continue
        native, off = record['native'], record['off']
        trajectory = [t for t in off['trace'] if t['context'] == 'generation']
        input_edits = [t for t in trajectory if t['phase'] == 'input' and t['removed']]
        assert [t['position'] for t in input_edits] == reference['native_input_active_positions']
        assert native['scope'] == off['scope'] == 'all'
        row = dict(id=record['id'], list_id=record['list_id'], format=record['format'], eleventh_word=record['eleventh_word'], twelfth_word=record['twelfth_word'], native_response=native['native']['response'], off_response=off['native']['response'], off_stop_reason=off['native']['stop_reason'], response_changed=native['native']['response'] != off['native']['response'], switches_to_twelfth=clean(off['native']['response']) == record['twelfth_word'] and off['native']['stop_reason'] == 'eos', retains_eleventh=clean(off['native']['response']) == record['eleventh_word'] and off['native']['stop_reason'] == 'eos', native_twelfth_control_correct=record['native_twelfth_question_control']['strict_correct'], native_twelfth_control_response=record['native_twelfth_question_control']['response'], edited_input_tokens=len(input_edits), edited_generated_tokens=sum(t['removed'] for t in trajectory if t['phase'] == 'response' and not t['trace_only_terminal']), edited_terminal_tokens=sum(t['removed'] for t in trajectory if t['trace_only_terminal']))
        for condition, run in [('native', native), ('off', off)]:
            for target, score in run['candidates'].items():
                row[f'{condition}_p_{target}_word'] = score['word_probability']
                row[f'{condition}_p_{target}_word_then_eos'] = score['exact_word_then_eos_probability']
            row[f'{condition}_generated_sequence_probability'] = math.prod(t['probability'] for t in run['native']['steps'])
        rows.append(row)
    summary = dict(cases=len(rows), scope='All processed input tokens and generated tokens, including numbered-list reading and chat answer prefix. Dynamic gate values on each trajectory. Native MLP retained; subtract only phi(z) * ((z @ V @ U) * output_std).', probability_definition='P11/P12 are products over all tokens of the exact lowercase word continuation with no added whitespace. Following punctuation/EOS and formatting variants are excluded. Exact-word-then-any-allowed-EOS probabilities are separately retained.', switched_to_twelfth=sum(c['switches_to_twelfth'] for c in rows), retained_eleventh=sum(c['retains_eleventh'] for c in rows), responses_changed=sum(c['response_changed'] for c in rows), native_baselines_and_zero_controls_exact=True, native_twelfth_controls_correct=sum(c['native_twelfth_control_correct'] for c in rows), edited_input_tokens=sum(c['edited_input_tokens'] for c in rows), edited_generated_tokens=sum(c['edited_generated_tokens'] for c in rows), edited_terminal_tokens=sum(c['edited_terminal_tokens'] for c in rows), audit=dict(independent_cpu_operator_and_gate_verified=True, actual_mlp_output_matches_exact_bf16_rounding=True, maximum_actual_minus_intended_delta_relative_l2=max(audit_errors)), calibration=dict(scope='Final input token only; separate raw month prompt, excluded from seven cases.', native_response=r['calibration']['native']['native']['response'], off_response=r['calibration']['off']['native']['response'], scores={mode:{target:score['word_probability'] for target,score in r['calibration'][mode]['candidates'].items()} for mode in ['native','off']}), rows=rows, protocol_sha256=r['protocol_sha256'], results_sha256=sha(OUT / 'results.json'), source_sha256=r['source_sha256'])
    (OUT / 'analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    with (OUT / 'probabilities.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# All-token T2375 removal on correctly answered eleventh-word list requests', '', f"Switched to the twelfth word: {summary['switched_to_twelfth']}/7. Retained the eleventh word: {summary['retained_eleventh']}/7. Exact response changes: {summary['responses_changed']}/7.", '', summary['scope'], '', summary['probability_definition'], '', '| Case | 11th / 12th target | Native response | Gate-off response | P11 native | P11 off | P12 native | P12 off |', '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['id']} | {row['eleventh_word']} / {row['twelfth_word']} | {row['native_response']} | {row['off_response']} | {row['native_p_eleventh_word']:.9g} | {row['off_p_eleventh_word']:.9g} | {row['native_p_twelfth_word']:.9g} | {row['off_p_twelfth_word']:.9g} |")
    lines += ['', 'The probabilities above are fractions, not percentages. All seven cases were selected by native eleventh-word correctness before seeing edits. The corresponding native twelfth-word question succeeds in five numbered cases; it fails in both retained bullet cases. All unhooked and zero-strength decoding outputs/logits exactly match the saved native baselines.', '', f"Removed {summary['edited_input_tokens']} positive input contributions. The policy remained active during generation; {summary['edited_generated_tokens']} generated-token and {summary['edited_terminal_tokens']} terminal-token positive contributions were encountered. CPU reconstruction and exact BF16 output-rounding checks passed.", '', 'The separate month calibration changes a November-leading completion into a December-leading completion with final-input-only removal. This is an implementation calibration, not an additional list case.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in ['cases','switched_to_twelfth','retained_eleventh','responses_changed','edited_input_tokens','edited_generated_tokens','audit']}))

if __name__ == '__main__':
    main()
