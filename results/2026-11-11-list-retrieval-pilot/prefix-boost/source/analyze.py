"""Verify the fixed-scalar sweep and separate new retrieval from existing errors."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot/prefix-boost'
BASE = OUT.parent / 'without-starting-one'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clean(text):
    return text.strip().strip('`*_\"\'“”‘’').rstrip('.!?').strip().casefold()

def first_word_position(text, words):
    matches = sorted((m.start(), i) for i, word in enumerate(words, 1) for m in re.finditer(r'(?<![A-Za-z])' + re.escape(word) + r'(?![A-Za-z])', text, re.I))
    return matches[0][1] if matches else None

def main():
    r = json.loads((OUT / 'results.json').read_text())
    p = json.loads((HERE / 'protocol.json').read_text())
    base = json.loads((BASE / 'analyzed-cases.json').read_text())
    selected = [c for c in base['cases'] if c['ordinal'] in [5, 12]]
    assert r['status'] == 'complete' and r['hooks_remaining'] == 0
    assert r['protocol'] == p and r['protocol_sha256'] == sha(HERE / 'protocol.json')
    assert r['source_sha256'] == sha(HERE / 'run.py') == sha(OUT / 'run.py')
    assert sha(BASE / 'results.json') == p['source_results_sha256']
    assert [c['id'] for c in selected] == [c['id'] for c in p['cases']] == [c['id'] for c in r['cases']]
    assert len(r['cases']) == 20
    with np.load(HERE / 'weights/L29-T2375.npz', allow_pickle=False) as f:
        w = {k: f[k].astype(np.float32) for k in f.files}
    rows, errors = [], []
    for case, reference in zip(r['cases'], p['cases']):
        assert case['native']['native'] == reference['native']
        assert case['zero']['native'] == case['native']['native']
        assert case['zero']['candidates'] == case['native']['candidates']
        assert [v['target_gate'] for v in case['runs']] == p['strengths']
        for v in [case['native'], case['zero']] + case['runs']:
            assert v['scope'] == 'two_chat_prefix_input_newlines_only'
            file = OUT / v['arrays']['file']
            assert sha(file) == v['arrays']['sha256']
            with np.load(file, allow_pickle=False) as a:
                probes = dict(zip(a['first_probe_ids'].tolist(), a['first_probe_logits'].tolist()))
                lse = float(a['first_logsumexp'][0])
                for score in v['candidates'].values():
                    assert score['token_ids'] == [x['token_id'] for x in score['conditional']]
                    assert abs(score['conditional'][0]['log_probability'] - (probes[score['token_ids'][0]] - lse)) < 3e-5
                    for t in score['conditional']:
                        assert abs(t['log_probability'] - (t['logit'] - t['logsumexp'])) < 3e-5
                        if t['probability'] > 0:
                            assert math.isclose(math.exp(t['log_probability']), t['probability'], rel_tol=3e-5, abs_tol=1e-40)
                    logp = sum(t['log_probability'] for t in score['conditional'])
                    assert logp == score['word_log_probability'] and math.exp(logp) == score['word_probability']
                    assert 0 <= score['eos_probability_given_word'] <= 1
                    assert score['exact_word_then_eos_probability'] == score['word_probability'] * score['eos_probability_given_word']
                if v['alpha'] == 1:
                    assert a['selected_positions'].tolist() == [t['position'] for t in case['prefix_positions']]
                    np.testing.assert_array_equal(a['gate_after'], np.full(2, v['target_gate'], dtype=np.float32))
                    z = (a['residual_input'] - w['input_mean']) / w['input_std']
                    raw = z @ w['e'].reshape(-1) + w['bias'].reshape(-1)[0]
                    gate = np.where((raw > w['theta'].reshape(-1)[0]) & (raw > 0), raw, 0)
                    operator = (z @ w['V'] @ w['U']) * w['output_std']
                    np.testing.assert_allclose(a['gate_before'], gate, rtol=1e-4, atol=1e-4)
                    np.testing.assert_allclose(a['operator'], operator, rtol=5e-4, atol=5e-5)
                    np.testing.assert_allclose(a['intended_delta'], (a['gate_after'] - a['gate_before'])[:,None] * a['operator'], rtol=1e-6, atol=1e-7)
                    assert np.all(a['gate_after'] > a['gate_before'])
                    desired = (a['native_mlp'] + a['intended_delta']).astype(np.float32)
                    bits = desired.view(np.uint32)
                    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
                    np.testing.assert_array_equal(a['edited_mlp'], rounded)
                    np.testing.assert_array_equal(a['actual_delta'], a['edited_mlp'] - a['native_mlp'])
                    errors.extend(float(np.linalg.norm(actual - target) / np.linalg.norm(target)) for actual, target in zip(a['actual_delta'], a['intended_delta']))
            assert all(t['actual_delta_norm'] == t['intended_delta_norm'] == 0 for t in v['trace'] if not t['edited'])
            if v['condition'] == 'zero':
                assert all(t['actual_delta_norm'] == 0 for t in v['trace'])
            elif v['alpha'] == 1:
                edits = [t for t in v['trace'] if t['edited']]
                assert len(edits) == 2 and [t['position'] for t in edits] == [t['position'] for t in case['prefix_positions']]
                assert all(t['phase'] == 'input' and t['context'] == 'generation' and t['region'] == 'answer_prefix' and t['effective_gate'] == v['target_gate'] and t['actual_delta_norm'] > 0 for t in edits)
            # When a candidate is the exact greedy response, verify its word probability against that actual decoding path.
            for score in v['candidates'].values():
                n = len(score['token_ids'])
                if v['native']['output_token_ids'][:n] == score['token_ids']:
                    generated_p = math.prod(t['probability'] for t in v['native']['steps'][:n])
                    assert math.isclose(score['word_probability'], generated_p, rel_tol=3e-5, abs_tol=1e-40)
        for v in case['runs']:
            response = v['native']['response']
            is11 = clean(response) == case['eleventh_word'] and v['native']['stop_reason'] == 'eos'
            first = first_word_position(response, case['words'])
            row = dict(id=case['id'], list_id=case['list_id'], format=case['format'], ordinal=case['ordinal'], target_gate=v['target_gate'], expected=case['expected'], eleventh_word=case['eleventh_word'], native_response=case['native']['native']['response'], response=response, stop_reason=v['native']['stop_reason'], strict_eleventh=is11, first_listed_position=first, first_listed_eleventh=first==11, new_strict_eleventh=is11 and not case['native_already_eleventh'], native_already_eleventh=case['native_already_eleventh'], native_requested_correct=case['native_requested_correct'], native_eleventh_control_correct=case['native_eleventh_question_control']['strict_correct'], both_native_controls_correct=case['native_requested_correct'] and case['native_eleventh_question_control']['strict_correct'], retains_requested=clean(response)==case['expected'] and v['native']['stop_reason']=='eos', response_changed=response!=case['native']['native']['response'], truncated=v['native']['stop_reason']=='token_limit', first_probability=v['native']['steps'][0]['probability'], edited_input_tokens=2, edited_generated_tokens=0)
            for mode, scored in [('native',case['native']),('steered',v)]:
                for target in [5,11,12]:
                    s = scored['candidates'][f'word_{target}']
                    row[f'{mode}_p_word_{target}'] = s['word_probability']
                    row[f'{mode}_p_word_{target}_then_eos'] = s['exact_word_then_eos_probability']
            rows.append(row)
            row['new_first_listed_eleventh'] = row['first_listed_eleventh'] and not row['native_already_eleventh']
            mentioned = sorted({i for i, word in enumerate(case['words'], 1) if re.search(r'(?<![A-Za-z])' + re.escape(word) + r'(?![A-Za-z])', response, re.I)})
            row['mentioned_listed_positions'] = mentioned
            row['only_listed_word_is_eleventh'] = mentioned == [11] and v['native']['stop_reason'] == 'eos'
            row['new_only_listed_word_is_eleventh'] = row['only_listed_word_is_eleventh'] and not row['native_already_eleventh']
            row['generated_sequence_probability'] = math.prod(t['probability'] for t in v['native']['steps'])
    groups = []
    for strength in p['strengths']:
        subset = [x for x in rows if x['target_gate'] == strength]
        groups.append(dict(target_gate=strength, cases=len(subset), strict_eleventh=sum(x['strict_eleventh'] for x in subset), new_strict_eleventh=sum(x['new_strict_eleventh'] for x in subset), new_eleventh_from_native_correct=sum(x['new_strict_eleventh'] and x['native_requested_correct'] for x in subset), new_eleventh_both_controls_correct=sum(x['new_strict_eleventh'] and x['both_native_controls_correct'] for x in subset), first_listed_eleventh=sum(x['first_listed_eleventh'] for x in subset), literal_eleven=sum(clean(x['response']) in ['eleven','11','eleventh','11th'] for x in subset), truncated=sum(x['truncated'] for x in subset), responses_changed=sum(x['response_changed'] for x in subset), retains_requested=sum(x['retains_requested'] for x in subset), by_ordinal={str(n):dict(strict_eleventh=sum(x['strict_eleventh'] for x in subset if x['ordinal']==n),new_strict_eleventh=sum(x['new_strict_eleventh'] for x in subset if x['ordinal']==n),cases=10) for n in [5,12]}))
        groups[-1].update(new_first_listed_eleventh=sum(x['new_first_listed_eleventh'] for x in subset),new_only_listed_word_is_eleventh=sum(x['new_only_listed_word_is_eleventh'] for x in subset),new_answer_eleventh_from_native_correct=sum(x['new_first_listed_eleventh'] and x['native_requested_correct'] for x in subset),new_answer_eleventh_both_controls_correct=sum(x['new_first_listed_eleventh'] and x['both_native_controls_correct'] for x in subset))
    summary = dict(cases=20,variants=len(rows),strengths=p['strengths'],scope='Fixed post-JumpReLU scalar at both input-prefix newline tokens only. Local recipient operator retained. List-reading and generated positions unedited.',native_already_eleventh=sum(c['native_already_eleventh'] for c in r['cases']),eligible_new_target_cases=sum(not c['native_already_eleventh'] for c in r['cases']),native_requested_correct=sum(c['native_requested_correct'] for c in r['cases']),both_native_controls_correct=sum(c['native_requested_correct'] and c['native_eleventh_question_control']['strict_correct'] for c in r['cases']),native_and_zero_replays_exact=True,groups=groups,rows=rows,audit=dict(independent_cpu_gates_operators_verified=True,exact_bf16_rounding_verified=True,total_input_positions_edited=len(rows)*2,generated_positions_edited=0,max_actual_minus_intended_delta_relative_l2=max(errors)),source_sha256=r['source_sha256'],protocol_sha256=r['protocol_sha256'],results_sha256=sha(OUT/'results.json'),probability_definition=r['runtime']['probability_precision']+' Exact lowercase complete-word continuations; EOS, punctuation and formatting variants excluded from word-only scores. Word-then-EOS scores are separate.')
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'probabilities.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Fixed T2375 gates at both chat-prefix newline tokens','',summary['scope'],'',summary['probability_definition'],'',f"Native already returns the eleventh word in {summary['native_already_eleventh']}/20 cases; those do not count as newly steered. Native requested-word answers are correct in {summary['native_requested_correct']}/20 cases, and both requested/eleventh controls are correct in {summary['both_native_controls_correct']}/20.",'','| Gate | Strict eleventh /20 | Newly eleventh /18 | Newly eleventh from correct native /13 | Both controls correct /9 | Truncated /20 |','|---:|---:|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['target_gate']:g} | {g['strict_eleventh']} | {g['new_strict_eleventh']} | {g['new_eleventh_from_native_correct']} | {g['new_eleventh_both_controls_correct']} | {g['truncated']} |")
    lines += ['', '## Item 11 inside a sentence', '', 'Word-only counts above are a formatting-sensitive primary measure. The separate first-listed-word measure finds one newly induced item-11 answer at gate 40: list-3-bullets-12 responds “Thank you for the list. The twelfth word in the list is **ring**.” Ring is item 11; apple is item 12. The native twelfth response was bell (item 14), and the saved native eleventh control was lamp (item 10). This is a completed answer containing item 11, with extra text, on a case that failed both controls. It must be shown separately, not described as no new item-11 answers.', '']
    lines+=['','All 100 outputs are retained in results.json and probabilities.csv, including wrong words, number-only answers, formatting failures and truncation. Literal eleven/11 is not the eleventh listed word. All native and zero-strength replays are exact. Every steered run edits exactly two input tokens; independent CPU contribution and BF16 rounding audits pass.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in ['cases','variants','native_already_eleventh','native_requested_correct','both_native_controls_correct','groups','audit']},indent=2))

if __name__ == '__main__':
    main()
