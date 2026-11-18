"""Audit question-only replay and extract the paired candidate probabilities."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot/all-token-off'
OUT = PILOT / 'question-only-off'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def get_first_logits(run):
    path = OUT / run['arrays']['file']
    assert sha(path) == run['arrays']['sha256']
    with np.load(path, allow_pickle=False) as arrays:
        logits = arrays['first_logits']
    first = run['native']['steps'][0]
    assert hashlib.sha256(logits.tobytes()).hexdigest() == first['logits_sha256']
    probabilities = np.exp(logits.astype(np.float64) - logits.max())
    probabilities /= probabilities.sum()
    for item in first['top5']:
        assert math.isclose(probabilities[item['token_id']], item['probability'], rel_tol=2e-6)
    for item in run['candidates'].values():
        assert math.prod(token['probability'] for token in item['conditional']) == item['word_probability']
        assert math.isclose(probabilities[item['token_ids'][0]], item['conditional'][0]['probability'], rel_tol=2e-6)
    return logits


def main():
    result = json.loads((OUT/'results.json').read_text())
    reference = json.loads((PILOT/'results.json').read_text())
    prototype = {r['id']:r for r in result['protocol']['cases']}
    assert result['status'] == 'complete' and result['hooks_remaining'] == 0
    assert sha(PILOT/'results.json') == result['protocol']['reference_all_token_results_sha256']
    for name, checksum in json.loads((OUT/'transfer-verification.json').read_text()).items():
        assert sha(OUT/name) == checksum
    assert result['protocol_sha256'] == sha(HERE/'protocol.json') and result['source_sha256'] == sha(HERE/'run.py')
    assert len(result['cases']) == len(prototype) == 7
    rows = []
    for case, prior in zip(result['cases'], reference['cases'], strict=True):
        assert case['id'] == prior['id']
        protocol = prototype[case['id']]
        native, zero, off = [case[mode] for mode in ('native', 'zero', 'off')]
        assert zero['native'] == native['native'] == prior['native']['native']
        assert zero['candidates'] == native['candidates'] == prior['native']['candidates']
        native_logits, zero_logits, off_logits = [get_first_logits(run) for run in (native, zero, off)]
        np.testing.assert_array_equal(native_logits, zero_logits)
        initial_input = [t for t in off['trace'] if t['context'] == 'generation' and t['phase'] == 'input']
        question = [t for t in initial_input if t['region'] == 'question']
        assert [t['position'] for t in question] == protocol['expected_question_positions']
        assert len(question) == 21
        for t in off['trace']:
            assert t['in_scope'] == (t['region'] == 'question' and t['phase'] == 'input')
            if not t['in_scope']:
                assert not t['removed'] and t['intended_delta_norm'] == t['actual_delta_norm'] == 0
                assert t['effective_gate'] == t['natural_gate']
        for t in question:
            assert t['effective_gate'] == 0
            assert t['removed'] == (t['natural_gate'] > 0)
        prefix = [t for t in initial_input if t['region'] == 'answer_prefix' and t['natural_gate'] > 0]
        list_tokens = [t for t in initial_input if t['region'] == 'list' and t['natural_gate'] > 0]
        row = {field:case[field] for field in ('id', 'list_id', 'format', 'eleventh_word', 'twelfth_word')}
        row.update(
            native_response=native['native']['response'],
            off_response=off['native']['response'],
            native_twelfth_control_correct=case['native_twelfth_question_control']['strict_correct'],
            question_text=''.join(t['token'] for t in question),
            question_positions=[t['position'] for t in question],
            question_active_positions=[t['position'] for t in question if t['natural_gate'] > 0],
            edited_question_positions=[t['position'] for t in question if t['removed']],
            prefix_active_positions=[t['position'] for t in prefix],
            prefix_active_gates=[t['natural_gate'] for t in prefix],
            list_active_positions=[t['position'] for t in list_tokens],
            all_decoding_logits_exact=(native['native'] == off['native']),
            first_logits_exact=np.array_equal(native_logits, off_logits),
        )
        if not row['edited_question_positions']:
            assert row['all_decoding_logits_exact'] and row['first_logits_exact']
            assert native['candidates'] == off['candidates']
            assert [(t['position'], t['natural_gate']) for t in initial_input] == [
                (t['position'],t['natural_gate']) for t in zero['trace'] if t['context'] == 'generation' and t['phase'] == 'input'
            ]
        for mode, run in [('native',native),('off',off)]:
            for target in ('eleventh', 'twelfth'):
                row[f'{mode}_p_{target}_word'] = run['candidates'][target]['word_probability']
        rows.append(row)
    summary = dict(
        cases=len(rows),scope=result['protocol']['question_only_scope'],
        responses_changed=sum(row['native_response'] != row['off_response'] for row in rows),
        retained_eleventh=sum(row['off_response'].strip()==row['eleventh_word'] for row in rows),
        switched_to_twelfth=sum(row['off_response'].strip()==row['twelfth_word'] for row in rows),
        question_tokens=sum(len(row['question_positions']) for row in rows),
        active_question_tokens=sum(len(row['question_active_positions']) for row in rows),
        edited_question_tokens=sum(len(row['edited_question_positions']) for row in rows),
        active_prefix_tokens=sum(len(row['prefix_active_positions']) for row in rows),
        active_list_tokens=sum(len(row['list_active_positions']) for row in rows),
        all_decoding_logits_exact=all(row['all_decoding_logits_exact'] for row in rows),
        rows=rows,run_sha256=sha(OUT/'results.json'),
    )
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'probabilities.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key:value for key,value in summary.items() if key!='rows'}))


if __name__=='__main__':
    main()
