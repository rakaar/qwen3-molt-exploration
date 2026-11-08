"""Audit captured vectors and compare every prespecified token pair."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else None


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source, destination):
    assert not (destination / 'summary.json').exists(), 'Preserve prior analysis'
    destination.mkdir(parents=True, exist_ok=True)
    result_path = source / 'results.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'complete' and result['hooks_remaining'] == 0
    assert len(result['cases']) == len(result['protocol']['cases'])
    review_path = HERE / 'native-answer-review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    with np.load(HERE / 'weights/L29-T2375.npz', allow_pickle=False) as archive:
        w = {k: archive[k].astype(np.float64) for k in archive.files}
    vectors, traces = {}, []
    for case in result['cases']:
        assert case['observer_all_logits_and_output_exact']
        path = source / case['vectors_file']
        assert sha(path) == case['vectors_sha256']
        with np.load(path, allow_pickle=False) as a:
            v = {k: a[k].copy() for k in a.files}
        assert np.array_equal(v['input_ids'], case['input_ids'])
        z = (v['residual'].astype(np.float64) - w['input_mean']) / w['input_std']
        raw = (z @ w['e'].reshape(-1)) + w['bias'].item()
        active = (raw > w['theta'].item()) & (raw > 0)
        assert np.array_equal(active, v['gate'] > 0)
        assert np.allclose(raw, v['raw'], atol=2e-5, rtol=2e-5)
        operator = (z @ w['V'] @ w['U']) * w['output_std']
        assert np.allclose(operator, v['operator'], atol=2e-4, rtol=2e-4)
        assert np.allclose(np.where(active, raw, 0)[:, None] * operator,
                           v['contribution'], atol=5e-4, rtol=5e-4)
        vectors[case['id']] = v
        traces.extend({'case': case['id'], 'variant': case['variant'],
                       'domain': case['domain'], 'ordinal': case['ordinal'],
                       'final_input': t['position'] == len(case['tokens']) - 1, **t}
                      for t in case['trace'])

    def pair(a, i, b, j):
        row = {
            'case_a': a['id'], 'position_a': i, 'token_a': a['tokens'][i],
            'token_id_a': a['input_ids'][i], 'gate_a': float(vectors[a['id']]['gate'][i]),
            'case_b': b['id'], 'position_b': j, 'token_b': b['tokens'][j],
            'token_id_b': b['input_ids'][j], 'gate_b': float(vectors[b['id']]['gate'][j]),
            'same_token_id': a['input_ids'][i] == b['input_ids'][j],
            'native_answer_correct_a': review.get(a['id'], {}).get('correct'),
            'native_answer_correct_b': review.get(b['id'], {}).get('correct'),
        }
        for name in ['contribution', 'operator', 'residual']:
            av, bv = vectors[a['id']][name][i], vectors[b['id']][name][j]
            row[name + '_cosine'] = cosine(av, bv)
            row[name + '_norm_a'] = float(np.linalg.norm(av.astype(np.float64)))
            row[name + '_norm_b'] = float(np.linalg.norm(bv.astype(np.float64)))
        return row

    final_pairs, active_pairs, ordinal_controls, matrices = [], [], [], []
    cases = result['cases']
    for a, b in itertools.combinations(cases, 2):
        if a['ordinal'] == b['ordinal'] and a['domain'] != b['domain']:
            final_pairs.append(pair(a, len(a['tokens']) - 1, b, len(b['tokens']) - 1))
            if a['ordinal'] == 11:
                for ta, tb in itertools.product(a['active_input_tokens'], b['active_input_tokens']):
                    active_pairs.append(pair(a, ta['position'], b, tb['position']))
        if a['variant'] == b['variant'] and a['domain'] == b['domain'] and a['ordinal'] != b['ordinal']:
            ordinal_controls.append(pair(a, len(a['tokens']) - 1, b, len(b['tokens']) - 1))
    for variant in dict.fromkeys(c['variant'] for c in cases):
        for ordinal in [10, 11, 12]:
            triplet = [next(c for c in cases if c['variant'] == variant and c['ordinal'] == ordinal
                            and c['domain'] == domain) for domain in ['month', 'alphabet', 'roman']]
            matrices.append({
                'variant': variant, 'ordinal': ordinal, 'domains': ['month', 'alphabet', 'roman'],
                'tokens': [c['final_input'] for c in triplet],
                'cosines': {kind: [[cosine(vectors[a['id']][kind][-1], vectors[b['id']][kind][-1])
                                    for b in triplet] for a in triplet]
                            for kind in ['contribution', 'operator', 'residual']},
            })
    pair_summary = {}
    for label, same in [('same_token', True), ('different_token', False)]:
        values = [p['contribution_cosine'] for p in active_pairs if p['same_token_id'] == same]
        pair_summary[label] = {'pairs': len(values), 'minimum': min(values) if values else None,
                               'median': float(np.median(values)) if values else None,
                               'maximum': max(values) if values else None}
    correct_different = [p['contribution_cosine'] for p in active_pairs
                         if not p['same_token_id'] and p['native_answer_correct_a'] is True
                         and p['native_answer_correct_b'] is True]
    pair_summary['different_token_correct_native_answers'] = {
        'pairs': len(correct_different), 'minimum': min(correct_different) if correct_different else None,
        'median': float(np.median(correct_different)) if correct_different else None,
        'maximum': max(correct_different) if correct_different else None}
    write_csv(destination / 'all-input-token-traces.csv', traces)
    write_csv(destination / 'all-active-eleven-token-pairs.csv', active_pairs)
    write_csv(destination / 'final-input-pairs.csv', final_pairs)
    write_csv(destination / 'matched-ordinal-controls.csv', ordinal_controls)
    summary = {
        'status': 'complete', 'source_sha256': sha(result_path), 'analysis_sha256': sha(Path(__file__)),
        'cases': [{**{k: c[k] for k in ['id', 'variant', 'domain', 'ordinal', 'prompt', 'expected_answer', 'native', 'final_input',
                                    'active_input_tokens', 'original_anchor_exact']},
                   'native_answer_review': review.get(c['id'])} for c in cases],
        'final_token_matrices': matrices, 'active_eleven_cross_domain_pairs': pair_summary,
        'native_answer_review_sha256': sha(review_path) if review_path.exists() else None,
        'scope': result['protocol']['design'], 'independent_CPU_vector_audit': 'pass',
    }
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'cases': len(cases), 'input_tokens': len(traces),
                      'all_active_pair_summary': pair_summary, 'CPU_audit': 'pass'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Verified remote artifacts directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
