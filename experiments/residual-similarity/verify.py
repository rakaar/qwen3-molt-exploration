"""Independent local audit of downloaded vectors, logits, and reported metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm else None


def probability(logits, token):
    logits = np.asarray(logits, dtype=np.float32)
    exp = np.exp(logits - logits.max(), dtype=np.float32)
    return float(exp[token] / exp.sum(dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    root = args.result_directory
    data = root / 'artifacts'
    r = json.loads((data / 'results.json').read_text())
    hashes = json.loads((data / 'export-checksums.json').read_text())
    for name, expected in hashes.items():
        with (data / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    assert r['status'] == 'complete'
    required = ['zero_strength_bitwise_equal', 'attention_forwards_unchanged',
                'downstream_attention_changed_both_methods', 'only_layer22_token9_directly_edited',
                'residual_addition_boundary_verified', 'reference_conditions_repeat_bitwise',
                'restoration_bitwise_equal', 'upstream_residuals_unchanged_every_trial']
    assert all(r['checks'][name] for name in required)
    vectors = np.load(data / 'residual-vectors.npz')
    logits = np.load(data / 'next-token-logits.npz')
    assert vectors['transcoder'].shape == (len(r['trials']), 36, 2560)
    assert np.array_equal(vectors['strengths'], r['protocol']['transcoder_strengths'])
    assert all(np.isfinite(vectors[key]).all() for key in vectors.files)
    assert all(np.isfinite(logits[key]).all() for key in logits.files)
    max_cos_error, max_prob_error = 0.0, 0.0
    fixed, clean = vectors['molt'], vectors['clean']
    for index, row in enumerate(r['trials']):
        tc = vectors['transcoder'][index]
        assert np.array_equal(tc[:22], clean[:22])
        for name, actual in [('cosine_residual', cosine(fixed[22], tc[22])),
                             ('cosine_residual_changes', cosine(fixed[22] - clean[22], tc[22] - clean[22])),
                             ('cosine_intended_edits', cosine(vectors['molt_injected_delta'],
                                      row['strength'] * vectors['tc_unit_delta']))]:
            expected = row[name]
            assert (actual is None) == (expected is None), (index, name)
            if actual is not None:
                error = abs(actual - expected)
                assert error < 2e-7, (index, name, error)
                max_cos_error = max(max_cos_error, error)
        for name, token in [('P_cold', 87072), ('P_large', 16767)]:
            error = abs(probability(logits['transcoder'][index], token) - row[name])
            assert error < 2e-6, (index, name, error)
            max_prob_error = max(max_prob_error, error)
        if row['strength'] == 0:
            assert np.array_equal(tc, clean)
            assert row['cosine_residual_changes'] is None
            assert row['cosine_intended_edits'] is None
    assert abs(probability(logits['molt'], 87072) - r['molt_fixed']['P_cold']) < 2e-6
    audit = {'status': 'passed', 'trial_count': len(r['trials']), 'all_export_checksums_verified': True,
             'metrics_recomputed_independently_from_saved_vectors_and_logits': True,
             'maximum_cosine_error': max_cos_error, 'maximum_probability_error': max_prob_error,
             'exact_zero_edit_and_upstream_controls_passed': True,
             'prior_capture_comparison': {k: v for k, v in r['checks'].items() if k.startswith(('receiver_', 'donor_'))}}
    (root / 'local-verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
