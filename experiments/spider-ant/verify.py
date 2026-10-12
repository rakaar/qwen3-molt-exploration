"""Independently verify saved probabilities, edit norms, and exported hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_directory', type=Path)
    out = parser.parse_args().result_directory
    artifacts = out / 'artifacts'
    r = json.loads((artifacts / 'results.json').read_text())
    assert r['status'] == 'complete'
    for name, expected in json.loads((artifacts / 'export-checksums.json').read_text()).items():
        with (artifacts / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    assert hashlib.sha256((Path(__file__).parent / 'run.py').read_bytes()).hexdigest() == r['code_sha256']
    assert 'web_anthill' in r['checks']['valid_baseline_pairs']
    assert all(v is True for k, v in r['checks'].items() if k != 'valid_baseline_pairs')
    logits = np.load(artifacts / 'next-token-logits.npz')['logits']
    vectors = np.load(artifacts / 'intervention-vectors.npz')
    intended, actual = vectors['intended'], vectors['actual_mlp_changes']
    assert intended.shape == actual.shape == (len(r['trials']), 2560)
    assert np.isfinite(logits).all() and np.isfinite(intended).all() and np.isfinite(actual).all()
    maximum_probability_error = 0
    maximum_relative_norm_error = 0
    for row in [*r['baselines'].values(), *r['trials']]:
        values = logits[row['logits_index']].astype(np.float64)
        probabilities = np.exp(values - values.max())
        probabilities /= probabilities.sum()
        for number in ['6', '8', '4']:
            error = abs(float(probabilities[r['number_token_ids'][number]]) - row['P' + number])
            maximum_probability_error = max(maximum_probability_error, error)
            assert error < 2e-6
    for row in r['trials']:
        i = row['vector_index']
        for array, key in [(intended, 'intended_delta_norm'), (actual, 'actual_mlp_change_norm')]:
            norm = float(np.linalg.norm(array[i].astype(np.float64)))
            error = abs(norm - row[key]) / max(norm, 1)
            maximum_relative_norm_error = max(maximum_relative_norm_error, error)
            assert error < 2e-5
        if row['strength'] == 0 or row['intended_delta_norm'] == 0:
            baseline = r['baselines'][row['pair'] + '_receiver']
            assert np.array_equal(logits[row['logits_index']], logits[baseline['logits_index']])
    control_count = 0
    if (artifacts / 'random-control-results.json').exists():
        controls = json.loads((artifacts / 'random-control-results.json').read_text())
        assert hashlib.sha256((Path(__file__).parent / 'controls.py').read_bytes()).hexdigest() == controls['source_sha256']
        control_logits = np.load(artifacts / 'control-logits.npz')['logits']
        assert np.isfinite(control_logits).all()
        control_count = len(controls['trials'])
        for row in controls['trials']:
            values = control_logits[row['logits_index']].astype(np.float64)
            probabilities = np.exp(values-values.max())
            probabilities /= probabilities.sum()
            for number in ['6', '8', '4']:
                error = abs(float(probabilities[r['number_token_ids'][number]])-row['P'+number])
                maximum_probability_error = max(maximum_probability_error, error)
                assert error < 2e-6
    fields = [k for k in r['trials'][0] if k != 'top_tokens']
    with (out / 'strength-sweep.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(r['trials'])
    audit = {'status': 'passed', 'trial_count': len(r['trials']),
             'additional_control_count': control_count,
             'all_export_hashes_verified': True, 'code_hash_verified': True,
             'probabilities_recomputed_from_full_vocabulary_logits': True,
             'maximum_probability_error': maximum_probability_error,
             'maximum_relative_edit_norm_error': maximum_relative_norm_error,
             'zero_and_inactive_edits_bitwise_equal_clean': True,
             'valid_baseline_pairs': r['checks']['valid_baseline_pairs']}
    (out / 'local-verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
