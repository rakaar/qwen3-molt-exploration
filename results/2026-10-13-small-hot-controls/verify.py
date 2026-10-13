"""Independent CPU verification of full-vocabulary probabilities and direct writes."""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
R = json.loads((OUT / 'results.json').read_text())
assert R['status'] == 'complete'
assert all(R['checks'].values())
assert all(row['bitwise_equal'] for row in R['batch_clean_checks'])
for name, expected in R['source_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
for name, expected in json.loads((OUT / 'export-checksums.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
assert hashlib.sha256((ROOT / 'weights/molt-22-selected.safetensors').read_bytes()).hexdigest() == R['selected_asset_sha256']
selection = R['selection']
assert selection['saved_before_interventions']
assert selection == json.loads((OUT / 'selection.json').read_text())
pool = [i for i in range(2480) if i not in (941, 2339)]
assert random.Random(R['protocol']['random_seed']).sample(pool, 100) == selection['uniform_ids']
assert len(set(selection['uniform_ids'])) == 100
assert all(a != b for a, b in selection['uniform_pairs'])
assert set(a for a, b in selection['uniform_pairs']) == set(b for a, b in selection['uniform_pairs']) == set(selection['uniform_ids'])

bits = np.load(OUT / 'next-token-logits.npz')['logits_bf16_bits']
logits = (bits.astype(np.uint32) << 16).view(np.float32)
assert logits.shape[1] > 87072 and np.isfinite(logits).all()
q = logits.astype(np.float64)
q -= q.max(axis=1, keepdims=True)
probability = np.exp(q)
probability /= probability.sum(axis=1, keepdims=True)
direct = np.load(OUT / 'direct-write.npz')
contribution = np.load(OUT / 'contributions.npz')
ids = {int(f): i for i, f in enumerate(contribution['ids'])}
receiver, donor = contribution['receiver'], contribution['donor']
gates = {'receiver': contribution['receiver_gates'], 'donor': contribution['donor_gates']}
assert len(gates['receiver']) == len(gates['donor']) == 2480
for f, i in ids.items():
    for source, vectors in [('receiver', receiver), ('donor', donor)]:
        if gates[source][f] == 0:
            assert not np.any(vectors[i])


def bf16_round(array):
    u = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


by_id = {row['id']: row for row in R['rows']}
assert len(by_id) == len(R['rows'])
max_error, max_norm_error, max_vector_error = 0, 0, 0
for row in R['rows']:
    index, vi = row['logit_row'], row['vector_row']
    assert hashlib.sha256(bits[index].tobytes()).hexdigest() == row['logit_sha256']
    for name, token in [('P_cold', 87072), ('P_large', 16767)]:
        error = abs(probability[index, token] - row[name])
        max_error = max(error, max_error)
        assert error < 2e-6, (row['id'], name, error)
    assert int(np.argmax(logits[index])) == row['top_tokens'][0]['id']
    pre, post, delta = direct['before'][vi], direct['after'][vi], direct['intended_delta'][vi]
    norm = np.linalg.norm((post - pre).astype(np.float64))
    error = abs(norm - row['actual_delta_norm']) / max(1, norm)
    max_norm_error = max(max_norm_error, error)
    assert error < 3e-6
    assert (norm > 0) == row['nonzero_edit']
    if row['mode'] == 'add':
        reconstructed = np.zeros(2560, dtype=np.float32)
        for source, f, coefficient in row['terms']:
            reconstructed += np.float32(coefficient) * (receiver if source == 'receiver' else donor)[ids[f]]
        error = np.linalg.norm((reconstructed - delta).astype(np.float64)) / max(1, np.linalg.norm(delta.astype(np.float64)))
        max_vector_error = max(max_vector_error, error)
        assert error < 3e-6
        assert np.array_equal(bf16_round(pre + bf16_round(delta)), post), row['id']
    if row['group'] != 'baseline_donor' and not row['nonzero_edit']:
        assert np.array_equal(logits[index], logits[by_id['clean_receiver']['logit_row']])
    if row['group'] not in ('baseline', 'baseline_donor'):
        assert np.array_equal(pre, direct['before'][by_id['clean_receiver']['vector_row']])
    for term in row.get('source_gates', []):
        assert gates[term['source']][term['feature']] == term['gate']
    if row['group'].startswith('active_'):
        assert all(term['gate'] > 0 for term in row['source_gates'])
whole = by_id['whole_mlp_donor_replace']
assert np.array_equal(direct['after'][whole['vector_row']], direct['before'][by_id['clean_donor']['vector_row']])
assert np.array_equal(logits[by_id['selected_pair']['logit_row']], logits[by_id['selected_pair_repeat']['logit_row']])
for name in ('self_mlp_replace', 'zero_delta', 'restored_receiver'):
    assert np.array_equal(logits[by_id[name]['logit_row']], logits[by_id['clean_receiver']['logit_row']])
groups = {}
selected = by_id['selected_pair']['P_cold']
for row in R['rows']:
    groups.setdefault(row['group'], []).append(row)
stats = {group: {'count': len(rows), 'nonzero_edits': sum(r['nonzero_edit'] for r in rows),
                 'maximum_P_cold': max(r['P_cold'] for r in rows),
                 'median_P_cold': float(np.median([r['P_cold'] for r in rows])),
                 'at_least_99_percent': sum(r['P_cold'] >= .99 for r in rows),
                 'at_least_selected_P_cold': sum(r['P_cold'] >= selected for r in rows)}
         for group, rows in groups.items()}
result = {'status': 'passed', 'conditions': len(R['rows']), 'unique_full_vocabulary_logit_rows': len(logits),
          'vocabulary_size': logits.shape[1], 'maximum_probability_absolute_error': max_error,
          'maximum_actual_norm_relative_error': max_norm_error,
          'maximum_contribution_reconstruction_relative_error': max_vector_error,
          'random_selection_matches_prespecified_seed': True, 'exact_full_mlp_replacement_confirmed': True,
          'BF16_additive_writes_recomputed': True, 'inactive_edits_recover_clean_logits': True,
          'source_and_artifact_hashes_match': True, 'groups': stats,
          'limits': 'Random samples are exploratory controls, not independent formal significance tests. Strengths are fixed, not norm-matched; ranks and contribution magnitudes vary.'}
(ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
