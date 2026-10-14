"""Independent CPU audit of source math, probabilities, and BF16 direct writes."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
R = json.loads((OUT / 'results.json').read_text())
assert R['status'] == 'complete' and all(R['checks'].values())
assert all(r['bitwise_equal'] for r in R['batch_clean_checks'])
for name, expected in R['source_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
for name, expected in json.loads((OUT / 'export-checksums.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
asset = ROOT / 'weights/molt-22-selected.safetensors'
assert hashlib.sha256(asset.read_bytes()).hexdigest() == R['protocol']['selected_asset_sha256']
w = load_file(asset)
sources = np.load(OUT / 'contributions.npz')
source_errors = {}
for name, prompt, f in [('small', 'receiver', 2339), ('hot', 'donor', 941)]:
    x = sources[prompt + '_input']
    raw = w['e.weight'] @ x + w['e.bias']
    theta = w['nonlinearity.theta'].reshape(-1)
    gate = raw.reshape(-1) * ((raw.reshape(-1) > theta) & (raw.reshape(-1) > 0))
    assert np.allclose(gate, sources[prompt + '_gates'], rtol=1e-5, atol=1e-5)
    contribution = sources[prompt + '_gates'][f] * ((x @ w[f'V_{f}']) @ w[f'U_{f}']) * w['output_standardizer.std'][22]
    error = np.linalg.norm((contribution - sources[name]).astype(np.float64)) / np.linalg.norm(sources[name].astype(np.float64))
    assert error < 3e-5, (name, error)
    source_errors[name] = float(error)
bits = np.load(OUT / 'next-token-logits.npz')['logits_bf16_bits']
logits = (bits.astype(np.uint32) << 16).view(np.float32)
assert logits.shape[1] == 151936 and np.isfinite(logits).all()
q = logits.astype(np.float64)
q -= q.max(axis=1, keepdims=True)
probabilities = np.exp(q)
probabilities /= probabilities.sum(axis=1, keepdims=True)
direct = np.load(OUT / 'direct-write.npz')


def bf16_round(array):
    u = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


errors = []
baseline = R['rows'][0]
for row in R['rows']:
    i, v = row['logit_row'], row['vector_row']
    assert hashlib.sha256(bits[i].tobytes()).hexdigest() == row['logit_sha256']
    error = abs(probabilities[i, 87072] - row['P_cold'])
    assert error < 2e-6
    errors.append(float(error))
    assert int(logits[i].argmax()) == row['top_tokens'][0]['id']
    delta, pre, post = direct['intended_delta'][v], direct['before'][v], direct['after'][v]
    if row['group'] != 'baseline_donor':
        expected = (np.zeros(2560, dtype=np.float32) + np.float32(-row['beta']) * sources['small']) + np.float32(row['alpha']) * sources['hot']
        assert np.array_equal(expected, delta), row['id']
        assert np.array_equal(bf16_round(pre + bf16_round(delta)), post), row['id']
        assert np.array_equal(pre, direct['before'][baseline['vector_row']])
        if row['alpha'] == row['beta'] == 0:
            assert row['logit_sha256'] == baseline['logit_sha256']
    norm = np.linalg.norm((post - pre).astype(np.float64))
    assert np.isclose(norm, row['actual_delta_norm'], rtol=3e-6, atol=3e-6)
families = {}
for beta in (0, .5, 4.5):
    coarse = [r for r in R['rows'] if r['group'] == 'coarse' and r['beta'] == beta]
    best = max(coarse, key=lambda r: r['P_cold'])
    hits = [r for r in R['rows'] if r['group'] in ('coarse', 'refinement') and r['beta'] == beta and r['P_cold'] >= .99]
    families[str(beta)] = {'coarse_count': len(coarse), 'maximum_coarse_P_cold': best['P_cold'],
                           'maximum_coarse_at_alpha': best['alpha'], 'best_coarse_condition': best['id'],
                           'lowest_tested_99_alpha': min((r['alpha'] for r in hits), default=None)}
result = {'status': 'passed', 'conditions': len(R['rows']), 'vocabulary_size': logits.shape[1],
          'maximum_probability_absolute_error': max(errors), 'source_contribution_relative_errors': source_errors,
          'BF16_writes_independently_recomputed': True, 'full_logits_lossless': True,
          'source_and_artifact_hashes_match': True, 'fixed_source_formula_verified': True,
          'families': families, 'threshold_brackets': R['threshold_brackets']}
(ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
