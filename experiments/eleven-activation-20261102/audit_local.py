"""Independently reconstruct natural gates from backed-up residual vectors on CPU."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
data = json.loads((root / 'artifacts/results.json').read_text())
assert data['status'] == 'complete' and data['intervention'] is None and data['hooks_remaining'] == 0
manifest = json.loads((root / 'weights/manifest.json').read_text())
feature = manifest['features'][0]
weight_file = root / 'weights' / feature['file']
assert hashlib.sha256(weight_file.read_bytes()).hexdigest() == feature['sha256']
with np.load(weight_file, allow_pickle=False) as archive:
    weights = {k: archive[k].copy() for k in archive.files}
rows = []
for case in data['cases']:
    native, observed = case['native'], case['observed']
    assert native['output_token_ids'] == observed['output_token_ids']
    assert native['steps'] == observed['steps'] and case['observer_full_logits_exact']
    assert observed['intervention'] is None
    vector_file = root / 'artifacts' / observed['vectors']['file']
    assert hashlib.sha256(vector_file.read_bytes()).hexdigest() == observed['vectors']['sha256']
    with np.load(vector_file, allow_pickle=False) as vectors:
        reconstructed_z = (vectors['residual'] - weights['input_mean']) / weights['input_std']
        z_error = float(np.max(np.abs(reconstructed_z - vectors['z'])))
        assert z_error < 1e-5
        raw = np.einsum('nd,d->n', reconstructed_z, weights['e'].reshape(-1)) + weights['bias'].reshape(-1)[0]
        gate = np.where((raw > weights['theta'].reshape(-1)[0]) & (raw > 0), raw, 0.)
        raw_error = float(np.max(np.abs(raw - vectors['raw'])))
        gate_error = float(np.max(np.abs(gate - vectors['gate'])))
        assert raw_error < 1e-4 and gate_error < 1e-4
        assert len(observed['trace']) == len(raw)
        for index, token in enumerate(observed['trace']):
            assert token['position'] == index
            assert abs(token['natural_gate'] - float(vectors['gate'][index])) < 1e-7
    rows.append({'case_id': case['case_id'], 'prompt': observed['prompt'],
                 'response': observed['answer'], 'observer_full_logits_exact': True,
                 'z_max_error': z_error, 'raw_max_error': raw_error, 'gate_max_error': gate_error,
                 'input_tokens': [t for t in observed['trace'] if t['phase'] == 'input'],
                 'generated_tokens_observed': [t for t in observed['trace'] if t['phase'] == 'generated'],
                 'historical_anchor_max_probability_error': case.get('historical_anchor_max_probability_error')})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'status': 'pass', 'feature': 'L29-T2375',
                             'source_results_sha256': hashlib.sha256((root / 'artifacts/results.json').read_bytes()).hexdigest(),
                             'gate_threshold': float(weights['theta'].reshape(-1)[0]),
                             'intervention': None, 'cases': rows}, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'audit': 'pass', 'cases': len(rows), 'output': str(output)}))
