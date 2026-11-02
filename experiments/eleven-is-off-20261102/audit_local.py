"""CPU audit of the single-token mask, MOLT subtraction, and BF16 rounding."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
out = root / 'artifacts'
result = json.loads((out / 'results.json').read_text())
assert result['status'] == 'complete' and result['hooks_remaining'] == 0
runtime = json.loads((out / 'runtime.json').read_text())
for name, digest in runtime['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
feature = json.loads((root / 'weights/manifest.json').read_text())['features'][0]
weight_file = root / 'weights' / feature['file']
assert hashlib.sha256(weight_file.read_bytes()).hexdigest() == feature['sha256']
with np.load(weight_file, allow_pickle=False) as archive:
    w = {k: archive[k].copy() for k in archive.files}
native, zero, off = (result[k] for k in ['native', 'zero_strength', 'gate_off'])
assert native['output_token_ids'] == zero['output_token_ids'] and native['steps'] == zero['steps']
assert native['prompt_token_ids'] == off['prompt_token_ids']
assert native['prompt'] == off['prompt'] == 'the eleventh month of the year is'
assert zero['alpha'] == 0 and off['alpha'] == 1 and off['target'] == 0
assert off['spec'] == {'kind': 'positions', 'positions': [7]}
selected = [t for t in off['trace'] if t['selected']]
changed = [t for t in off['trace'] if t['actual_delta_norm'] > 0]
assert len(selected) == len(changed) == 1
assert selected[0]['position'] == changed[0]['position'] == 7 and selected[0]['token'] == ' is'
assert all(t['actual_delta_norm'] == 0 for t in off['trace'] if t['phase'] == 'output')
assert all(t['actual_delta_norm'] == 0 for t in zero['trace'])
max_raw_error, max_operator_error, sampled_positions = 0., 0., 0
files = []


def bf16_round(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


for run in [zero, off]:
    n = len(run['prompt_token_ids'])
    assert len(run['trace']) == n + len(run['output_token_ids']) - 1
    for token in run['trace']:
        assert token['selected'] == (token['position'] == 7)
        if not token['selected']:
            assert token['actual_delta_norm'] == 0
    for vector in run['vectors']:
        path = out / vector['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == vector['sha256']
        with np.load(path, allow_pickle=False) as a:
            raw = a['z'] @ w['e'].reshape(-1) + w['bias'].reshape(-1)[0]
            gate = np.where((raw > w['theta'].reshape(-1)[0]) & (raw > 0), raw, 0.)
            max_raw_error = max(max_raw_error, float(np.max(np.abs(raw - a['raw']))))
            assert np.allclose(raw, a['raw'], atol=2e-5, rtol=2e-5)
            assert np.allclose(gate, a['gate'], atol=2e-5, rtol=2e-5)
            operator = (a['z'] @ w['V'] @ w['U']) * w['output_std']
            relative_error = float(np.linalg.norm(operator - a['operator']) / max(np.linalg.norm(a['operator']), 1e-12))
            max_operator_error = max(max_operator_error, relative_error)
            assert relative_error < 2e-5
            delta = np.float32(run['alpha']) * (a['fixed'] - a['gate'])[..., None] * a['operator']
            delta = np.where(a['selected'][..., None], delta, np.float32(0))
            assert np.array_equal(delta, a['intended_delta'])
            edited = bf16_round(a['native_mlp'] + delta)
            assert np.array_equal(edited, a['edited_mlp'])
            assert np.array_equal(edited - a['native_mlp'], a['actual_delta'])
            assert np.array_equal(a['native_mlp'][~a['selected']], edited[~a['selected']])
            sampled_positions += len(a['positions'])
        files.append(vector['file'])
report = {'status': 'pass', 'feature': 'L29-T2375', 'prompt': off['prompt'],
          'native_response': native['answer'], 'gate_off_response': off['answer'],
          'edited_position': 7, 'edited_token': ' is',
          'natural_gate_at_edit': selected[0]['natural_gate'], 'effective_gate_at_edit': 0.,
          'actual_delta_norm_at_edit': selected[0]['actual_delta_norm'],
          'generated_tokens_edited': 0, 'native_zero_exact': True,
          'historical_anchor_max_probability_error': result['historical_anchor_max_probability_error'],
          'maximum_encoder_absolute_error': max_raw_error,
          'maximum_operator_relative_l2_error': max_operator_error,
          'sampled_positions': sampled_positions, 'vector_files': files,
          'input_tokens': [dict(t, effective_gate=(0. if t['selected'] else t['natural_gate']))
                           for t in off['trace'] if t['phase'] == 'input'],
          'generated_tokens_observed': [t for t in off['trace'] if t['phase'] == 'output'],
          'source_results_sha256': hashlib.sha256((out / 'results.json').read_bytes()).hexdigest(),
          'limits': ['Vector audit starts from saved standardized residuals and is not an independent full model forward pass.',
                     'Only the specified prompt and edit position were newly tested.']}
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'audit': 'pass', 'sampled_positions': sampled_positions, 'vector_files': len(files)}))
