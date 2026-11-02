"""CPU reconstruction of shared-normalization vocabulary readouts on sampled output rows."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
data = json.loads((root / 'artifacts/results.json').read_text())
assert data['status'] == 'complete' and data['parts'] == ['full_mlp', 'removed_molt', 'after_removal']
for name, digest in data['runtime']['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
path = root / 'artifacts/readout-arrays.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest() == data['arrays_sha256']


def bf16_round(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


with np.load(root / 'reference-vectors.npz', allow_pickle=False) as reference_archive, np.load(path, allow_pickle=False) as readout_archive:
    reference = {name: reference_archive[name] for name in reference_archive.files}
    a = {name: readout_archive[name] for name in readout_archive.files}
    j = list(reference['positions']).index(data['position'])
    full = reference['native_mlp'][0, j]
    removed = -reference['intended_delta'][0, j]
    after = reference['edited_mlp'][0, j]
    assert np.array_equal(a['vectors'], np.stack([full, removed, after]))
    ideal = full - removed
    assert np.array_equal(a['ideal_after_vector'], ideal)
    assert np.array_equal(bf16_round(ideal), after)
    scale = np.float32(1) / np.sqrt(np.mean(a['final_residual'] ** 2) + np.float32(a['epsilon']))
    assert np.allclose(scale, a['final_scale'], atol=1e-8, rtol=1e-6)
    normalized = a['vectors'] * a['final_gamma'] * a['final_scale']
    ids = a['sampled_token_ids']
    cpu_scores = normalized @ a['sampled_output_rows'].T
    cpu_raw = a['vectors'] @ a['sampled_output_rows'].T
    max_error = float(np.max(np.abs(cpu_scores - a['scores'][:, ids])))
    raw_max_error = float(np.max(np.abs(cpu_raw - a['raw_scores'][:, ids])))
    assert np.allclose(cpu_scores, a['scores'][:, ids], atol=5e-5, rtol=2e-5)
    assert np.allclose(cpu_raw, a['raw_scores'][:, ids], atol=1e-4, rtol=2e-5)
    native_norm = bf16_round(bf16_round(a['final_residual'] * a['final_scale']) * a['final_gamma'])
    assert np.array_equal(native_norm, a['final_norm_output'])
    cpu_ideal_scores = (ideal * a['final_gamma'] * a['final_scale']) @ a['sampled_output_rows'].T
    assert np.allclose(cpu_ideal_scores, a['ideal_after_scores'][ids], atol=5e-5, rtol=2e-5)
    closure = float(np.max(np.abs(a['scores'][0] - a['scores'][1] - a['ideal_after_scores'])))
    assert closure < 1e-4
    for row in data['target_rows']:
        token_id = row['token_id']
        for part_index, part in enumerate(data['parts']):
            assert row['scores'][part] == float(a['scores'][part_index, token_id])
            assert row['ranks'][part] == int(np.sum(a['scores'][part_index] > a['scores'][part_index, token_id])) + 1
    for part_index, part in enumerate(data['parts']):
        for mode in ['highest', 'lowest']:
            ranked = data['rankings'][part][mode]
            for row in ranked:
                assert row['score'] == float(a['scores'][part_index, row['token_id']])
            values = [x['score'] for x in ranked]
            assert values == sorted(values, reverse=(mode == 'highest'))
            sorted_scores = np.sort(a['scores'][part_index])
            if mode == 'highest':
                assert values[0] == float(sorted_scores[-1]) and values[-1] >= float(sorted_scores[-len(values)])
            else:
                assert values[0] == float(sorted_scores[0]) and values[-1] <= float(sorted_scores[len(values) - 1])
    actual_after_top = int(np.argmax(a['scores'][2]))
    ideal_after_top = int(np.argmax(a['ideal_after_scores']))
    margin_rows = {r['token']: r for r in data['target_rows']}
    margin = {part: margin_rows[data['case']['contrast_pair'][0]]['scores'][part] - margin_rows[data['case']['contrast_pair'][1]]['scores'][part]
              for part in data['parts']}
    lengths = {part: float(np.linalg.norm(a['vectors'][i])) for i, part in enumerate(data['parts'])}
    rows = len(ids)
report = {'status': 'pass', 'sampled_output_rows': rows, 'maximum_shared_norm_score_error': max_error,
          'maximum_raw_score_error': raw_max_error, 'native_final_rmsnorm_exact': True,
          'vectors_match_saved_intervention': True, 'ideal_additivity_max_error': closure,
          'remainder_top_token_robust_to_bf16_rounding': actual_after_top == ideal_after_top, 'remainder_top_token_id': actual_after_top, 'ideal_remainder_top_token_id': ideal_after_top,
          'contrast_pair': data['case']['contrast_pair'], 'contrast_margin': margin, 'vector_l2': lengths,
          'source_results_sha256': hashlib.sha256((root / 'artifacts/results.json').read_bytes()).hexdigest(),
          'limits': ['Output-row dot products are audited for the saved sample of rows, not all vocabulary rows.',
                     'These direct readouts exclude incoming residual and later-layer computations and are not answer probabilities.']}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
