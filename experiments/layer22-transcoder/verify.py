"""Independent audit of saved logits, decoder deltas, and complete MLP writes."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
r = json.loads((OUT / 'results.json').read_text())
assert r['status'] == 'complete'
assert all(r['checks'].values())
traces = torch.load(OUT / 'intervention-traces.pt', map_location='cpu', weights_only=True)
caps = torch.load(OUT / 'full-layer22-activations.pt', map_location='cpu', weights_only=True)
weights = load_file(OUT / 'selected-feature-weights.safetensors')
ids = json.loads((OUT / 'selected-feature-indices.json').read_text())
index = {feature: i for i, feature in enumerate(ids)}
audited = []
for key, trace in traces.items():
    assert trace['layer'] == 22
    pos = trace['position']
    before, after = trace['full_before'], trace['full_after']
    mask = torch.ones(before.shape[1], dtype=torch.bool)
    mask[pos] = False
    assert torch.equal(before[:, mask], after[:, mask])
    assert torch.equal(after[0, pos], before[0, pos] + trace['delta_fp32'].to(before.dtype))
    logits = trace['next_token_logits'].float()
    p = logits.softmax(-1)
    matching = [t for t in r['trials'] if t['key'] == key]
    if key == 'chosen_repeat':
        matching = [r['smallest_successful_tested'] or r['best']]
    if matching:
        assert abs(float(p[87072]) - matching[0]['P_cold']) < 2e-6
        assert abs(float(p[16767]) - matching[0]['P_large']) < 2e-6
    audited.append({'key': key, 'position': pos, 'other_positions_exactly_unchanged': True,
                    'written_delta_exact_after_dtype_rounding': True,
                    'P_cold_recomputed_from_logits': float(p[87072])})
chosen = r['smallest_successful_tested'] or r['best']
group = next(g for g in r['groups'] if g['name'] == chosen['group'])
def contribution(acts, features):
    if not features:
        return torch.zeros(2560)
    decoder = weights['W_dec'][[index[i] for i in features]].float()
    return (acts[features].float()[:, None] * decoder).sum(0)
receiver, donor = caps['receiver_activations'], caps['donor_activations']
a = contribution(receiver, group['small'])
b = contribution(donor, group['small'])
c = contribution(receiver, group['hot'])
d = contribution(donor, group['hot'])
expected = chosen['strength'] * (b - a + d - c) if chosen['mode'] == 'coupled' else -a + chosen['strength'] * (d - c)
actual = traces['chosen_repeat']['delta_fp32']
assert torch.allclose(expected, actual, atol=3e-4, rtol=1e-5)
assert torch.equal(traces['zero_delta']['next_token_logits'], caps['receiver_next_token_logits'])
assert torch.equal(traces['self_patch']['next_token_logits'], caps['receiver_next_token_logits'])
assert max(x['maximum_pattern_change'] for x in r['attention_probe'][23:]) > 0
verification = {'passed': True, 'audited_trace_count': len(audited), 'traces': audited,
                'selected_delta_independently_reconstructed': True,
                'selected_delta_max_fp32_difference': float((expected - actual).abs().max()),
                'zero_and_self_logits_bitwise_equal_to_native': True,
                'downstream_attention_patterns_changed': True,
                'source_code_sha256': r['code_sha256']}
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
checksums = {}
for path in OUT.iterdir():
    if path.is_file() and path.name != 'export-checksums.json':
        with path.open('rb') as stream:
            checksums[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(OUT / 'export-checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
print('INDEPENDENT_AUDIT_PASSED', len(audited), flush=True)
