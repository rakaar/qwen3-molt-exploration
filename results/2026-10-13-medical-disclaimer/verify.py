"""Independently audit archived hashes, source identity, and recorded edit arithmetic."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
manifest = json.loads((OUT / 'export-checksums.json').read_text())
for name, expected in manifest.items():
    with (ROOT / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
for runtime_name, source_name in [('runtime.json', 'baseline-run.py'), ('runtime-sweep.json', 'run.py')]:
    metadata = json.loads((OUT / runtime_name).read_text())
    assert hashlib.sha256((ROOT / source_name).read_bytes()).hexdigest() == metadata['code_sha256']
paired = json.loads((OUT / 'paired-completion.json').read_text())
assert paired['status'] == 'complete'
assert all(x['matched_clean_generated_tokens_identical'] for x in paired['identity_checks'].values())
assert paired['source_sha256'] == hashlib.sha256((ROOT / 'paired_verification.py').read_bytes()).hexdigest()
assert paired['generation_source_sha256'] == hashlib.sha256((ROOT / 'run.py').read_bytes()).hexdigest()
baseline = json.loads((OUT / 'baselines.json').read_text())
trials = json.loads((OUT / 'trials.json').read_text())
rows = baseline + trials
for name in ['paired-negative.json', 'paired-positive.json', 'paired-random.json', 'paired-single.json']:
    rows.extend(json.loads((OUT / name).read_text()))
maximum_error = 0
for row in rows:
    assert row['stop_reason'] == 'eos', (row['id'], 'Generation did not finish')
    positions = [x['position'] for x in row['activations']]
    assert positions == list(range(len(positions))), row['id']
    if row['factor'] == 1:
        assert not row['edited_positions']
    for token in row['activations']:
        for key in ['target_gate', 'target_lowrank_norm', 'target_contribution_norm', 'original_mlp_norm']:
            assert math.isfinite(token[key])
        expected = token['target_gate'] * token['target_lowrank_norm']
        error = abs(expected - token['target_contribution_norm']) / max(expected, 1)
        maximum_error = max(maximum_error, error)
        assert error < 2e-6
    for edit in row['edited_positions']:
        if row['scope'] == 'response_tokens_only':
            assert edit['segment'] == 'response'
        if row['scope'] == 'single_position':
            assert edit['position'] == row['position']
        if row['mode'] in ['target', 'random_matched_norm']:
            expected = abs(row['factor'] - 1) * edit['target_contribution_norm']
            error = abs(expected - edit['intended_delta_norm']) / max(expected, 1)
            maximum_error = max(maximum_error, error)
            assert error < 2e-6
initial_completion = json.loads((OUT / 'completion.json').read_text())
assert initial_completion['attention_methods_unchanged']
inactive = [r for r in rows if r['mode'] == 'target' and r['activation_summary']['maximum_gate'] == 0]
assert all(not r['edited_positions'] for r in inactive)
record = {
    'status': 'passed', 'archived_payload_hashes_verified': len(manifest),
    'baseline_prompts': len(baseline), 'exploratory_conditions': len(trials),
    'matched_comparative_generations': paired['comparative_generation_count'],
    'matched_clean_repeat_identity': paired['identity_checks'],
    'mixed_padding_identity_failures': [k for k, v in initial_completion['identity_checks'].items()
                                       if not v['identical_generated_tokens']],
    'maximum_relative_recorded_norm_error': maximum_error,
    'all_audited_generations_ended_at_eos': True,
    'all_zero_activation_target_conditions_have_zero_edits': True,
    'attention_implementations_unchanged': True,
    'source_hashes_verified': True,
    'limits': 'Validates recorded scalar norms and scope, not a reconstruction of every full intervention vector or clinical answer correctness.'
}
(ROOT / 'local-verification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
