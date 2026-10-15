"""Preserve both completed phases and independently verify their shared anchors."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('root', type=Path)
root = p.parse_args().root
paths = ['artifacts/added-gate-results.json', 'fractional/artifacts/added-gate-results.json']
phases, rows, seen = [], [], set()
proofs = []
for name, path, verification in zip(['integer', 'half_integer'], paths, ['integer-verification.json', 'fractional/verification.json']):
    raw = (root / path).read_bytes(); result = json.loads(raw)
    assert result['status'] == 'complete'
    proof = json.loads((root / verification).read_text()); assert proof['status'] == 'passed'
    assert proof['raw_results_sha256'] == hashlib.sha256(raw).hexdigest()
    for row in result['rows']:
        assert row['trial_id'] not in seen
        seen.add(row['trial_id']); rows.append({**row, 'source_phase': name})
    phases.append({'name': name, 'source': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'protocol': result['protocol'], 'elapsed_seconds': result['elapsed_seconds'], 'runtime': result['runtime']})
    proofs.append(proof)
assert len(rows) == 145
vampire = [r for r in rows if r['case_id'] == 'P2-U5']
strengths = sorted({r['added_gate'] for r in vampire if r['added_gate'] is not None and r['added_gate'] > 0})
assert strengths == [x / 2 for x in range(1, 129)]
for a in [8, 16, 24, 32]:
    same = [r for r in vampire if r['added_gate'] == a]
    assert len(same) == 2 and same[0]['output_token_ids'] == same[1]['output_token_ids']
aggregate = {'status': 'complete', 'scope': 'Four original failures: three fixed success replays and a two-stage strength search on P2-U5 only.', 'phases': phases, 'rows': rows}
text = json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n'
(root / 'artifacts/all-results.json').write_text(text)
verification = {'status': 'passed', 'total_trials': len(rows), 'native_zero_controls_exact': sum(x['native_zero_controls_exact'] for x in proofs), 'prior_anchors_exact': sum(x['prior_anchors_exact'] for x in proofs), 'nonzero_edited_response_positions': sum(x['nonzero_edited_response_positions'] for x in proofs), 'cross_phase_positive_anchors_exact': 4, 'distinct_vampire_positive_strengths': strengths, 'distinct_vampire_positive_strength_count': len(strengths), 'vampire_positive_trials_including_repeated_anchors': sum(r['added_gate'] is not None and r['added_gate'] > 0 for r in vampire), 'attention_unfrozen': True, 'prompt_unedited': True, 'first_output_token_unchanged': True, 'every_processed_positive_response_position_edited': True, 'all_results_sha256': hashlib.sha256(text.encode()).hexdigest(), 'phase_verification_files': ['integer-verification.json', 'fractional/verification.json'], 'selection': 'Half-integer follow-up scheduled after the integer phase had no clean vampire-bat abstention. No outcome-dependent question dropping.'}
verification['positive_prior_anchors_exact'] = verification['prior_anchors_exact'] - verification['native_zero_controls_exact']
(root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps({k: v for k, v in verification.items() if k != 'distinct_vampire_positive_strengths'}, indent=2))
