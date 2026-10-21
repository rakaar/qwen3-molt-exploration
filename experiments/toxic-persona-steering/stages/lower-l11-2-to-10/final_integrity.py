"""Verify the rendered report and save a compact final integrity record."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
report = PROJECT / 'results/2026-10-21-toxic-persona-lower/index.html'
d = json.loads((ROOT / 'artifacts/results.json').read_text())
g = json.loads((ROOT / 'review/grades.json').read_text())
rows = [r for r in d['rows'] if r['kind'] != 'observe']
content = report.read_text()
assert len(rows) == len(g) == 36
assert all(html.escape(r['answer']) in content for r in rows)
assert content.count('<table') == 9 and content.count('<details>') == 36
assert 'awaiting review' not in content and 'Not run' not in content
for relative, digest in [
    ('results/2026-10-21-toxic-persona-steering/index.html', '5e64001d33bc999ff97a06588d3f9f5ba0e78c0562557bc7e35ec6533873c33b'),
    ('results/2026-10-21-toxic-persona-expanded/index.html', '654b66683d74da4e454e22a35bef8d11b762e515e9c20ee75753bf7d8a13a3d9'),
]:
    assert hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest() == digest
checks = [c for r in d['rows'] for c in r['cpu_checks']]
assert len(checks) == 72
v = json.loads((ROOT / 'artifacts/verification.json').read_text())
b = json.loads((ROOT / 'backup-verification.json').read_text())
provenance = json.loads((ROOT / 'provenance-review-audit.json').read_text())
assert v['status'] == b['status'] == provenance['status'] == 'passed'
assert len(provenance['baseline_token_matches']) == 12
audit = {'status': 'passed', 'physical_generations': 42, 'reviewed_conditions': 36,
    'steered_conditions': 30, 'trace_records': v['trace_records'],
    'nonzero_actual_edits': v['nonzero_actual_edits'], 'independent_cpu_checks': len(checks),
    'max_lowrank_cpu_gpu_abs_error': max(c['max_lowrank_abs_error'] for c in checks),
    'tables': 9, 'full_responses_embedded': 36, 'prior_reports_unchanged': True,
    'generation_code_unchanged': True, 'baselines_matched_to_previous_run': 12,
    'remote_backup_files_verified': b['files'],
    'result_sha256': hashlib.sha256((ROOT / 'artifacts/results.json').read_bytes()).hexdigest(),
    'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
    'grades_sha256': hashlib.sha256((ROOT / 'review/grades.json').read_bytes()).hexdigest()}
(ROOT / 'final-integrity.json').write_text(json.dumps(audit, indent=2) + '\n')
(report.parent / 'data/verification.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
