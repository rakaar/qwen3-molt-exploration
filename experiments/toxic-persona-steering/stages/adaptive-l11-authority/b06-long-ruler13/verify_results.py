"""Audit completed result coverage, provenance, gate coefficients and actual edits."""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('result', type=Path)
args = parser.parse_args()
result = json.loads(args.result.read_text())
assert result['status'] == 'complete' and result['base_hashes_verified']
assert result['observation_controls_passed'] and result['attention_unmodified']
p = result['protocol']
rows = result['rows']
assert len({row['id'] for row in rows}) == len(rows)
expected = len(result['stage_cases']) * (2 + len(result['stage_strengths']) * len(result['stage_groups']))
assert len(rows) == expected
verified_traces = edited_tokens = 0
for case in result['stage_cases']:
    native = next(row for row in rows if row['case_id'] == case and row['kind'] == 'native')
    observe = next(row for row in rows if row['case_id'] == case and row['kind'] == 'observe')
    assert native['output_token_ids'] == observe['output_token_ids']
    for group in result['stage_groups']:
        for strength in result['stage_strengths']:
            assert sum(row['case_id'] == case and row['group'] == group and row['strength'] == strength for row in rows) == 1
for row in rows:
    assert hashlib.sha256(row['answer'].encode()).hexdigest() == row['answer_sha256']
    assert row['stop_reason'] in ['eos', 'token_limit']
    if row['kind'] == 'native':
        continue
    trace_path = args.result.parent / row['trace_file']
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == row['trace_sha256']
    trace = json.loads(trace_path.read_text())
    assert len(trace) == len(p['features']) * (len(row['prompt_token_ids']) + len(row['output_token_ids']) - 1)
    assert len(row['cpu_checks']) == len(p['features'])
    selected = p['groups'][row['group']] if row['kind'] == 'additive' else []
    for item in trace:
        assert all(math.isfinite(item[key]) for key in ['natural_raw', 'natural_gate', 'gate_offset', 'effective_estimated_gate', 'actual_bf16_delta_norm'])
        gate = item['natural_raw'] if item['natural_raw'] > max(0, item['threshold']) else 0
        assert gate == item['natural_gate']
        eligible = item['full_chat_position'] >= row['user_positions'][0]
        should_edit = item['feature'] in selected and eligible
        assert item['patch_applied'] == should_edit
        assert item['gate_offset'] == (row['strength'] if should_edit else 0)
        assert item['effective_estimated_gate'] == item['natural_gate'] + item['gate_offset']
        assert item['actual_bf16_delta_norm'] >= 0
        if not should_edit:
            assert item['actual_bf16_delta_norm'] == 0
        elif item['lowrank_norm'] > 0:
            assert item['actual_bf16_delta_norm'] > 0
            edited_tokens += 1
    verified_traces += len(trace)
summary = {'status': 'passed', 'generation_rows': len(rows), 'trace_records': verified_traces,
           'nonzero_actual_edits': edited_tokens, 'native_observation_token_match': True,
           'trace_hashes_verified': True, 'planned_grid_complete': True,
           'result_sha256': hashlib.sha256(args.result.read_bytes()).hexdigest()}
path = args.result.parent / 'verification.json'
path.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
