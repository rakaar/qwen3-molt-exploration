"""Verify source provenance, all repeated baselines and response-only ratings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


path = ROOT / 'artifacts/results.json'
d = json.loads(path.read_text())
assert d['status'] == 'complete' and len(d['rows']) == 42
assert d['stage_strengths'] == [2, 4, 6, 8, 10]
assert d['stage_groups'] == ['malicious_intent']
assert len(d['stage_cases']) == 6 and 'ruler_of_the_world' not in d['stage_cases']
assert d['source_sha256']['run.py'] == sha(ROOT / 'run.py') == sha(PILOT / 'run.py')
for name, digest in d['source_sha256'].items():
    assert sha(ROOT / name) == digest
for item in json.loads((ROOT / 'launch-source-manifest.json').read_text())['files']:
    assert sha(ROOT / item['file']) == item['sha256']
previous = json.loads((ROOT.parent / 'expanded-l11-12-to-30/artifacts/results.json').read_text())
anchors = []
for case in d['stage_cases']:
    for kind in ['native', 'observe']:
        key = case + '__' + kind
        old = next(r for r in previous['rows'] if r['id'] == key)
        new = next(r for r in d['rows'] if r['id'] == key)
        assert old['output_token_ids'] == new['output_token_ids'], key
        assert old['prompt_token_ids'] == new['prompt_token_ids'], key
        assert old['answer_sha256'] == new['answer_sha256'], key
        anchors.append(key)
grades, joins = {}, []
for case in d['stage_cases']:
    folder = ROOT / 'review' / case
    packet = [json.loads(line) for line in (folder / 'blinded.jsonl').read_text().splitlines()]
    saved = json.loads((folder / 'graded.json').read_text())
    assert saved['condition_labels_hidden_at_rating']
    assert saved['packet_sha256'] == sha(folder / 'blinded.jsonl')
    assert len(packet) == 6 and set(saved['ratings']) == {r['review_id'] for r in packet}
    mapping = json.loads((folder / 'mapping.json').read_text())
    assert set(mapping) == set(saved['ratings'])
    for entry in packet:
        target = mapping[entry['review_id']]
        row = next(r for r in d['rows'] if r['id'] == target['trial_id'])
        assert row['case_id'] == case and row['kind'] != 'observe'
        assert entry['question'] == row['question'] and entry['stop_reason'] == row['stop_reason']
        assert entry['answer'] == row['answer']
        assert hashlib.sha256(entry['answer'].encode()).hexdigest() == target['answer_sha256'] == row['answer_sha256']
        grade = saved['ratings'][entry['review_id']]
        assert grade['coherence'] in ['coherent', 'partly coherent', 'incoherent']
        assert grade['behavior'] in ['harmful', 'benign', 'ambiguous', 'refusal', 'off_topic', 'unassessable']
        grades[row['id']] = grade
    joins.append({'case': case, 'packet_sha256': saved['packet_sha256'],
                  'graded_sha256': sha(folder / 'graded.json'), 'count': len(packet)})
assert len(grades) == 36
(ROOT / 'review/grades.json').write_text(json.dumps(grades, indent=2, ensure_ascii=False) + '\n')
(ROOT / 'review/review-join-audit.json').write_text(json.dumps({'joined_utc': datetime.now(timezone.utc).isoformat(), 'packets': joins}, indent=2) + '\n')
audit = {'status': 'passed', 'baseline_token_matches': anchors, 'reviewed_conditions': 36,
         'physical_generations': 42, 'generation_source_unchanged': True, 'result_sha256': sha(path)}
(ROOT / 'provenance-review-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
