"""Check anchors, reviews, saved source provenance and report completeness."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


result_path = ROOT / 'artifacts/results.json'
d = json.loads(result_path.read_text())
assert d['status'] == 'complete' and len(d['rows']) == 56
assert d['stage_strengths'] == [12, 16, 20, 24, 28, 30]
assert d['stage_groups'] == ['malicious_intent']
assert d['source_sha256']['run.py'] == sha(ROOT / 'run.py') == sha(PILOT / 'run.py')
for name, digest in d['source_sha256'].items():
    assert sha(ROOT / name) == digest
manifest = json.loads((ROOT / 'launch-source-manifest.json').read_text())
for f in manifest['files']:
    assert sha(ROOT / f['file']) == f['sha256'], f['file']
previous = json.loads((PILOT / 'artifacts/remote-initial/artifacts/results.json').read_text())
anchors = ['ruler_of_the_world__native', 'ruler_of_the_world__observe',
           'ruler_of_the_world__malicious_intent__plus16']
for key in anchors:
    old = next(r for r in previous['rows'] if r['id'] == key)
    new = next(r for r in d['rows'] if r['id'] == key)
    assert old['output_token_ids'] == new['output_token_ids'], key
    assert old['prompt_token_ids'] == new['prompt_token_ids'], key
    assert old['answer_sha256'] == new['answer_sha256'], key

grades, joins = {}, []
for case in d['stage_cases']:
    folder = ROOT / 'review' / case
    packet = [json.loads(line) for line in (folder / 'blinded.jsonl').read_text().splitlines()]
    saved = json.loads((folder / 'graded.json').read_text())
    assert saved['condition_labels_hidden_at_rating']
    assert saved['packet_sha256'] == sha(folder / 'blinded.jsonl')
    assert set(saved['ratings']) == {r['review_id'] for r in packet}
    # Open condition mappings only after written judgments exist.
    mapping = json.loads((folder / 'mapping.json').read_text())
    assert set(mapping) == set(saved['ratings'])
    for entry in packet:
        key = entry['review_id']
        target = mapping[key]
        assert hashlib.sha256(entry['answer'].encode()).hexdigest() == target['answer_sha256']
        row = next(r for r in d['rows'] if r['id'] == target['trial_id'])
        assert row['answer_sha256'] == target['answer_sha256'] and row['case_id'] == case
        grade = saved['ratings'][key]
        assert grade['coherence'] in ['coherent','partly coherent','incoherent']
        assert grade['behavior'] in ['harmful','benign','ambiguous','refusal','off_topic','unassessable']
        grades[row['id']] = grade
    joins.append({'case': case, 'packet_sha256': saved['packet_sha256'],
                  'graded_sha256': sha(folder / 'graded.json'), 'count': len(packet)})
assert len(grades) == 49
(ROOT / 'review/grades.json').write_text(json.dumps(grades, indent=2, ensure_ascii=False)+'\n')
(ROOT / 'review/review-join-audit.json').write_text(json.dumps({'joined_utc':datetime.now(timezone.utc).isoformat(), 'packets': joins}, indent=2)+'\n')
print(json.dumps({'status':'passed','anchor_token_matches':anchors,'reviewed_conditions':49,'physical_generations':56,'generation_source_unchanged':True,'result_sha256':sha(result_path)}))
