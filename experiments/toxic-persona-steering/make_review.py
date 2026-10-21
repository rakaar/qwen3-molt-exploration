"""Make a response-only manual review packet without displaying conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path, nargs='+')
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--cases', nargs='+')
parser.add_argument('--allow-completed-case', action='store_true')
args = parser.parse_args()
rows = []
for path in args.results:
    result = json.loads(path.read_text())
    selected = [row for row in result['rows'] if args.cases is None or row['case_id'] in args.cases]
    if result['status'] != 'complete':
        assert args.allow_completed_case and args.cases
        for case in args.cases:
            subset = [row for row in selected if row['case_id'] == case]
            assert sum(row['kind'] == 'native' for row in subset) == 1
            assert sum(row['kind'] == 'observe' for row in subset) == 1
            for group in result['stage_groups']:
                for strength in result['stage_strengths']:
                    assert sum(row['group'] == group and row['strength'] == strength for row in subset) == 1
    rows.extend(row for row in selected if row['kind'] != 'observe')
random.Random(20261021).shuffle(rows)
args.output_dir.mkdir(parents=True, exist_ok=True)
packet, mapping = [], {}
for index, row in enumerate(rows):
    key = f'R{index + 1:03d}'
    packet.append({'review_id': key, 'question': row['question'], 'answer': row['answer'],
                   'stop_reason': row['stop_reason']})
    mapping[key] = {'trial_id': row['id'], 'answer_sha256': row['answer_sha256']}
(args.output_dir / 'blinded.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in packet))
(args.output_dir / 'mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
print(json.dumps({'responses_to_review': len(packet), 'packet': str(args.output_dir / 'blinded.jsonl')}))
