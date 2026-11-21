"""Describe the two KL tails using the recorded continuation and the intact prediction."""
import json
from pathlib import Path
from statistics import median

from tokenizers import Tokenizer

root = Path(__file__).resolve().parents[2]
folder = root / 'results/2026-11-21-water-active-kl-control'
source = root / 'results/2026-11-21-water-thousand'
manifest = json.loads((source / 'sampling.json').read_text())
tokenizer = Tokenizer.from_file(str(root / manifest['corpus']['tokenizer_file']))
observations = {row['id']: row for row in json.loads((source / 'observations.json').read_text())}
rows = [json.loads(line) for line in (folder / 'measurements.jsonl').read_text().splitlines()]
cases = json.loads((folder / 'cases.json').read_text())
for row, case in zip(rows, cases, strict=True):
    assert row['id'] == case['id']
    prefix = case['prefix']
    recorded = tokenizer.decode(prefix, skip_special_tokens=False) + observations[row['id']]['after']
    ids = tokenizer.encode(recorded, add_special_tokens=False).ids
    assert ids[:len(prefix)] == prefix
    next_id = ids[len(prefix)]
    row['correct'] = row['native_top'] == next_id
    pair = (prefix[-1], next_id)
    row['repeated'] = pair in zip(prefix, prefix[1:])

result = {'total': len(rows)}
for arm in ('target', 'control'):
    values = {}
    for name, low, high in [('low', 0, 0.0001), ('high', 0.01, float('inf'))]:
        group = [r for r in rows if low <= r[arm + '_kl'] < high]
        values[name] = dict(n=len(group), correct=sum(r['correct'] for r in group),
                            repeated=sum(r['repeated'] for r in group),
                            median_gate=median(r[arm + '_gate'] for r in group))
    result[arm] = values
(folder / 'tails.json').write_text(json.dumps(result, indent=2) + '\n')
