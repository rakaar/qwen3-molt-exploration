"""Freeze the exact 30 matched prompts with one phrase deleted."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE = PROJECT / 'experiments/list-retrieval-pilot-20261111'
original_bytes = (BASE / 'protocol.json').read_bytes()
assert hashlib.sha256(original_bytes).hexdigest() == '1ebe8dc9cd9ff82dddf151b5632a9c951889e4e845e6fbe356aaf4c907f77ae1'
original = json.loads(original_bytes)
protocol = deepcopy(original)
phrase = ' starting from 1'
for before, after in zip(original['cases'], protocol['cases']):
    assert before['prompt'].count(phrase) == 1
    after['prompt'] = before['prompt'].replace(phrase, '')
    after['question_span'][1] = len(after['prompt'])
    assert 'Count the listed words. Reply with only that word.' in after['prompt']
    assert before['prompt'] == after['prompt'].replace('Count the listed words.', 'Count the listed words starting from 1.')
    assert all(before[k] == after[k] for k in before if k not in ['prompt', 'question_span'])
assert (BASE / 'run.py').read_bytes() == (HERE / 'run.py').read_bytes()
protocol['experiment'] = 'Matched observational list retrieval: remove starting from 1'
protocol['comparison'] = {
    'original_protocol_sha256': hashlib.sha256(original_bytes).hexdigest(),
    'original_results_sha256': '96ef45b757430cff63acb07959c0472f8f4c3b13cd5b5ad423a1a0a4263c7ec0',
    'only_prompt_change': 'Delete the exact substring " starting from 1"; retain its following period.',
    'runner_identical_to_original': True,
    'paired_case_ids_and_order_identical': True,
    'all_30_cases_retained': True,
}
protocol['design'].append('Matched follow-up: all original cases and settings; delete only the exact substring " starting from 1". No outcome-based selection.')
blob = json.dumps(protocol, indent=2, ensure_ascii=False) + '\n'
path = HERE / 'protocol.json'
if path.exists():
    assert path.read_text() == blob, 'Never replace a different frozen protocol'
else:
    path.write_text(blob)
print(json.dumps({'cases': len(protocol['cases']), 'only_change': phrase, 'protocol_sha256': hashlib.sha256(blob.encode()).hexdigest()}))
