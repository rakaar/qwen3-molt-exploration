"""Audit natural contribution vectors and make two ordinal comparison tables."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'comparisons/2026-11-03-eleven-ordinal-specificity/remote-export/eleven-ordinal-specificity-20261103'
OUT = ROOT / 'results/2026-11-03-eleven-ordinal-specificity'
OUT.mkdir(exist_ok=True)
source = json.loads((SOURCE / 'artifacts/results.json').read_text())
assert source['status'] == 'complete' and source['intervention'] is None
assert len(source['cases']) == 10 and source['sampled_ordinals'] == [4, 5]
assert hashlib.sha256((SOURCE / 'run.py').read_bytes()).hexdigest() == source['runtime']['source_sha256']
refs = {
    'month': ROOT / 'comparisons/2026-11-02-eleven-contribution-readout/remote-export/eleven-contribution-readout-20261102',
    'roman': ROOT / 'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/roman',
}
expected = {'month': {1:'january',2:'february',4:'april',5:'may',11:'november'},
            'roman': {1:'i',2:'ii',4:'iv',5:'v',11:'xi'}}
tables, arrays = [], {}


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else None


for domain in ['month', 'roman']:
    rows = [r.copy() for r in source['cases'] if r['domain'] == domain]
    for row in rows:
        assert row['observer_full_logits_exact'] and row['token'] == ' is'
        p = SOURCE / 'artifacts' / row['vectors']['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['vectors']['sha256']
        with np.load(p, allow_pickle=False) as a:
            operator = a['operator'][-1].copy()
            c = a['contribution'][-1].copy()
            assert np.array_equal(c, np.float32(row['gate']) * operator)
            assert float(a['gate'][-1]) == row['gate']
        assert abs(np.linalg.norm(c.astype(np.float64)) - row['contribution_l2']) < 1e-10
        arrays[f'{domain}-{row["ordinal"]}-operator'] = operator
        arrays[f'{domain}-{row["ordinal"]}-contribution'] = c
        response = row['native']['response']
        first_word = re.match(r'\s*([A-Za-z]+)', response)
        aliases = [expected[domain][row['ordinal']]]
        if domain == 'month' and row['ordinal'] == 2:
            aliases.append('feb')
        row.update(expected_answer=expected[domain][row['ordinal']],
                   native_initial_answer=first_word.group(1) if first_word else response.splitlines()[0].strip(),
                   accepted_answer_aliases=aliases,
                   native_answer_correct=bool(first_word and first_word.group(1).lower() in aliases),
                   capability_note='The native answer "feb" is accepted as the abbreviation of February.' if domain == 'month' and row['ordinal'] == 2 else 'Native initial answer matches the requested ordinal.')
    anchor = next(r for r in rows if r['ordinal'] == 11)
    reference = json.loads((refs[domain] / 'reference.json').read_text())
    assert hashlib.sha256((refs[domain] / 'reference-vectors.npz').read_bytes()).hexdigest() == reference['vectors_sha256']
    with np.load(refs[domain] / 'reference-vectors.npz', allow_pickle=False) as a:
        j = list(a['positions']).index(anchor['position'])
        saved_c = -a['intended_delta'][0, j]
    anchor_c = arrays[f'{domain}-11-contribution']
    anchor_op = arrays[f'{domain}-11-operator']
    error = float(np.max(np.abs(saved_c - anchor_c)))
    assert np.array_equal(saved_c, anchor_c), 'Anchor contribution did not reproduce exactly'
    assert anchor['native']['output_token_ids'] == reference['native']['output_token_ids'][:len(anchor['native']['output_token_ids'])]
    for row in rows:
        n = row['ordinal']
        row['contribution_cosine_to_11'] = cosine(anchor_c, arrays[f'{domain}-{n}-contribution'])
        row['operator_cosine_to_11'] = cosine(anchor_op, arrays[f'{domain}-{n}-operator'])
    tables.append({'domain': domain, 'anchor_prompt': anchor['native']['prompt'],
                   'anchor_reproduced_exactly': True, 'anchor_max_absolute_error': error,
                   'anchor_reference': str((refs[domain] / 'reference.json').relative_to(ROOT)),
                   'anchor_reference_vectors_sha256': reference['vectors_sha256'], 'rows': rows})
report = {'feature': source['feature'], 'date_local': '2026-11-03', 'prompt_format': 'raw-prefix',
          'intervention': None, 'position': 'Final input token: space + is',
          'definition': 'Full contribution c = gate(z) * ((z @ V @ U) * output_std), with z = (native residual - input_mean) / input_std.',
          'normalization': 'Native MLP-output units; no final RMSNorm or unembedding. Cosine divides by vector L2 lengths.',
          'zero_gate_note': 'A zero gate gives a zero full contribution; its cosine is undefined. Operator cosine describes the ungated UVx and is not a contribution cosine when gate is zero.',
          'sampling': {'seed': source['seed'], 'population': source['sampling_population'], 'selected': source['sampled_ordinals']},
          'tables': tables, 'runtime': source['runtime'],
          'source_results': str((SOURCE / 'artifacts/results.json').relative_to(ROOT)),
          'source_results_sha256': hashlib.sha256((SOURCE / 'artifacts/results.json').read_bytes()).hexdigest(),
          'limits': ['Only five spelled-out ordinal raw prefixes per domain.',
                     'No new ablations were performed: these are natural observations, not causal evidence of behavior at other ordinals.',
                     'Native responses are the first 12 greedily generated tokens; correctness refers to the initial answer only.']}
(OUT / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
np.savez_compressed(OUT / 'vectors.npz', **arrays)
fields = ['ordinal','ordinal_word','native_initial_answer','native_answer_correct','gate','operator_l2','contribution_l2','contribution_cosine_to_11','operator_cosine_to_11']
for table in tables:
    with (OUT / f'{table["domain"]}-comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in table['rows']: writer.writerow({k: row[k] for k in fields})
print(json.dumps({'tables': [{'domain': t['domain'], 'rows': [{k:r[k] for k in fields} for r in t['rows']]} for t in tables]}, indent=2))
