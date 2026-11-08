"""Freeze a small observational test of the shared-final-token confound."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDINALS = {10: 'tenth', 11: 'eleventh', 12: 'twelfth'}
ANSWERS = {
    'month': {10: 'October', 11: 'November', 12: 'December'},
    'alphabet': {10: 'J', 11: 'K', 12: 'L'},
    'roman': {10: 'X', 11: 'XI', 12: 'XII'},
}
TEMPLATES = {
    'original': {
        'month': 'the {ordinal} month of the year is',
        'alphabet': 'the {ordinal} letter of the english alphabet is',
        'roman': 'the {ordinal} positive integer in roman numerals is',
    },
    'word_rewrite': {
        'month': 'the {ordinal} month of the year is named',
        'alphabet': 'the {ordinal} position in the english alphabet contains the letter',
        'roman': 'the {ordinal} positive integer in roman numerals is written',
    },
    'digit_rewrite': {
        'month': 'month number {number} in the calendar has the name',
        'alphabet': 'for position {number} in the english alphabet, give the symbol',
        'roman': 'integer {number} expressed using roman numerals has the representation',
    },
}

cases = []
for variant, domains in TEMPLATES.items():
    for domain, template in domains.items():
        for number, ordinal in ORDINALS.items():
            cases.append({
                'id': f'{variant}-{domain}-{number}', 'variant': variant,
                'domain': domain, 'ordinal': number, 'format': 'raw',
                'prompt': template.format(ordinal=ordinal, number=number),
                'expected_answer': ANSWERS[domain][number],
            })
protocol = {
    'date_local': '2026-11-08', 'feature': 'L29-T2375',
    'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
    'molt_revision': '081ce901b090b64aeecf9bc8709b57c10411e32a',
    'intervention': None, 'maximum_runtime_minutes': 15, 'max_new_tokens': 32,
    'cases': cases,
    'design': [
        'Three domains, three fixed phrasings, and matched tenth/eleventh/twelfth questions: 27 prompts.',
        'Raw completion prefixes avoid a shared assistant-template suffix.',
        'Word rewrite terminal words: named / letter / written. Digit rewrite: name / symbol / representation.',
        'Verify terminal token IDs are pairwise different within each rewrite triplet before model inference.',
        'Observe every input position; retain exact tokens, IDs, raw gate scores, zero gates and native completions.',
        'Primary fixed-position comparison: final input tokens. Also report every cross-domain active-token pair, separating equal and unequal token IDs.',
        'Never choose a token or rewrite according to its cosine. Report peak-gate positions only as a separately labelled activation summary.',
        'Compare full MOLT contribution, ungated operator output, and native residual separately. Zero contributions have undefined cosine.',
        'Compare tenth/twelfth controls at the same wording and token positions to assess ordinal specificity.',
        'Retain baseline answer failures; label capability separately from natural activation and geometry.',
        'Returning-None observer must preserve every generated token and next-token logit hash.',
        'No gate ablation, full-MLP replacement, final RMSNorm or vocabulary projection in vector comparisons.',
    ],
}
out = HERE / 'protocol.json'
text = json.dumps(protocol, indent=2, ensure_ascii=False) + '\n'
if out.exists():
    assert out.read_text() == text, 'Do not overwrite a different protocol'
else:
    out.write_text(text)
print(json.dumps({'cases': len(cases), 'path': str(out),
                  'sha256': hashlib.sha256(text.encode()).hexdigest()}))
