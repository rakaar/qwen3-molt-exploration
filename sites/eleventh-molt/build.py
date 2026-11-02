"""Build a local, self-contained report from hash-bound saved experiment records."""
import hashlib
import json
from pathlib import Path
import shutil
from failure_geometry import build_geometry
from sweep_data import build_sweeps
from t1181_data import build_t1181

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'dist'
OUT.mkdir(exist_ok=True)
controls_path = ROOT / 'comparisons/2026-11-01-eleven-multilingual/remote-export/eleven-multilingual-20261101/artifacts/controls.json'
controls = json.loads(controls_path.read_text())['controls']
specs = [
    ('Month', 'en_month_1', 'November → December', 'eleventh month → twelfth month', 1,
     'comparisons/2026-11-02-eleven-contribution-readout/remote-export/eleven-contribution-readout-20261102'),
    ('Alphabet', 'en_alphabet_1', 'K → L', 'eleventh letter → twelfth letter', 1,
     'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/alphabet'),
    ('Roman numerals', 'en_roman_1', 'XI → XII', 'eleventh integer → twelfth integer', 2,
     'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/roman'),
]
cases = []
for label, cid, change, caption, off_count, relative in specs:
    path = ROOT / relative / 'reference.json'
    ref = json.loads(path.read_text())
    native, off = ref['native'], ref['gate_off']
    row = next(x for x in controls if x['case_id'] == cid)
    zero = row['ordinals']['11']['zero_strength']
    assert native['output_token_ids'] == zero['output_token_ids']
    assert native['prompt_token_ids'] == off['prompt_token_ids']
    selected = [x for x in off['trace'] if x['selected']]
    assert len(selected) == 1 and selected[0]['token'] == ' is'
    assert all(x['actual_delta_norm'] == 0 for x in off['trace'] if x['phase'] == 'output')
    def condition(run, trace, answer_count, edited=False):
        indexed = {x['position']: x for x in trace}
        n = len(run['prompt_token_ids'])
        def token(position, token_id, text):
            t = indexed.get(position)
            is_edit = edited and t is not None and t.get('selected', False)
            natural = t['natural_gate'] if t else None
            return {'position': position, 'token_id': token_id, 'text': text,
                    'natural_gate': natural, 'gate': 0. if is_edit else natural, 'edited': is_edit}
        inputs = [token(i, tid, indexed[i]['token']) for i, tid in enumerate(run['prompt_token_ids'])]
        output = [token(n + i, s['token_id'], s['token']) for i, s in enumerate(run['steps'])]
        p = 1.
        for step in run['steps'][:answer_count]: p *= step['selected_probability']
        answer = ''.join(s['token'] for s in run['steps'][:answer_count])
        return {'input': inputs, 'output': output, 'answer_tokens': output[:answer_count],
                'answer': answer, 'probability': p, 'probability_factors': [s['selected_probability'] for s in run['steps'][:answer_count]],
                'full_response': run['answer'], 'stop_reason': run['stop_reason']}
    cases.append({'name': label, 'id': cid, 'change': change, 'caption': caption, 'prompt': native['prompt'],
                  'position': selected[0]['position'], 'natural_gate': selected[0]['natural_gate'],
                  'native': condition(native, zero['trace'], 1),
                  'off': condition(off, off['trace'], off_count, True),
                  'native_twelfth_response': row['ordinals']['12']['native']['answer'],
                  'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
geometry_path = ROOT / 'results/2026-11-03-eleven-contribution-similarity/summary.json'
geometry = json.loads(geometry_path.read_text())
expanded_geometry = build_geometry(ROOT, OUT)
ordinal_path = ROOT / 'results/2026-11-03-eleven-ordinal-specificity/summary.json'
ordinals = json.loads(ordinal_path.read_text())
assert all(t['anchor_reproduced_exactly'] for t in ordinals['tables'])
assert all(r['observer_full_logits_exact'] for t in ordinals['tables'] for r in t['rows'])
shutil.copy2(ordinal_path, OUT / 'ordinal-specificity.json')
for name in ['month-comparison.csv', 'roman-comparison.csv', 'vectors.npz']:
    shutil.copy2(ordinal_path.parent / name, OUT / ('ordinal-' + name))
language_path = ROOT / 'results/2026-11-03-eleven-month-language-similarity/summary.json'
languages = json.loads(language_path.read_text())
assert languages['new_inference'] is False and languages['intervention'] is None
assert all(r['observer_full_logits_exact'] for rows in languages['records'].values() for r in rows)
shutil.copy2(language_path, OUT / 'language-similarity.json')
for name in ['raw-cosines.csv', 'chat-cosines.csv', 'magnitudes.csv', 'contributions.npz']:
    shutil.copy2(language_path.parent / name, OUT / ('language-' + name))
sweeps = build_sweeps(ROOT, OUT)
t1181 = build_t1181(ROOT, OUT)
data = {'cases': cases, 'geometry': geometry, 'expanded_geometry': expanded_geometry,
        'ordinals': ordinals, 'languages': languages, 'sweeps': sweeps, 't1181': t1181,
        'color_max': max(t['gate'] or 0 for c in cases for cond in ['native', 'off'] for t in c[cond]['input'] + c[cond]['output']),
        'model': 'Qwen3-4B', 'layer': 29, 'transform': 2375,
        'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
        'molt_revision': '081ce901b090b64aeecf9bc8709b57c10411e32a',
        'source_controls_sha256': hashlib.sha256(controls_path.read_bytes()).hexdigest(),
        'geometry_source_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest()}
(OUT / 'data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
template = (HERE / 'index.template.html').read_text()
assert template.count('__TRANSPLANTS__') == 1
template = template.replace('__TRANSPLANTS__', (HERE / 'transplants.section.html').read_text())
for marker, filename in [('__TRANSPLANT_STYLE__', 'transplants.css'), ('__TRANSPLANT_SCRIPT__', 'transplants.js')]:
    assert template.count(marker) == 1
    template = template.replace(marker, (HERE / filename).read_text())
for marker, filename in [('__T1181_SECTION__', 't1181.section.html'), ('__T1181_STYLE__', 't1181.css'), ('__T1181_SCRIPT__', 't1181.js')]:
    assert template.count(marker) == 1
    template = template.replace(marker, (HERE / filename).read_text())
assert template.count('__STUDY_DATA__') == 1
(OUT / 'index.html').write_text(template.replace('__STUDY_DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))
for name in ['cosine-similarity.csv', 'magnitudes.csv']:
    shutil.copy2(ROOT / 'results/2026-11-03-eleven-contribution-similarity' / name, OUT / name)
print(json.dumps({'site': str(OUT / 'index.html'), 'probabilities': {c['name']: [c['native']['probability'], c['off']['probability']] for c in cases}}))
