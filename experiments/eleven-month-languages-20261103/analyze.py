"""Audit natural input activation, separating question text from chat markup."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'comparisons/2026-11-03-eleven-month-languages/remote-export/eleven-month-languages-20261103'
OUT = ROOT / 'results/2026-11-03-eleven-month-languages'
OUT.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


d = json.loads((SOURCE / 'artifacts/results.json').read_text())
assert d['status'] == 'complete' and d['intervention'] is None and len(d['cases']) == 18
assert sha(SOURCE / 'run.py') == d['runtime']['source_sha256']
assert sha(SOURCE / 'protocol.json') == d['runtime']['protocol_sha256']
wp = ROOT / 'experiments/eleven-activation-20261102/weights/L29-T2375.npz'
assert sha(wp) == d['runtime']['feature_sha256']
with np.load(wp, allow_pickle=False) as a:
    w = {k: a[k].copy() for k in a.files}
max_encoder_error, max_operator_error, npositions = 0., 0., 0
for r in d['cases']:
    assert r['observer_full_logits_exact']
    path = SOURCE / 'artifacts' / r['vectors']['file']
    assert sha(path) == r['vectors']['sha256']
    with np.load(path, allow_pickle=False) as a:
        z = (a['residual'] - w['input_mean']) / w['input_std']
        assert np.allclose(z, a['z'], atol=1e-6, rtol=1e-6)
        raw = z @ w['e'].reshape(-1) + w['bias'].reshape(-1)[0]
        raw_error = float(np.max(np.abs(raw - a['raw'])))
        max_encoder_error = max(max_encoder_error, raw_error)
        assert raw_error < 1e-5
        gate = np.where((raw > w['theta'].reshape(-1)[0]) & (raw > 0), raw, 0.)
        assert np.allclose(gate, a['gate'], atol=1e-5, rtol=1e-5)
        op = (z @ w['V'] @ w['U']) * w['output_std']
        err = float(np.linalg.norm((op-a['operator']).astype(np.float64)) / np.linalg.norm(a['operator'].astype(np.float64)))
        assert err < 2e-6
        max_operator_error = max(max_operator_error, err)
        assert np.array_equal(a['contribution'], a['gate'][:, None] * a['operator'])
        assert [t['gate'] for t in r['trace']] == a['gate'].astype(float).tolist()
        npositions += len(a['gate'])
    native = r['native']
    assert len(r['trace']) == len(native['prompt_token_ids']) == len(native['token_offsets'])
    rendered = native['rendered_input']
    start, end = rendered.index(r['prompt']), rendered.index(r['prompt']) + len(r['prompt'])
    for t, (a,b) in zip(r['trace'], native['token_offsets']):
        expected_role = 'raw_input' if r['format'] == 'raw' else ('user_question' if a < end and b > start else 'chat_template')
        assert t['role'] == expected_role
    r['any_input_gate'] = max(t['gate'] for t in r['trace'])

# Reproduce previously observed Spanish/Chinese natural activation, without relying on causal outcomes.
anchors = []
for study,folder in [('raw','eleven-multilingual-20261101'), ('chat','eleven-multilingual-chat-20261101')]:
    p = ROOT / 'comparisons/2026-11-01-eleven-multilingual/remote-export' / folder / 'artifacts/controls.json'
    old = json.loads(p.read_text())
    for lang in ['es','zh']:
        cid = f'{lang}_month_1' if study == 'raw' else f'{lang}_month_chat'
        saved = next(r for r in old['controls'] if r['case_id'] == cid)['ordinals']['11']
        new = next(r for r in d['cases'] if r['id'] == f'{lang}-{study}-11')
        assert new['native']['rendered_input'] == saved['native']['chat']
        n = min(len(new['native']['output_token_ids']),len(saved['native']['output_token_ids']))
        assert new['native']['output_token_ids'][:n] == saved['native']['output_token_ids'][:n]
        old_input = [t for t in saved['zero_strength']['trace'] if t['position'] < len(saved['native']['prompt_token_ids'])]
        assert len(old_input) == len(new['trace'])
        assert all(a['natural_gate'] == b['gate'] for a,b in zip(old_input,new['trace']))
        anchors.append({'id':new['id'],'input_gates_reproduced_exactly':True,'source':str(p.relative_to(ROOT)),'source_sha256':sha(p)})

historical=[]
old_path=ROOT/'comparisons/2026-11-01-eleven-multilingual/remote-export/eleven-multilingual-20261101/artifacts/controls.json'
old=json.loads(old_path.read_text())
for cid in ['es_month_2','zh_month_2']:
    saved=next(r for r in old['controls'] if r['case_id']==cid)['ordinals']['11']
    trace=[t for t in saved['zero_strength']['trace'] if t['position']<len(saved['native']['prompt_token_ids'])]
    historical.append({'case_id':cid,'prompt':saved['native']['prompt'],'max_input_gate':max(t['natural_gate'] for t in trace),
                       'response':saved['native']['answer'],'source_sha256':sha(old_path)})
result={'feature':d['feature'],'date_local':'2026-11-03','intervention':None,
        'cases':d['cases'],'historical_zero_gate_variants':historical,'reproduced_anchors':anchors,
        'runtime':d['runtime'],'limits':d['protocol']['limits'],
        'source_results_sha256':sha(SOURCE/'artifacts/results.json'),
        'definitions':{'question_max_gate':'Maximum natural gate on actual user-question text, or the whole raw prefix.',
                       'last_input_gate':'Natural gate at final processed input token; this belongs to the assistant chat prefix for chat runs.',
                       'any_input_gate':'Maximum gate over question text plus any chat-template tokens.'}}
(OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
fields=['id','language','format','ordinal','prompt','question_max_gate','last_input_gate','any_input_gate','response']
with (OUT/'activation.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in result['cases']:
        writer.writerow({**{k:r[k] for k in ['id','language','format','ordinal','prompt','question_max_gate','any_input_gate']},
                         'last_input_gate':r['last_input']['gate'],'response':r['native']['response']})
validation={'status':'pass','cases':len(d['cases']),'input_positions':npositions,
            'no_intervention':True,'observer_logits_exact':True,'previous_anchor_gates_exact':len(anchors),
            'maximum_encoder_error':max_encoder_error,'maximum_operator_relative_l2_error':max_operator_error,
            'source_and_vector_hashes_checked':True,'question_template_roles_checked':True}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
for r in result['cases']:
    print(json.dumps({'id':r['id'],'question_max':r['question_max_gate'],'last_input_gate':r['last_input']['gate'],
                      'active_question_tokens':[(t['token'],t['gate']) for t in r['question_active_tokens']],
                      'response':r['native']['response']},ensure_ascii=False))
print(json.dumps(validation))
