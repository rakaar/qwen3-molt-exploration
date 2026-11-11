"""Audit the single-prompt all-input scalar sweep and report exact output-prefix probabilities."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-12-flask-gate-refine'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def aggregate(v, word):
    scores = [x for x in v['candidates'].values() if x['text'].strip().casefold() == word]
    assert len(scores) == 4
    ids = [tuple(s['token_ids']) for s in scores]
    assert len(set(ids)) == len(ids)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            assert a[:len(b)] != b and b[:len(a)] != a
    return sum(x['word_probability'] for x in scores)

def main():
    r = json.loads((OUT/'results.json').read_text())
    p = json.loads((HERE/'protocol.json').read_text())
    b = json.loads((OUT/'baseline-results.json').read_text())
    assert r['status'] == 'complete' and r['hooks_remaining'] == 0
    assert r['protocol'] == p and r['protocol_sha256'] == sha(HERE/'protocol.json')
    assert r['source_sha256'] == sha(HERE/'run.py') == sha(OUT/'run.py')
    assert r['baseline_lock']['sha256'] == sha(OUT/'baseline-results.json')
    assert len(r['cases']) == len(b['cases']) == 1
    c, bc = r['cases'][0], b['cases'][0]
    assert bc['runs'] == [] and bc['native_correct'] and bc['eligible']
    assert c['case'] == bc['case'] and c['native'] == bc['native'] and c['zero'] == bc['zero']
    assert c['native']['native'] == c['zero']['native'] and c['native']['candidates'] == c['zero']['candidates']
    assert [v['condition'] for v in c['runs']] == p['scopes']
    prior = json.loads((HERE.parents[1]/'results/2026-11-12-flask-binding/results.json').read_text())
    old = next(x for x in prior['cases'] if x['id'] == c['id'])
    assert c['native']['native'] == old['native']['native'] and c['native']['candidates'] == old['native']['candidates']
    prior_sweep = json.loads((HERE.parents[1]/'results/2026-11-12-flask-gate-increase/results.json').read_text())
    reference = next(v for v in prior_sweep['cases'][0]['runs'] if v['condition']=='gate-64')
    repeated = next(v for v in c['runs'] if v['condition']=='gate-64')
    assert repeated['native'] == reference['native'] and repeated['candidates'] == reference['candidates']
    tr = [t for t in c['zero']['trace'] if t['context'] == 'generation' and t['phase'] == 'input']
    n = len(c['case']['input_token_ids'])
    assert len(tr) == n
    before = np.array([t['natural_gate'] for t in tr], dtype=np.float32)
    with np.load(HERE/'weights/L29-T2375.npz', allow_pickle=False) as f:
        w = {k: f[k].astype(np.float32) for k in f.files}
    rows, samples = [], 0
    for v in [c['native'], c['zero']] + c['runs']:
        f = OUT/v['arrays']['file']
        assert sha(f) == v['arrays']['sha256']
        with np.load(f, allow_pickle=False) as a:
            probes = dict(zip(a['first_probe_ids'].tolist(), a['first_probe_logits'].tolist()))
            lse = float(a['first_logsumexp'][0])
            for score in v['candidates'].values():
                assert abs(score['conditional'][0]['log_probability'] - (probes[score['token_ids'][0]] - lse)) < 3e-5
                assert sum(t['log_probability'] for t in score['conditional']) == score['word_log_probability']
                assert math.exp(score['word_log_probability']) == score['word_probability']
                for t in score['conditional']:
                    assert abs(t['log_probability'] - (t['logit'] - t['logsumexp'])) < 3e-5
                k = len(score['token_ids'])
                if v['native']['output_token_ids'][:k] == score['token_ids']:
                    assert math.isclose(math.prod(t['probability'] for t in v['native']['steps'][:k]), score['word_probability'], rel_tol=3e-5, abs_tol=1e-40)
            if v['alpha'] == 1:
                target = v['target_gate']
                assert a['selected_positions'].tolist() == a['audit_positions'].tolist() == list(range(n))
                np.testing.assert_array_equal(a['all_gate_before'], before)
                np.testing.assert_array_equal(a['all_gate_after'], np.full(n, target, dtype=np.float32))
                z = (a['residual_input'] - w['input_mean']) / w['input_std']
                raw = z @ w['e'].reshape(-1) + w['bias'][0]
                gate = np.where((raw > w['theta'][0]) & (raw > 0), raw, 0)
                op = (z @ w['V'] @ w['U']) * w['output_std']
                np.testing.assert_allclose(a['gate_before'], gate, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(a['operator'], op, rtol=5e-4, atol=5e-5)
                np.testing.assert_allclose(a['intended_delta'], (target-a['gate_before'])[:,None]*a['operator'], rtol=1e-6, atol=1e-6)
                bits = (a['native_mlp'] + a['intended_delta']).astype(np.float32).view(np.uint32)
                rounded = ((bits+np.uint32(0x7fff)+((bits>>16)&1)) & np.uint32(0xffff0000)).view(np.float32)
                np.testing.assert_array_equal(a['edited_mlp'], rounded)
                np.testing.assert_array_equal(a['actual_delta'], a['edited_mlp'] - a['native_mlp'])
                np.testing.assert_allclose(a['all_actual_delta_norm'], np.linalg.norm(a['actual_delta'],axis=-1),rtol=1e-5,atol=1e-6)
                samples += n
        edits = [t for t in v['trace'] if t['edited']]
        assert all(t['actual_delta_norm'] == t['intended_delta_norm'] == 0 for t in v['trace'] if not t['edited'])
        if v['alpha'] == 1:
            assert {t['position'] for t in edits} == set(np.flatnonzero(before != v['target_gate']).tolist())
            assert all(t['phase'] == 'input' and t['context'] == 'generation' and t['effective_gate'] == v['target_gate'] for t in edits)
        if v['condition'] == 'zero':
            continue
        response = v['native']['response']
        norm = response.strip().rstrip('.!?').strip().casefold()
        rows.append(dict(condition=v['condition'],target_gate=v['target_gate'],response=response,stop_reason=v['native']['stop_reason'],p_water=aggregate(v,'water'),p_oil=aggregate(v,'oil'),clean_water_answer=norm=='water' and v['native']['stop_reason']=='eos',clean_oil_answer=norm=='oil' and v['native']['stop_reason']=='eos',edited_input_positions=len(edits),generated_positions_edited=0,top5=v['native']['steps'][0]['top5']))
    result = dict(prompt=c['case']['prompt'],formatted_prompt=c['case']['formatted_prompt'],input_tokens=n,native_positive=[dict(position=t['position'],token=t['token'],gate=t['natural_gate']) for t in tr if t['natural_gate']>0],rows=rows,audit=dict(native_zero_exact=True,previous_native_reproduced_exactly=True,previous_gate64_reproduced_exactly=True,source_hash_verified=True,baseline_locked_before_sweep=True,independent_vector_positions=samples,bf16_rounding_exact=True,all_input_positions_verified=True,generated_tokens_edited=0),results_sha256=sha(OUT/'results.json'))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'probabilities.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='top5']);writer.writeheader();writer.writerows([{k:v for k,v in row.items() if k!='top5'} for row in rows])
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
