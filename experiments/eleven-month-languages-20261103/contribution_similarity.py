"""Pairwise multilingual native MOLT geometry, with raw/chat formats separate."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'comparisons/2026-11-03-eleven-month-languages/remote-export/eleven-month-languages-20261103'
OUT = ROOT / 'results/2026-11-03-eleven-month-language-similarity'
OUT.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


data_path = SOURCE / 'artifacts/results.json'
d = json.loads(data_path.read_text())
assert d['status'] == 'complete' and d['intervention'] is None
assert sha(SOURCE / 'run.py') == d['runtime']['source_sha256']
audit = ROOT / 'results/2026-11-03-eleven-month-languages/validation.json'
assert json.loads(audit.read_text())['status'] == 'pass'
summary = json.loads((audit.parent / 'summary.json').read_text())
assert summary['source_results_sha256'] == sha(data_path)
old_base = ROOT / 'comparisons/2026-11-01-eleven-multilingual/remote-export/eleven-multilingual-chat-20261101/artifacts'
old_path = old_base / 'controls.json'
old = json.loads(old_path.read_text())
assert old['protocol']['base_revision'] == d['runtime']['base_revision']
assert old['protocol']['molt_revision'] == d['runtime']['molt_revision']
assert old['protocol']['enable_thinking'] is False
assert old['protocol']['runtime']['base_dtype'] == 'BF16'
languages = [('en','English'), ('es','Spanish'), ('zh','Chinese'), ('fr','French'), ('hi','Hindi')]
records, matrices, arrays = {}, {}, {}
for fmt in ['raw','chat']:
    rows, vectors = [], []
    for code, language in languages:
        if fmt == 'chat' and code == 'en':
            control = next(r for r in old['controls'] if r['case_id'] == 'en_month_chat')['ordinals']['11']
            native, zero = control['native'], control['zero_strength']
            assert native['output_token_ids'] == zero['output_token_ids'] and native['steps'] == zero['steps']
            pos = len(native['prompt_token_ids']) - 1
            v = next(v for v in zero['vectors'] if '--step-0.' in v['file'])
            path = old_base / v['file']
            assert sha(path) == v['sha256']
            with np.load(path, allow_pickle=False) as a:
                j = list(a['positions']).index(pos)
                gate = float(a['gate'][0,j])
                operator = a['operator'][0,j].copy()
            trace = [{'position':t['position'],'token':t['token'],'gate':t['natural_gate']}
                     for t in zero['trace'] if t['position'] < len(native['prompt_token_ids'])]
            assert ''.join(t['token'] for t in trace) == native['chat']
            offsets, cursor = [], 0
            for t in trace:
                offsets.append((cursor, cursor + len(t['token'])))
                cursor += len(t['token'])
            prompt_ids = native['prompt_token_ids']
            token_id = native['prompt_token_ids'][pos]
            token_context_span = trace[-1]['token']
            row = {'id':'en-chat-11','prompt':native['prompt'],'rendered_input':native['chat'],
                   'response':native['answer'],'observer_full_logits_exact':True,
                   'source_results':str(old_path.relative_to(ROOT)),'source_results_sha256':sha(old_path),
                   'native_answer_note':'Correct November answer in the saved no-thinking chat run.'}
        else:
            r = next(r for r in d['cases'] if r['id'] == f'{code}-{fmt}-11')
            assert r['observer_full_logits_exact']
            pos = len(r['native']['prompt_token_ids']) - 1
            path = SOURCE / 'artifacts' / r['vectors']['file']
            assert sha(path) == r['vectors']['sha256']
            with np.load(path, allow_pickle=False) as a:
                gate = float(a['gate'][pos])
                operator = a['operator'][pos].copy()
                assert np.array_equal(a['contribution'][pos], np.float32(gate) * operator)
            trace = [{'position':t['position'],'token':t['token'],'gate':t['gate']} for t in r['trace']]
            offsets = r['native']['token_offsets']
            prompt_ids = r['native']['prompt_token_ids']
            token_id = r['native']['prompt_token_ids'][pos]
            start, end = r['native']['token_offsets'][pos]
            token_context_span = r['native']['rendered_input'][start:end]
            note = 'Correct month label; raw Chinese first asks which month, then states November.' if code == 'zh' and fmt == 'raw' else 'Correct November month label.'
            if code == 'fr' and fmt == 'raw':
                note = 'Native completion incorrectly says December. Retained as an inactive prompt, not treated as a successful capability control.'
            if code == 'hi' and fmt == 'raw':
                note = 'Native 24-token continuation does not directly answer the month. Retained as an inactive prompt, not treated as a successful capability control.'
            row = {'id':r['id'],'prompt':r['prompt'],'rendered_input':r['native']['rendered_input'],
                   'response':r['native']['response'],'observer_full_logits_exact':True,
                   'source_results':str(data_path.relative_to(ROOT)),'source_results_sha256':sha(data_path),
                   'native_answer_note':note}
        question_start = row['rendered_input'].index(row['prompt'])
        question_end = question_start + len(row['prompt'])
        assert len(trace) == len(offsets) == len(prompt_ids)
        for t, (start, end), tid in zip(trace, offsets, prompt_ids):
            t['token_id'] = tid
            t['source_span'] = row['rendered_input'][start:end]
            t['region'] = ('completion' if fmt == 'raw' else
                           'question' if start < question_end and end > question_start else
                           'chat_header' if end <= question_start else 'chat_suffix')
        assert any(t['region'] == ('completion' if fmt == 'raw' else 'question') for t in trace)
        c = np.float32(gate) * operator
        norm = float(np.linalg.norm(c.astype(np.float64)))
        assert trace[-1]['position'] == pos and trace[-1]['gate'] == gate
        assert norm > 0 or all(t['gate'] == 0 for t in trace), 'Do not silently ignore an earlier active position'
        row.update(language=language,code=code.upper(),format=fmt,position=pos,token=trace[-1]['token'],
                   token_id=token_id,token_context_span=token_context_span,gate=gate,
                   operator_l2=float(np.linalg.norm(operator.astype(np.float64))),contribution_l2=norm,
                   active=norm>0,trace=trace,active_positions=[t for t in trace if t['gate']>0],
                   source_vectors=str(path.relative_to(ROOT)),source_vectors_sha256=sha(path))
        rows.append(row)
        vectors.append(c.astype(np.float64))
    values = np.stack(vectors)
    norms = np.linalg.norm(values,axis=1)
    cos = [[float(values[i] @ values[j] / (norms[i]*norms[j])) if norms[i]>0 and norms[j]>0 else None
            for j in range(5)] for i in range(5)]
    for i in range(5):
        if norms[i]>0: cos[i][i]=1.
        for j in range(5):
            assert cos[i][j] == cos[j][i] or abs(cos[i][j]-cos[j][i])<1e-12
    assert sum(r['active'] for r in rows) == (3 if fmt == 'raw' else 5)
    records[fmt], matrices[fmt] = rows, cos
    arrays[fmt+'_contributions'] = values
    arrays[fmt+'_cosines'] = np.array([[np.nan if x is None else x for x in row] for row in cos])
    with (OUT / f'{fmt}-cosines.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['language',*[n for _,n in languages]])
        for r,c in zip(rows,cos):writer.writerow([r['language'],*['undefined' if x is None else x for x in c]])
result={'feature':'L29-T2375','date_local':'2026-11-03','languages':[n for _,n in languages],
        'contribution_definition':'c = gate(z) * ((z @ V @ U) * output_std); z = (native residual - input_mean) / input_std.',
        'position_rule':'Final input position for every language; inspectable token and zero-based position per prompt.',
        'space':'Original positive full MOLT contribution in native MLP-output units; no final RMSNorm, unembedding or BF16 subtraction rounding.',
        'intervention':None,'new_inference':False,'records':records,'cosines':matrices,
        'source_audit_sha256':sha(audit),
        'limits':['The two formats use different wording as well as a chat template; format and wording effects are not isolated.',
                  'Raw French and Hindi have no active input token and do not supply valid eleventh-month answers in these continuations.',
                  'No causal intervention, cross-language patching or test of a language-independent ordinal mechanism is performed.',
                  'Cosine of a zero contribution is undefined, including its diagonal; no language-specific token search was performed.']}
offdiag=[matrices['chat'][i][j] for i in range(5) for j in range(i)]
result['chat_pairwise_range']=[min(offdiag),max(offdiag)]
(OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
np.savez_compressed(OUT/'contributions.npz',languages=np.array(result['languages']),**arrays)
with (OUT/'magnitudes.csv').open('w',newline='') as f:
    fields=['format','language','position','token','gate','operator_l2','contribution_l2','active']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for fmt in ['raw','chat']:
        for r in records[fmt]:writer.writerow({k:r[k] for k in fields})
print(json.dumps({'cosines':matrices,'chat_pairwise_range':result['chat_pairwise_range'],
                  'magnitudes':{fmt:[{k:r[k] for k in ['language','position','token','gate','contribution_l2']} for r in records[fmt]] for fmt in ['raw','chat']}},indent=2,ensure_ascii=False))
