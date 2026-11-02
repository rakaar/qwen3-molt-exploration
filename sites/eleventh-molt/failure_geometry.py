"""All eligible, active final-input contribution comparisons from the saved study."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_geometry(root, out):
    summary_path = root / 'results/2026-11-01-eleven-multilingual/summary.json'
    summary = json.loads(summary_path.read_text())
    records, inactive, vectors, sources = [], [], [], []
    codes = {'en_month_1': 'EN-M', 'en_alphabet_1': 'EN-A', 'en_roman_1': 'EN-R',
             'es_month_1': 'ES-M', 'es_month_2': 'ES-M2', 'es_alphabet_1': 'ES-A',
             'es_alphabet_2': 'ES-A2', 'es_roman_2': 'ES-R', 'zh_month_1': 'ZH-M'}
    domains = {'month': 'month', 'alphabet': 'alphabet', 'roman': 'Roman numerals'}
    for study, folder in [('raw', 'eleven-multilingual-20261101'), ('chat', 'eleven-multilingual-chat-20261101')]:
        base = root / 'comparisons/2026-11-01-eleven-multilingual/remote-export' / folder
        cp, rp = base / 'artifacts/controls.json', base / 'artifacts/results.json'
        controls = json.loads(cp.read_text())['controls']
        runs = json.loads(rp.read_text())['runs']
        sources += [{'path': str(p.relative_to(root)), 'sha256': digest(p)} for p in [cp, rp]]
        for row in summary['cases']:
            if row['study'] != study or not row['eligible']:
                continue
            cid = row['case_id']
            control = next(c for c in controls if c['case_id'] == cid)['ordinals']
            zero, native = control['11']['zero_strength'], control['11']['native']
            off = next(r for r in runs if r['case_id'] == cid and r['name'] == 'ordinal-11--off-last_input_only')
            assert native['output_token_ids'] == zero['output_token_ids'] and native['steps'] == zero['steps']
            pos = len(native['prompt_token_ids']) - 1
            assert off['spec'] == {'kind': 'positions', 'positions': [pos]} and off['target'] == 0 and off['alpha'] == 1
            assert all(t['actual_delta_norm'] == 0 for t in off['trace'] if t['phase'] == 'output')
            vector = next(v for v in zero['vectors'] if '--step-0.' in v['file'])
            vp = base / 'artifacts' / vector['file']
            assert digest(vp) == vector['sha256']
            with np.load(vp, allow_pickle=False) as a:
                j = list(a['positions']).index(pos)
                gate = float(a['gate'][0, j])
                operator = a['operator'][0, j].copy()
                contribution = np.float32(gate) * operator
            off_vector = next(v for v in off['vectors'] if '--step-0.' in v['file'])
            op = base / 'artifacts' / off_vector['file']
            assert digest(op) == off_vector['sha256']
            with np.load(op, allow_pickle=False) as a:
                j = list(a['positions']).index(pos)
                assert np.array_equal(contribution, -a['intended_delta'][0, j])
            norm = float(np.linalg.norm(contribution.astype(np.float64)))
            ordinal = row['last_input_only_answer_ordinal']
            outcome = 'success' if ordinal == 12 else ('unchanged' if ordinal == 11 else 'off_target')
            code = codes.get(cid, cid.split('_')[0].upper() + '-' + {'month': 'M', 'alphabet': 'A', 'roman': 'R'}[row['domain']] + 'c')
            record = {'id': cid, 'code': code, 'study': study, 'language': row['language'],
                      'domain': domains[row['domain']], 'prompt': native['prompt'], 'rendered_input': native['chat'],
                      'position': pos, 'token': next(t['token'] for t in zero['trace'] if t['position'] == pos),
                      'gate': gate, 'operator_l2': float(np.linalg.norm(operator.astype(np.float64))),
                      'contribution_l2': norm, 'outcome': outcome, 'answer_ordinal': ordinal,
                      'native_response': native['answer'], 'off_response': off['answer'],
                      'native_twelfth_response': control['12']['native']['answer'],
                      'capability_note': row['capability_review_note'],
                      'other_scope_success': any(row.get(s + '_answer_ordinal') == 12 for s in ['input_only', 'all_positions', 'output_only']),
                      'source_vectors': str(vp.relative_to(root)), 'source_vectors_sha256': vector['sha256'],
                      'source_off_vectors_sha256': off_vector['sha256']}
            if norm == 0:
                assert gate == 0
                record['outcome'] = 'inactive'
                inactive.append(record)
            else:
                records.append(record)
                vectors.append(contribution.astype(np.float64))
    values = np.stack(vectors)
    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    cosine = unit @ unit.T
    np.fill_diagonal(cosine, 1.)
    anchors = [i for i, r in enumerate(records) if r['id'] in ['en_month_1', 'en_alphabet_1', 'en_roman_1']]
    assert len(anchors) == 3
    for i, r in enumerate(records):
        r['cosines_to_english_successes'] = {records[j]['code']: float(cosine[i, j]) for j in anchors}
        r['maximum_cosine_to_english_success'] = max(r['cosines_to_english_successes'].values())
    raw_ids = [i for i, r in enumerate(records) if r['study'] == 'raw']
    assert len(raw_ids) == 7 and len(records) == 16 and len(inactive) == 2
    assert sum(r['outcome'] == 'success' for r in records) == 4
    assert sum(r['outcome'] == 'unchanged' for r in records) == 11
    assert sum(r['outcome'] == 'off_target' for r in records) == 1
    for filename, indices in [('raw-outcome-cosines.csv', raw_ids), ('all-active-outcome-cosines.csv', list(range(len(records))))]:
        with (out / filename).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['prompt', *[records[j]['code'] for j in indices]])
            for i in indices: writer.writerow([records[i]['code'], *[float(cosine[i, j]) for j in indices]])
    result = {'scope': 'last_input_only', 'records': records, 'inactive': inactive,
              'cosines': cosine.tolist(), 'raw_indices': raw_ids, 'sources': sources,
              'source_summary_sha256': digest(summary_path), 'excluded_raw_capability_pairs': 9,
              'interpretation': 'High cosine is present in active no-shift cases too; low similarity does not separate the observed outcomes in this selected set.'}
    (out / 'outcome-geometry.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(out / 'outcome-contributions.npz', ids=np.array([r['id'] for r in records]), contributions=values, cosines=cosine)
    return result
