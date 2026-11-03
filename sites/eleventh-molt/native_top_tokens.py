"""Recover native next-token rankings from audited, saved full-vocabulary logits."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_native_top_tokens(root, out):
    sources = [
        ('en_month_1', 'Month', 'comparisons/2026-11-02-eleven-contribution-readout/remote-export/eleven-contribution-readout-20261102'),
        ('en_alphabet_1', 'Alphabet', 'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/alphabet'),
        ('en_roman_1', 'Roman numerals', 'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/roman'),
    ]
    cases = []
    for case_id, name, relative in sources:
        folder = root / relative
        reference_path = folder / 'reference.json'
        readout_path = folder / 'artifacts/results.json'
        arrays_path = folder / 'artifacts/readout-arrays.npz'
        reference = json.loads(reference_path.read_text())
        readout = json.loads(readout_path.read_text())
        native = reference['native']
        assert native['spec'] is None, 'A native ranking must not contain an intervention'
        assert native['chat'] == native['prompt'], 'The rendered input must be the raw prompt'
        assert readout['native_mlp_reproduced_exactly']
        assert readout['native_final_norm_reconstructed_exactly']
        assert sha(arrays_path) == readout['arrays_sha256']
        with np.load(arrays_path, allow_pickle=False) as arrays:
            logits = arrays['actual_native_final_logits'].astype(np.float64)
        assert logits.shape == (151936,) and np.isfinite(logits).all()
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        step = native['steps'][0]
        assert int(logits.argmax()) == step['token_id']
        assert float(logits[step['token_id']]) == step['selected_logit']
        decoded = {row['token_id']: row['token'] for row in step['top_tokens']}
        max_error = max(abs(probabilities[row['token_id']] - row['probability'])
                        for row in step['top_tokens'])
        assert max_error < 1e-6, 'Saved first-step probabilities must reproduce'
        # Equal logits share a rank; token ID provides deterministic display order.
        selected = np.lexsort((np.arange(logits.size), -logits))[:5]
        rows = []
        for token_id in selected:
            token_id = int(token_id)
            assert token_id in decoded, 'Use the recorded decoder output for every listed token'
            rows.append({'rank': int((logits > logits[token_id]).sum()) + 1,
                         'token_id': token_id, 'token': decoded[token_id],
                         'probability': float(probabilities[token_id]),
                         'logit': float(logits[token_id])})
        cases.append({'case_id': case_id, 'name': name, 'prompt': native['prompt'],
                      'input_token_ids': native['prompt_token_ids'],
                      'native_initial_token': step['token'], 'tokens': rows,
                      'remaining_probability': 1 - sum(row['probability'] for row in rows),
                      'full_vocabulary_size': int(logits.size),
                      'saved_probability_max_absolute_error': float(max_error),
                      'sources': {str(path.relative_to(root)): sha(path)
                                  for path in [reference_path, readout_path, arrays_path]}})
    result = {
        'model': 'Qwen3-4B', 'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
        'condition': 'Native model, before any intervention; raw completion prefix without chat formatting.',
        'method': 'FP64 softmax of the saved native final-input full-vocabulary logits. No temperature scaling or top-k renormalization.',
        'ranking': 'Descending logit; equal logits share a rank and are displayed by ascending token ID.',
        'probability_scope': 'Exact next token, including leading whitespace and case. No grouping of answer variants.',
        'roman_note': 'The token " x" is a prefix, not a complete XII answer. XII uses two tokens in the recorded ablated response.',
        'new_inference': False, 'new_gpu_spend_usd': 0, 'cases': cases,
    }
    (out / 'native-top-five.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    with (out / 'native-top-five.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'prompt', 'rank', 'token_id', 'token', 'probability', 'percent', 'logit'])
        for case in cases:
            for row in case['tokens']:
                writer.writerow([case['case_id'], case['prompt'], row['rank'], row['token_id'],
                                 row['token'], row['probability'], 100 * row['probability'], row['logit']])
    return result


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    summary = build_native_top_tokens(here.parents[1], here / 'dist')
    print(json.dumps({'new_inference': False, 'new_gpu_spend_usd': 0,
                      'cases': [{k: case[k] for k in ['name', 'prompt', 'tokens']}
                                for case in summary['cases']]}, ensure_ascii=False, indent=2))
