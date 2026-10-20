"""Export L28-T1123 input gates from the frozen all-layer scan; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Original experiment tree, including local activation/weight caches')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root / 'experiments/ch-alllayer-causal-screen'
    capture_path = source / 'capture/measurements.json'
    capture = json.loads(capture_path.read_text())
    manifest_path = source / 'gates/manifest.json'
    asset = next(a for a in json.loads(manifest_path.read_text())['layers'] if a['layer'] == 28)
    weight_path = source / 'gates' / asset['file']
    assert sha(weight_path) == asset['sha256']
    with np.load(weight_path) as arrays:
        weights = {k: arrays[k] for k in arrays.files}
    theta = float(weights['threshold'].reshape(-1)[1123])
    records, inputs = [], {}
    max_error = 0.0
    assert capture['status'] == 'complete'
    for case in capture['rows']:
        cache_path = source / 'capture' / case['input_file']
        assert sha(cache_path) == case['input_sha256']
        gate_path = source / 'survey' / (case['id'] + '-gates.npz')
        with np.load(cache_path) as arrays:
            native = (arrays['input_bf16'][28].astype(np.uint32) << 16).view(np.float32)
        standardized = (native - weights['mean']) / weights['std']
        independent_raw = standardized @ weights['encoder'][1123] + weights['bias'][1123]
        with np.load(gate_path) as arrays:
            saved_gates = arrays['gates'][28, :, 1123]
        task = 'uppercase' if case['family'] == 'fewshot_upper' else 'lowercase'
        tokens = ['child', 'chin', 'chair', 'chief']
        token_texts = [case['target']] if case['target'] in tokens else ['che', 'ef']
        assert len(token_texts) == len(case['target_positions'])
        for position, token in zip(case['target_positions'], token_texts):
            raw = float(independent_raw[position])
            expected_gate = raw if raw > max(0.0, theta) else 0.0
            saved_gate = float(saved_gates[position])
            error = abs(expected_gate - saved_gate)
            assert error < 3e-5
            max_error = max(max_error, error)
            records.append({'case_id': case['id'], 'task': task, 'word': case['target'],
                            'segment': 'prompt', 'position': position,
                            'token_id': case['prompt_token_ids'][position], 'token': token,
                            'target_token': True, 'feature': 'L28-T1123', 'layer': 28,
                            'raw': raw, 'theta': theta, 'gate': saved_gate,
                            'active': saved_gate > 0})
        inputs[case['id']] = {'capture_file': str(cache_path.relative_to(args.source_root)),
                              'capture_sha256': sha(cache_path),
                              'survey_file': str(gate_path.relative_to(args.source_root)),
                              'survey_sha256': sha(gate_path),
                              'prompt_token_ids': case['prompt_token_ids'],
                              'target_positions': case['target_positions']}
    assert len(records) == 12
    output = {'feature': 'L28-T1123', 'label': 'Ch-prefix food and noun tokens',
              'scope': 'Natural gates at input word subtokens in ten original few-shot prompts',
              'records': records,
              'provenance': {'no_new_inference': True,
                             'capture_measurements_sha256': sha(capture_path),
                             'gate_manifest_sha256': sha(manifest_path),
                             'gate_weights_sha256': sha(weight_path),
                             'independent_cpu_gate_max_abs_error': max_error,
                             'inputs': inputs,
                             'extractor': 'experiments/ch-absorption-overview/extract_l28_gates.py'}}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'exported_gates': len(records), 'cpu_max_abs_error': max_error}))


if __name__ == '__main__':
    main()
