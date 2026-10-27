"""Run four isolated single-gate studies using the resident native model."""
import hashlib
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def run_all(model, tokenizer):
    assert not (ROOT / 'combined-results.json').exists(), 'Inspect existing results before retrying.'
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    expected = json.loads((ROOT / 'baseline-reference.json').read_text())
    combined = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'protocol': protocol, 'gate_runs': [], 'simultaneous_gate_ablations': False,
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        for feature in protocol['features']:
            name = f"L{feature['layer']}-T{feature['transform']}"
            folder = ROOT / name
            with np.load(folder / feature['file'], allow_pickle=False) as arrays:
                weights = {k: torch.from_numpy(arrays[k].copy()).float().cuda() for k in arrays.files}
            namespace = runpy.run_path(str(folder / 'run.py'))
            result = namespace['run_experiment'](model, tokenizer, weights)
            assert result['status'] == 'complete' and result['hooks_remaining'] == 0
            assert result['only_edited_feature'] == {'layer': feature['layer'], 'transform': feature['transform']}
            summaries = []
            for row in result['cases']:
                native = row['native']
                reference = expected[row['summary']['case_id']]
                assert native['prompt_token_ids'] == reference['prompt_token_ids']
                assert native['output_token_ids'] == reference['output_token_ids']
                assert native['steps'] == reference['steps'], 'Native baseline differed from the preceding study.'
                summaries.append(row['summary'])
            combined['gate_runs'].append({'feature': feature, 'name': name,
                                          'result_file': str((folder / 'artifacts/results.json').relative_to(ROOT)),
                                          'baseline_matches_previous_study_exactly': True,
                                          'summaries': summaries})
            save(ROOT / 'combined-results.json', combined)
            print('GATE_COMPLETE', name, '12 cases; baseline exactly matches previous study.', flush=True)
            del weights, namespace, result
        assert len(combined['gate_runs']) == 4
        combined.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        total_gate_case_comparisons=48,
                        hooks_remaining=sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()))
        assert combined['hooks_remaining'] == 0
        save(ROOT / 'combined-results.json', combined)
        print('ALL_COMPLETE: four separately tested gates, 48 gate/prompt comparisons.', flush=True)
    except BaseException as exc:
        combined.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save(ROOT / 'combined-results.json', combined)
        raise
    return combined
