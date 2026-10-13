"""Additional strengths with the unchanged original dynamic contribution edit."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import run as experiment

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--stage', required=True)
parser.add_argument('--factors', type=float, nargs=4, required=True)
parser.add_argument('--prompts', nargs='+')
args = parser.parse_args()
assert args.factors[0] == 1
factors = [int(f) if f.is_integer() else f for f in args.factors]
selected = json.loads((ROOT / 'selection.json').read_text())['prompt_ids']
pids = args.prompts or selected
assert set(pids).issubset(selected)
reference_path = ROOT / 'reference-clean.json'
if reference_path.exists():
    reference = json.loads(reference_path.read_text())
else:
    reference = json.loads((ROOT / 'artifacts/extension-trials.json').read_text())
baseline = {r['prompt_id']: r for r in reference if r['factor'] == 1}
records, checks = [], {}
for pid in pids:
    specs = [{'id': f"{args.stage}_{pid}_factor{factor}", 'prompt_id': pid,
              'factor': factor, 'scope': 'all_tokens', 'mode': 'target'} for factor in factors]
    rows = experiment.generate_batch(specs)
    records.extend(rows)
    experiment.save(f'sweep-{args.stage}.json', records)
    checks[pid] = rows[0]['generated_token_ids'] == baseline[pid]['generated_token_ids']
    assert checks[pid], f'Clean continuation changed: {pid}'
    for row in rows:
        print('RESULT', row['id'], 'stop', row['stop_reason'],
              'edited', len(row['edited_positions']), 'output', row['generation'], flush=True)
experiment.save(f'sweep-{args.stage}-completion.json', {
    'status': 'complete', 'stage': args.stage, 'factors': factors, 'prompts': pids,
    'trial_count': len(records), 'clean_generated_ids_match_previous_run': checks,
    'source_sha256': hashlib.sha256((ROOT/'high_strength.py').read_bytes()).hexdigest(),
    'intervention_source_sha256': hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest(),
    'selected_weight_sha256': hashlib.sha256((ROOT/'weights/molt-19-selected.safetensors').read_bytes()).hexdigest(),
    'torch': experiment.torch.__version__, 'transformers': experiment.transformers.__version__,
    'python': platform.python_version(), 'gpu': experiment.torch.cuda.get_device_name(),
    'attention': 'unfrozen eager', 'dtype': 'BF16 model; FP32 MOLT projections',
    'donor_used': False})
print('SWEEP_STAGE_COMPLETE', args.stage, flush=True)
