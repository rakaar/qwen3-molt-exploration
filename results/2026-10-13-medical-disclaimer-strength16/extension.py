"""Matched negative-factor-16 extension, using unchanged original intervention code."""
import hashlib
import json
from pathlib import Path
import run as experiment

ROOT = Path(__file__).resolve().parent
selection = json.loads((ROOT / 'selection.json').read_text())
records, repeats, checks = [], [], {}
for pid in selection['prompt_ids']:
    specs = [{'id': f'extension_{pid}_factor{f}', 'prompt_id': pid,
              'factor': f, 'scope': 'all_tokens', 'mode': 'target'} for f in [1, 0, -8, -16]]
    rows = experiment.generate_batch(specs)
    records.extend(rows)
    experiment.save('extension-trials.json', records)
    reference_specs = [{'id': f'identity_{pid}_row{i}', 'prompt_id': pid,
                        'factor': 1, 'scope': 'all_tokens', 'mode': 'target'} for i in range(4)]
    reference = experiment.generate_batch(reference_specs)
    repeats.extend(reference)
    experiment.save('extension-identities.json', repeats)
    checks[pid] = {'clean_ids_identical_to_all_identity_rows':
                  all(r['generated_token_ids'] == rows[0]['generated_token_ids'] for r in reference)}
    assert checks[pid]['clean_ids_identical_to_all_identity_rows'], pid
    for row in rows:
        print('RESULT',pid,'factor',row['factor'],'gate',round(row['activation_summary']['maximum_gate'],6),
              'edited',len(row['edited_positions']),'stop',row['stop_reason'],
              'output',row['generation'],flush=True)
experiment.save('extension-completion.json', {
    'status': 'complete', 'trial_count': len(records), 'identity_generation_count': len(repeats),
    'checks': checks, 'attention': 'unfrozen eager', 'donor_used': False,
    'extension_source_sha256': hashlib.sha256((ROOT/'extension.py').read_bytes()).hexdigest(),
    'intervention_source_sha256': hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest(),
    'torch': experiment.torch.__version__, 'transformers': experiment.transformers.__version__,
    'gpu':experiment.torch.cuda.get_device_name(),
    'dtype':'BF16 model; FP32 MOLT projections'})
print('EXTENSION_COMPLETE',checks,flush=True)
