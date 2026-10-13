"""Resolve mixed-padding BF16 differences with matched four-row comparisons."""
import hashlib
import json
from pathlib import Path

import run as experiment

ROOT = Path(__file__).resolve().parent
selection = json.loads((ROOT / 'selection.json').read_text())
groups = {name: [] for name in ['negative', 'positive', 'random', 'single']}
for pid in selection['prompt_ids']:
    def spec(label, factor, mode='target', **extra):
        return {'id': f'paired_{pid}_{label}', 'prompt_id': pid,
                'factor': factor, 'scope': 'all_tokens', 'mode': mode, **extra}
    negative = experiment.generate_batch([spec('clean', 1), spec('remove', 0),
                                          spec('negative4', -4), spec('negative8', -8)])
    groups['negative'].extend(negative)
    experiment.save('paired-negative.json', groups['negative'])
    positive = experiment.generate_batch([spec('clean', 1), spec('positive2', 2),
                                          spec('positive4', 4), spec('remove', 0)])
    groups['positive'].extend(positive)
    experiment.save('paired-positive.json', groups['positive'])
    random = experiment.generate_batch([spec('clean', 1), spec('remove', 0),
                                        spec('random0', 0, 'random_matched_norm', seed=0),
                                        spec('random1', 0, 'random_matched_norm', seed=1)])
    groups['random'].extend(random)
    experiment.save('paired-random.json', groups['random'])
    clean = negative[0]
    active = [a for a in clean['activations'] if a['segment'] == 'response' and a['target_gate'] > 0]
    site = max(active, key=lambda a: a['target_gate'])['position'] if active else clean['activations'][-1]['position']
    single_specs = [spec('clean', 1), spec('single_remove', 0, position=site),
                    spec('single_negative4', -4, position=site), spec('unrelated', 0, 'unrelated')]
    for s in single_specs[1:3]:
        s['scope'] = 'single_position'
        s['site_selected_from_matched_clean'] = True
    single = experiment.generate_batch(single_specs)
    groups['single'].extend(single)
    experiment.save('paired-single.json', groups['single'])
    consistent = all(r[0]['generated_token_ids'] == clean['generated_token_ids']
                     for r in [positive, random, single])
    print('PAIRED', pid, 'matched_clean_identical', consistent,
          'clean', clean['generation'][:120].replace('\n', ' '),
          'removed', negative[1]['generation'][:120].replace('\n', ' '),
          'negative8', negative[3]['generation'][:120].replace('\n', ' '), flush=True)

checks = {}
for pid in selection['prompt_ids']:
    clean_cases = [next(r for r in rows if r['prompt_id'] == pid and r['factor'] == 1)
                   for rows in groups.values()]
    checks[pid] = {'matched_clean_generated_tokens_identical':
                   all(r['generated_token_ids'] == clean_cases[0]['generated_token_ids'] for r in clean_cases)}
experiment.save('paired-completion.json', {
    'status': 'complete', 'comparative_generation_count': sum(map(len, groups.values())),
    'comparison_policy': 'Every comparison uses four identical prompt strings and identical initial padding. Row zero is unchanged. Batch rows remain independent.',
    'identity_checks': checks,
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'generation_source_sha256': hashlib.sha256((ROOT / 'run.py').read_bytes()).hexdigest(),
})
print('PAIRED_VERIFICATION_COMPLETE', checks, flush=True)
