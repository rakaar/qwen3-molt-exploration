"""Verify the matched deletion and summarize all 30 original/revised pairs."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    old = json.loads((OUT / 'analyzed-cases.json').read_text())
    new = json.loads((OUT / 'without-starting-one/analyzed-cases.json').read_text())
    summaries = [json.loads((OUT / p).read_text()) for p in ['analysis.json', 'without-starting-one/analysis.json']]
    assert sha(OUT / 'results.json') == new['protocol']['comparison']['original_results_sha256']
    assert old['source_sha256'] == new['source_sha256'] == sha(HERE / 'run.py')
    assert old['protocol']['lists'] == new['protocol']['lists']
    for key in ['base_revision', 'molt_revision', 'molt_checkpoint_sha256', 'feature_sha256', 'enable_thinking', 'decode', 'max_new_tokens']:
        assert old['protocol'][key] == new['protocol'][key], key
    assert old['runtime'] == new['runtime']
    pairs = []
    for before, after in zip(old['cases'], new['cases']):
        assert before['id'] == after['id']
        assert before['prompt'].replace(' starting from 1', '') == after['prompt']
        for key in ['list_id', 'format', 'ordinal', 'words', 'expected', 'list_span', 'item_spans', 'prompt_format']:
            assert before[key] == after[key], key
        assert before['native'] == before['observed'] and after['native'] == after['observed']
        pairs.append({
            'id': before['id'], 'list_id': before['list_id'], 'format': before['format'], 'ordinal': before['ordinal'],
            'expected': before['expected'], 'original_response': before['native']['response'], 'revised_response': after['native']['response'],
            'response_changed': before['native']['response'] != after['native']['response'],
            'strict_improved': not before['assessment']['strict_correct'] and after['assessment']['strict_correct'],
            'strict_worsened': before['assessment']['strict_correct'] and not after['assessment']['strict_correct'],
            'first_word_improved': not before['assessment']['first_word_correct'] and after['assessment']['first_word_correct'],
            'first_word_worsened': before['assessment']['first_word_correct'] and not after['assessment']['first_word_correct'],
        })
    groups = []
    for fmt in ['numbered', 'bullets']:
        for ordinal in [5, 11, 12]:
            groups.append({'format': fmt, 'ordinal': ordinal, **{
                label: next(g for g in summary['groups'] if g['format'] == fmt and g['ordinal'] == ordinal)
                for label, summary in zip(['original', 'revised'], summaries)
            }})
    diagnostic = json.loads((OUT / 'without-starting-one/prefix-diagnostic.json').read_text())
    assert diagnostic['source_sha256'] == sha(HERE / 'prefix_check.py')
    assert diagnostic['all_equal'] and diagnostic['maximum_raw_difference'] == 0
    assert diagnostic['generated_tokens'] == diagnostic['interventions'] == 0
    events = {}
    for label, result in [('original', old), ('revised', new)]:
        events[label] = [dict(case=c['id'], **t) for c in result['cases'] for t in c['trace'] if t['gate'] > 0]
    result = {
        'paired_cases': len(pairs), 'native_observer_exact_runs': 62, 'interventions': 0,
        'only_prompt_change': 'Delete " starting from 1"; preserve all other prompt text, word orders and settings.',
        'original_results_sha256': sha(OUT / 'results.json'),
        'revised_results_sha256': sha(OUT / 'without-starting-one/results.json'),
        'summaries': dict(zip(['original', 'revised'], summaries)), 'groups': groups, 'pairs': pairs,
        **{key: sum(p[key] for p in pairs) for key in ['response_changed', 'strict_improved', 'strict_worsened', 'first_word_improved', 'first_word_worsened']},
        'active_events': events, 'paired_prefix_numerical_check': diagnostic,
    }
    assert len(pairs) == 30
    (OUT / 'comparison-analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ['paired_cases', 'native_observer_exact_runs', 'response_changed', 'strict_improved', 'strict_worsened', 'first_word_improved', 'first_word_worsened']}))

if __name__ == '__main__':
    main()
