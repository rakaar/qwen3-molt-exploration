"""Verify the twelve paired judge queries and preserve historical comparisons."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/2026-11-10-t965-highlight-recheck'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example(messages):
    return messages[-1]['content'].split('Example 0: ', 1)[1].rsplit('\n\nClassify only', 1)[0]


def main():
    plan = load(HERE / 'plan.json')
    protocol = load(HERE / 'protocol.json')
    assert sha(HERE / 'plan.json') == protocol['plan_sha256']
    assert sha(OUT / 'plan.json') == sha(HERE / 'plan.json')
    assert load(OUT / 'judge-settings-comparison.json')['all_match']
    assert sha(OUT / 'run.py') == sha(HERE / 'run.py') == load(OUT / 'runtime.json')['source_sha256']
    assert load(OUT / 'summary.json')['status'] == 'complete'
    rows = [json.loads(s) for s in (OUT / 'audit.jsonl').read_text().splitlines()]
    assert len(rows) == 12
    index = {r['id']: r for r in rows}
    assert len(index) == 12
    controls = []
    historical = load(HERE / 'historical-controls.json')['controls']
    for task in plan['tasks']:
        row = index[task['id']]
        assert all(row[k] == v for k, v in task.items())
        assert row['finish_reason'] == 'stop' and row['response'].strip() in ['[0]', '[1]']
        assert row['prediction'] == (row['response'].strip() == '[1]')
        assert len(row['prompt_token_ids']) + 32 <= 4096
        if row['cached_original_response'] is not None:
            prior = historical[row['id']]
            assert all(row[k] == prior[k] for k in ['messages', 'prompt_token_ids', 'request_sha256']), 'Historical request changed'
            assert row['cached_original_response'] == prior['response'].strip()
            controls.append(dict(id=row['id'], previous=row['cached_original_response'],
                                 fresh=row['response'].strip(), matched=row['response'].strip() == row['cached_original_response'],
                                 exact_messages_tokens_and_request_hash_match=True))
    assert len(controls) == 9
    cases = []
    names = {13: 'Development → Kit', 17: 'Ton / ton → katsu', 20: '.wait → key'}
    for case_id in [13, 17, 20]:
        selected = [r for r in rows if r['original_fuzz_row'] == case_id]
        texts = [example(r['messages']).replace('<<', '').replace('>>', '') for r in selected]
        assert len(set(texts)) == 1, 'Underlying example text changed'
        for r in selected:
            assert example(r['messages']).count('<<') == (1 if r['condition'] == 'final_only' else {13:2,17:3,20:3}[case_id])
        cases.append(dict(original_fuzz_row=case_id, title=names[case_id], conditions={scope:{arm:index[f'row-{case_id}-{scope}-{arm}']['prediction']
                            for arm in ['baseline', 'candidate']} for scope in ['all_original', 'final_only']}))
    report = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), queries=12, historical_controls=9,
                  all_historical_controls_match=all(c['matched'] for c in controls), controls=controls,
                  unchanged_descriptions=True, unchanged_example_text=True, original_fewshots_retained=True,
                  runtime_matched=True, cases=cases, plan_sha256=protocol['plan_sha256'],
                  audit_sha256=sha(OUT / 'audit.jsonl'), posthoc=True,
                  scope='Three selected previously inspected extended positive cases; not a new held-out score or MOLT intervention.')
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# T965: all-original versus final-only highlights', '',
             'The example text, continuation and few-shot demonstrations are identical. Both frozen descriptions are tested under both highlight scopes.', '',
             '| Condition | Development → Kit | Ton / ton → katsu | .wait → key |', '|---|---:|---:|---:|']
    for scope in ['all_original', 'final_only']:
        for arm in ['baseline', 'candidate']:
            lines.append('| ' + scope + ' / ' + ('old' if arm == 'baseline' else 'new') + ' | ' + ' | '.join(str(int(c['conditions'][scope][arm])) for c in cases) + ' |')
    lines += ['', f"Historical responses reproduced: {sum(c['matched'] for c in controls)}/9. Old-description final-only conditions were not previously run and are new comparisons.", '',
              'These are judge answers, not native gate values. All retained marked tokens have measured positive activation in the original source windows. The code window also contains non-K activations when all original highlights are retained.', '',
              'The same pinned Qwen3.8-27B-NVFP4 weights and runtime settings are used. This is a fixed post-hoc recheck, not a new held-out feature benchmark.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
