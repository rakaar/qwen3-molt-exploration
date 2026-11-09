"""Verify every fixed diagnostic request and report effects without retuning."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/2026-11-10-t965-failed-cases'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = load(OUT / 'protocol.json')
    assert sha(OUT / 'plan.json') == protocol['plan_sha256']
    assert sha(OUT / 'plan.json') == sha(HERE / 'plan.json')
    plan = load(OUT / 'plan.json')
    assert hashlib.sha256(plan['candidate_explanation'].encode()).hexdigest() == protocol['source_candidate_sha256']
    assert load(OUT / 'judge-settings-comparison.json')['all_match']
    audit = [json.loads(s) for s in (OUT / 'audit.jsonl').read_text().splitlines()]
    assert len(audit) == len(plan['tasks']) == 22
    index = {r['id']: r for r in audit}
    assert len(index) == 22
    for task in plan['tasks']:
        row = index[task['id']]
        assert all(row[k] == v for k, v in task.items()), task['id']
        assert row['finish_reason'] == 'stop' and row['response'].strip() in ['[0]', '[1]']
        assert row['prediction'] == (row['response'].strip() == '[1]')
        assert len(row['prompt_token_ids']) + 32 <= 4096
    original_audit = [json.loads(s) for s in (ROOT / 'results/2026-11-10-before-k-description/audit.jsonl').read_text().splitlines()]
    controls = []
    for row in audit:
        if row['cached_original_response'] is None:
            continue
        source = next(r for r in original_audit if r['method'] == 'fuzz' and r['arm'] == 'candidate'
                      and r['context'] == 'plus_16_source_tokens' and r['original_row_index'] == row['original_fuzz_row'])
        assert row['messages'] == source['messages'] and row['prompt_token_ids'] == source['prompt_token_ids']
        assert row['request_sha256'] == source['request_sha256']
        controls.append(dict(id=row['id'], previous=source['response'].strip(), fresh=row['response'].strip(),
                             matched=row['prediction'] == source['prediction']))
    assert len(controls) == 4
    conditions = ['repeat_original', 'single_peak', 'single_peak_no_fewshot', 'literal_all_highlights',
                  'literal_single_peak', 'minimal_positive', 'minimal_negative']
    results = {str(case): {condition: dict(prediction=index[f'row-{case}-{condition}']['prediction'],
                                          response=index[f'row-{case}-{condition}']['response'],
                                          literal_pattern_expectation=index[f'row-{case}-{condition}']['literal_before_K_expectation'])
                          for condition in conditions} for case in [13, 17, 20]}
    summary = dict(verified_utc=datetime.now(timezone.utc).isoformat(), feature='L34-T965',
                   valid_queries=22, candidate_explanation_unchanged=True, posthoc=True,
                   all_original_repeats_match=all(c['matched'] for c in controls), repeats=controls,
                   cases=results, runtime_matched=True,
                   source_hashes={p.name: sha(p) for p in [OUT / 'audit.jsonl', OUT / 'plan.json', OUT / 'runtime.json', Path(__file__)]},
                   warning='This diagnoses prompt behavior. Literal tasks and rewritten minimal examples are not held-out feature scores or fresh subject activation labels.')
    (OUT / 'diagnostic-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    labels = {'repeat_original': 'Original query, repeated', 'single_peak': 'Only final peak marked; original few-shots',
              'single_peak_no_fewshot': 'Only final peak marked; no few-shots',
              'literal_all_highlights': 'Literal before-K task; all original marks',
              'literal_single_peak': 'Literal before-K task; final peak only',
              'minimal_positive': 'Minimal before-K string; original explanation/few-shots',
              'minimal_negative': 'Minimal non-K string; original explanation/few-shots'}
    lines = ['# L34–T965: fixed follow-up judge diagnostic', '',
             'These 22 queries were prepared and frozen before this run, after inspecting the three old misses. '
             'They are post-hoc prompt diagnostics, not new held-out feature scores. The candidate explanation '
             'is unchanged wherever it is used. The separate literal-character task changes the task explicitly.', '',
             f"All 22 responses are valid and untruncated. Same pinned judge/runtime: verified. All four original repeats match: **{summary['all_original_repeats_match']}**.", '',
             '| Condition | Development → Kit | Ton/ton → katsu | Code / .wait → key |', '|---|---:|---:|---:|']
    for condition in conditions:
        lines.append('| ' + labels[condition] + ' | ' + ' | '.join(str(int(results[str(i)][condition]['prediction'])) for i in [13, 17, 20]) + ' |')
    lines += ['', '## How to read this comparison', '',
              '- Original positive Fuzz labels are 1 in all three windows because every originally marked token has measured positive gate activation.',
              '- For the literal before-K task, all-highlight expectations are 1, 1, 0: the code window marks comma and plus before h as well as .wait before k. Final-peak-only expectations are 1, 1, 1.',
              '- Minimal positive strings fit the literal before-K rule; minimal negative strings do not. Their native MOLT activation was not measured, and the explanation allows other unspecified activations. They therefore test judge interpretation, not feature specificity.',
              '- Compare single-peak with single-peak/no-few-shot to isolate removing the demonstration bundle. Compare original with single-peak to isolate removing earlier highlight markers while preserving every example character.',
              '- Compare literal-task outputs to rule expectations to check character recognition. This changes both the instruction and explanatory framing and cannot isolate one particular word of the original description.',
              '- Causal interpretations of prompt changes require the original repeats to match. A failed repeat is evidence of instability and must be reported before attributing differences to the diagnostic manipulation.', '',
              '## Minimal constructed strings', '']
    for case in [13, 17, 20]:
        for condition in ['minimal_positive', 'minimal_negative']:
            text = next(t for t in plan['tasks'] if t['id'] == f'row-{case}-{condition}')['messages'][-1]['content']
            excerpt = text.split('Example 0: ', 1)[1].rsplit('\n\nClassify only', 1)[0]
            lines.append(f"- Row {case}, {condition}: `{excerpt}` → `{index[f'row-{case}-{condition}']['response'].strip()}`.")
    lines += ['', '[Original three examples and token gates](README.md)', '']
    (OUT / 'diagnostic-results.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
