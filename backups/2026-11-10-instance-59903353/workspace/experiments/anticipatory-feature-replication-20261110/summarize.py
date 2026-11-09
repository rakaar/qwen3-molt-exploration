"""Summarize frozen, matched explanation/context comparisons without new scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/2026-11-10-anticipatory-feature-replication'
K = ROOT / 'results/2026-11-10-before-k-description'
L = ROOT / 'results/2026-11-10-before-l-description'


def load(path):
    return json.loads(path.read_text())


def main():
    k = load(K / 'analysis.json')
    l = load(L / 'analysis.json')
    assert k['valid_untruncated_responses'] == 192
    assert l['responses'] == 448 and l['no_invalid_or_truncated_responses']
    OUT.mkdir(exist_ok=True)
    witness = next(w for w in k['context_rescue_witnesses']
                   if w['method'] == 'fuzz' and w['clean_four_way_pattern'])
    assert witness['peak_token'] == ' Henry'
    assert [witness[s] for s in ['old_prefix', 'new_prefix', 'old_extended', 'new_extended']] == [False, False, False, True]
    lines = [
        '# Another anticipatory MOLT: L34–T965 before K', '',
        'Yes: there is a concrete second example of the explanation-plus-truncation problem. '
        'Its aggregate improvement is modest, unlike the earlier T2375 result. A second candidate, '
        'L34–T1440 before L, also shows only small gains.', '',
        '## The clean example', '',
        'The original label is **End-of-Sequence Completion Signal**. Its explanation does not '
        'mention K or activation before a continuation. In the saved strongest dashboard examples, '
        '11 of 12 peaks immediately precede Latin K/k: examples include `DIS → K`, `4 → K`, and `honky-ton → k`.', '',
        'An original held-out positive Fuzz example ends:', '',
        '> ...led by President Richard Nixon and National Security Adviser **Henry** [excerpt ends]', '',
        'The actual gate on the token ` Henry` is **1.3203125**. The recovered original continuation '
        'begins ` Kissinger, wanted to maintain good relations ...`. The revised explanation, '
        'frozen before reading these evaluation passages, describes activation on the token '
        'immediately before K/k-leading text.', '',
        '| Description | Original excerpt ending at Henry | Same excerpt plus 16 original source tokens |',
        '|---|---:|---:|',
        '| Old | 0 — wrong | 0 — wrong |',
        '| Revised before-K explanation | 0 — wrong | 1 — correct |', '',
        'This is the same positive input in Fuzz and Active Fuzz; it counts as one underlying '
        'example, not two independent replications. Its true label, gate and highlighted token '
        'remain unchanged in every condition.', '',
        '## How common is the pattern in its evaluation set?', '',
        'Seven of the 12 original positive contexts end immediately before Latin K/k; one more '
        'precedes Cyrillic к. Five of the seven Latin-K prefixes already contain an earlier '
        'active-before-K example. Only two hide all such earlier active-before-K clues; the '
        'Henry example is one of them. The historical dashboard has 0/12 positive cases correct '
        'in both Fuzz and Active Fuzz. Historical and fresh judge results are kept separate.', '',
        'Fresh results on all seven Latin-K positive contexts:', '',
        '| Test | Old, original prefix | Revised, original prefix | Old, +16 tokens | Revised, +16 tokens |',
        '|---|---:|---:|---:|---:|',
    ]
    for method in ['detection', 'fuzz', 'active_fuzz']:
        s = k['methods'][method]['summary']
        p = s['prefix_only']['strict_before_latin_K']
        e = s['plus_16_source_tokens']['strict_before_latin_K']
        vals = [p['baseline']['correct'], p['candidate']['correct'], e['baseline']['correct'], e['candidate']['correct']]
        lines.append('| ' + method.replace('_', ' ').title() + ' | ' + ' | '.join(f'{v}/7' for v in vals) + ' |')
    lines += [
        '', 'The revised description improves positive Fuzz recognition, but worsens positive '
        'Detection recognition. Revealing the continuation rescues just one additional Fuzz '
        'case with the revised explanation. This does not reproduce the near-complete rescue '
        'seen for T2375.', '',
        'In the 12 matching Active Fuzz inactive-highlight controls, the revised explanation '
        'has 0/12 false positives with either context. The old explanation has 1/12 with '
        'the prefix and 0/12 with the extension.', '',
        '[Full K report, exact descriptions and methods](../2026-11-10-before-k-description/README.md)', '',
        '## Reserve candidate: L34–T1440 before L', '',
        'The original label is **Suffix and Acronym Token Gate**. Nine of 12 strongest dashboard '
        'peaks directly precede L/l; all 12 fit when an intervening separator is allowed. '
        'Of its 50 original positive evaluation contexts, 39 end directly before L/l. '
        'The 28 with no earlier active-before-L example in the prefix were selected by this '
        'visibility rule before any new judge calls. Every original highlighted token in '
        'this selected cohort fits the direct before-L pattern.', '',
        '| Test | Old, original prefix | Revised, original prefix | Old, +16 tokens | Revised, +16 tokens |',
        '|---|---:|---:|---:|---:|',
    ]
    for method in ['detection', 'fuzz', 'active_fuzz']:
        s = l['methods'][method]['contexts']
        p, e = s['prefix_only'], s['plus_16_source_tokens']
        vals = [p['baseline']['positive_correct'], p['candidate']['positive_correct'], e['baseline']['positive_correct'], e['candidate']['positive_correct']]
        lines.append('| ' + method.replace('_', ' ').title() + ' | ' + ' | '.join(f'{v}/28' for v in vals) + ' |')
    l_witness = next(w for w in l['context_rescue_witnesses']
                     if w['method'] == 'fuzz' and w['peak_token'] == ' the'
                     and w['clean_four_way_pattern'])
    lines += [
        '', 'A second clean Fuzz example ends `Next, locate the categories menu on <<the>>` '
        'and hides the continuation ` left-hand side of the website`. The measured gate '
        f"on ` the` is **{l_witness['gate']}**. Old-prefix, revised-prefix and old-extended "
        'all return 0; only revised-extended returns the correct 1. A Detection example '
        'ending `works of art or` before ` literature` has the same four-way pattern.',
        '', 'Active Fuzz inactive-highlight controls:', '']
    for context, label in [('prefix_only', 'Original prefix'), ('plus_16_source_tokens', '+16 source tokens')]:
        c = l['methods']['active_fuzz']['contexts'][context]
        lines.append(f"- {label}: old {c['baseline']['control_false_positives']}/28 false positives; revised {c['candidate']['control_false_positives']}/28.")
    lines += [
        '', 'This candidate provides small context-dependent rescues, not a broad recovery.', '',
        '[Full L report, exact descriptions and methods](../2026-11-10-before-l-description/README.md)', '',
        '## What this establishes', '',
        'The Henry → Kissinger case demonstrates that an omitted anticipatory pattern and '
        'a cut-off continuation can jointly prevent recognition of an actual positive example. '
        'These experiments do not establish that truncation explains all low scores, or that '
        'a before-letter description fully explains either feature. Showing the actual future '
        'changes the information available to the judge, so the extended-context scores are '
        'a diagnostic rather than directly interchangeable benchmark scores.', '',
        'Both revised descriptions were frozen from training-partition dashboard examples '
        'before their evaluation passages were inspected. Feature selection was exploratory. '
        'No description was tuned after these results. Both contexts use the same pinned '
        'local Qwen3.8-27B-NVFP4 judge and inference settings. Sixteen following original '
        'source tokens are appended; original labels and highlights are preserved.', '',
        'Counts above are positive-case recognition, not full-benchmark accuracy. Fuzz and '
        'Active Fuzz positives have identical request bodies and are not independent. '
        'The interrupted L run preserves its original 230 decisions; the remaining 110 '
        'unique requests were resumed, with identical completed request bodies reused. '
        'All 448 logical L decisions are accounted for and verified. The original partial '
        'audit remains separately backed up.', '',
        'No subject-model intervention or new generation was run. No canonical dashboard '
        'descriptions or scores were overwritten.', '',
        '[User-linked downstream activation report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-03-eleven-results-index/experiments/downstream-activation-examples/)', '',
    ]
    (OUT / 'README.md').write_text('\n'.join(lines))
    inputs = [K / 'analysis.json', L / 'analysis.json', K / 'README.md', L / 'README.md', Path(__file__)]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=[dict(
        path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        readme_sha256=hashlib.sha256((OUT / 'README.md').read_bytes()).hexdigest())
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(str(OUT / 'README.md'))


if __name__ == '__main__':
    main()
