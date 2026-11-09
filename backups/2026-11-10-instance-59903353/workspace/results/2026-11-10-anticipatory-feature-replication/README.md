# Another anticipatory MOLT: L34–T965 before K

Yes: there is a concrete second example of the explanation-plus-truncation problem. Its aggregate improvement is modest, unlike the earlier T2375 result. A second candidate, L34–T1440 before L, also shows only small gains.

## The clean example

The original label is **End-of-Sequence Completion Signal**. Its explanation does not mention K or activation before a continuation. In the saved strongest dashboard examples, 11 of 12 peaks immediately precede Latin K/k: examples include `DIS → K`, `4 → K`, and `honky-ton → k`.

An original held-out positive Fuzz example ends:

> ...led by President Richard Nixon and National Security Adviser **Henry** [excerpt ends]

The actual gate on the token ` Henry` is **1.3203125**. The recovered original continuation begins ` Kissinger, wanted to maintain good relations ...`. The revised explanation, frozen before reading these evaluation passages, describes activation on the token immediately before K/k-leading text.

| Description | Original excerpt ending at Henry | Same excerpt plus 16 original source tokens |
|---|---:|---:|
| Old | 0 — wrong | 0 — wrong |
| Revised before-K explanation | 0 — wrong | 1 — correct |

This is the same positive input in Fuzz and Active Fuzz; it counts as one underlying example, not two independent replications. Its true label, gate and highlighted token remain unchanged in every condition.

## How common is the pattern in its evaluation set?

Seven of the 12 original positive contexts end immediately before Latin K/k; one more precedes Cyrillic к. Five of the seven Latin-K prefixes already contain an earlier active-before-K example. Only two hide all such earlier active-before-K clues; the Henry example is one of them. The historical dashboard has 0/12 positive cases correct in both Fuzz and Active Fuzz. Historical and fresh judge results are kept separate.

Fresh results on all seven Latin-K positive contexts:

| Test | Old, original prefix | Revised, original prefix | Old, +16 tokens | Revised, +16 tokens |
|---|---:|---:|---:|---:|
| Detection | 5/7 | 2/7 | 6/7 | 3/7 |
| Fuzz | 0/7 | 3/7 | 0/7 | 4/7 |
| Active Fuzz | 0/7 | 3/7 | 0/7 | 4/7 |

The revised description improves positive Fuzz recognition, but worsens positive Detection recognition. Revealing the continuation rescues just one additional Fuzz case with the revised explanation. This does not reproduce the near-complete rescue seen for T2375.

In the 12 matching Active Fuzz inactive-highlight controls, the revised explanation has 0/12 false positives with either context. The old explanation has 1/12 with the prefix and 0/12 with the extension.

[Full K report, exact descriptions and methods](../2026-11-10-before-k-description/README.md)

## Reserve candidate: L34–T1440 before L

The original label is **Suffix and Acronym Token Gate**. Nine of 12 strongest dashboard peaks directly precede L/l; all 12 fit when an intervening separator is allowed. Of its 50 original positive evaluation contexts, 39 end directly before L/l. The 28 with no earlier active-before-L example in the prefix were selected by this visibility rule before any new judge calls. Every original highlighted token in this selected cohort fits the direct before-L pattern.

| Test | Old, original prefix | Revised, original prefix | Old, +16 tokens | Revised, +16 tokens |
|---|---:|---:|---:|---:|
| Detection | 3/28 | 4/28 | 3/28 | 5/28 |
| Fuzz | 1/28 | 1/28 | 0/28 | 2/28 |
| Active Fuzz | 1/28 | 1/28 | 0/28 | 2/28 |

A second clean Fuzz example ends `Next, locate the categories menu on <<the>>` and hides the continuation ` left-hand side of the website`. The measured gate on ` the` is **0.97265625**. Old-prefix, revised-prefix and old-extended all return 0; only revised-extended returns the correct 1. A Detection example ending `works of art or` before ` literature` has the same four-way pattern.

Active Fuzz inactive-highlight controls:

- Original prefix: old 0/28 false positives; revised 0/28.
- +16 source tokens: old 0/28 false positives; revised 0/28.

This candidate provides small context-dependent rescues, not a broad recovery.

[Full L report, exact descriptions and methods](../2026-11-10-before-l-description/README.md)

## What this establishes

The Henry → Kissinger case demonstrates that an omitted anticipatory pattern and a cut-off continuation can jointly prevent recognition of an actual positive example. These experiments do not establish that truncation explains all low scores, or that a before-letter description fully explains either feature. Showing the actual future changes the information available to the judge, so the extended-context scores are a diagnostic rather than directly interchangeable benchmark scores.

Both revised descriptions were frozen from training-partition dashboard examples before their evaluation passages were inspected. Feature selection was exploratory. No description was tuned after these results. Both contexts use the same pinned local Qwen3.8-27B-NVFP4 judge and inference settings. Sixteen following original source tokens are appended; original labels and highlights are preserved.

Counts above are positive-case recognition, not full-benchmark accuracy. Fuzz and Active Fuzz positives have identical request bodies and are not independent. The interrupted L run preserves its original 230 decisions; the remaining 110 unique requests were resumed, with identical completed request bodies reused. All 448 logical L decisions are accounted for and verified. The original partial audit remains separately backed up.

No subject-model intervention or new generation was run. No canonical dashboard descriptions or scores were overwritten.

[User-linked downstream activation report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-03-eleven-results-index/experiments/downstream-activation-examples/)
