# L34–T965: before-K explanation and truncation diagnostic

The candidate wording was frozen from training-partition top activation examples before reading this feature’s evaluation passages. This is an exploratory candidate selected from the user-linked downstream report; it is not a new preregistered or wholly unseen feature evaluation.

The original feature has 12 positive cases per method. Seven peaks immediately precede Latin K/k; one precedes Cyrillic к; four have other continuations. Five of the seven Latin-K passages already show an earlier active-before-k event inside the prefix.

## Fresh paired results: seven strict before-K cases

| Test | Old prefix | New prefix | Old +16 | New +16 |
|---|---:|---:|---:|---:|
| detection | 5/7 | 2/7 | 6/7 | 3/7 |
| fuzz | 0/7 | 3/7 | 0/7 | 4/7 |
| active_fuzz | 0/7 | 3/7 | 0/7 | 4/7 |

## All 12 original positive cases

| Test | Old prefix | New prefix | Old +16 | New +16 |
|---|---:|---:|---:|---:|
| detection | 7/12 | 2/12 | 9/12 | 3/12 |
| fuzz | 0/12 | 3/12 | 1/12 | 4/12 |
| active_fuzz | 0/12 | 3/12 | 1/12 | 4/12 |

## Active Fuzz controls

- prefix_only: old 1/12 false positives; new 0/12.
- plus_16_source_tokens: old 0/12 false positives; new 0/12.

## Cases rescued by showing the continuation

- detection row 9: token '.wait', next text 'key() \n``` \n\nin this code, we first load the face cascade classif'. Old prefix/extended: 1/1; new prefix/extended: 0/1.
- fuzz row 16: token ' Henry', next text ' Kissinger, wanted to maintain good relations with Pakistan as a '. Old prefix/extended: 0/0; new prefix/extended: 0/1.
- active_fuzz row 16: token ' Henry', next text ' Kissinger, wanted to maintain good relations with Pakistan as a '. Old prefix/extended: 0/0; new prefix/extended: 0/1.

## Exact descriptions

**Old**

The gate activates on the final token of a word or phrase, often occurring at the end of a line, sentence, or specific data field. It fires on diverse content types including numbers, proper nouns, technical terms, and common words, suggesting a general end-of-sequence or completion signal rather than a specific semantic theme.

**New, frozen**

Often activates on the token immediately before text beginning with K or k, including within words, names, code and quantities. Examples include DIS before K in DISK, 4 before K in 4K, and honky-ton before the final k. Activation belongs to the preceding word fragment, digit, space or punctuation, rather than necessarily to K itself. Some activations have non-K continuations, so this is a frequent pattern, not an exclusive rule.

## Method

- Same fresh local judge and arithmetic settings as the T2375 comparison. Both descriptions and both context conditions were newly run, with temperature 0 and thinking disabled. Historical dashboard metrics are separate.
- Original feature JSON supplies exact labels, token gates and highlighted positions. Shared verified Delphi system/few-shot prefixes are reused; final user messages are reconstructed with the same template. This is not claimed as a byte-verified historical prompt-export replay.
- All 12 positives are scored in all three methods; Active Fuzz also includes all 12 matching zero-gate highlighted controls. There are 192 logical decisions and 144 unique request bodies. Positive Fuzz and Active Fuzz inputs are identical and are not independent replications.
- Sixteen original source tokens are appended; existing highlights and labels are unchanged. Future context cannot alter the previously measured gate at an earlier causal-model token.
- Original source windows are matched to the same held-out UltraChat conversations. The native subject model was not rerun, and dashboard descriptions/scores were not overwritten.

[Original dashboard](https://amsterdam.taild80f17.ts.net/layers/layer-34/features/965.html) · [User-linked report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-03-eleven-results-index/experiments/downstream-activation-examples/)
