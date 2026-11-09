# L34–T1440: a second anticipatory-feature evaluation diagnostic

The candidate description was frozen from training-partition dashboard windows before its evaluation passages were inspected. This is an exploratory follow-up, not a wholly unseen or preregistered feature evaluation.

## Coverage

- 39/50 original positive evaluation windows end immediately before an L/l-leading continuation.
- 28/39 have no earlier active-before-L example visible in the prefix. All 28 have unambiguous recovered source continuations, and all originally highlighted tokens match the before-L pattern.
- Those 28 were selected by the visibility rule before any new judge calls, not by which responses improved. Both descriptions are unchanged.

## Fresh results: positive cases correct

| Test | Old prefix | New prefix | Old +16 source tokens | New +16 source tokens |
|---|---:|---:|---:|---:|
| detection | 3/28 | 4/28 | 3/28 | 5/28 |
| fuzz | 1/28 | 1/28 | 0/28 | 2/28 |
| active_fuzz | 1/28 | 1/28 | 0/28 | 2/28 |

## Active Fuzz controls

- prefix_only: old 0/28 false positives; new 0/28.
- plus_16_source_tokens: old 0/28 false positives; new 0/28.

Fuzz and Active Fuzz have the same positive requests; these are not independent replications. Counts concern the selected positive contexts and the matching Active Fuzz controls, not the original full 100-case benchmark.

## Exact original description

The gate activates on tokens that are part of a compound word, acronym, or proper noun where the specific token is a suffix, abbreviation, or capitalized component, often appearing in technical, legal, or medical contexts.

## Exact revised description, frozen before evaluation inspection

Often activates on the token immediately before an L- or l-initial continuation, in words, names, code, abbreviations and units. Examples include a comma before London, National before Lab, and 1 before L. Activation can occur on a word fragment, digit, quote, space or punctuation before the L-bearing text, sometimes with an intervening separator. Other contexts also activate the gate; this is a frequent anticipatory pattern rather than an exclusive rule.

## Rescued cases

- detection row 26: ' or' -> ' literature that originated in Kaluga and are famous worldwide:\n\n1. "F'; old prefix/extended 0/0, new prefix/extended 0/1.
- detection row 33: ' and' -> " lows of Gascoigne's career and the impact that he had on British foot"; old prefix/extended 0/1, new prefix/extended 0/1.
- fuzz row 24: ',\n' -> "Let's keep alive this priceless legacy.\n\nTogether, we can make a diffe"; old prefix/extended 0/0, new prefix/extended 0/1.
- fuzz row 59: ' the' -> " left-hand side of the website.\n3. Click on 'Engraving & Etching for G"; old prefix/extended 0/0, new prefix/extended 0/1.
- active_fuzz row 24: ',\n' -> "Let's keep alive this priceless legacy.\n\nTogether, we can make a diffe"; old prefix/extended 0/0, new prefix/extended 0/1.
- active_fuzz row 59: ' the' -> " left-hand side of the website.\n3. Click on 'Engraving & Etching for G"; old prefix/extended 0/0, new prefix/extended 0/1.

## Scope and verification

- The same pinned fresh Qwen3.8-27B-NVFP4 local judge and inference arithmetic as the preceding T2375 study were used. Both descriptions and both context conditions were newly evaluated. Historical original-dashboard scores remain separate.
- Original authenticated feature data supply actual gate values, ground-truth labels and highlight positions. Source windows are matched to the original held-out UltraChat conversations; the candidate wording was not tuned to these passages.
- Shared verified Delphi system/few-shot messages are reused. Final user messages follow the same original scoring template; this is not claimed as a byte-verified historical prompt-export replay.
- Sixteen following original source tokens are appended. Gate labels and original highlight positions stay unchanged. Future context cannot change the already recorded gate at an earlier token of a causal language model.
- Native subject-model generation or contribution ablation was not run. This tests explanation recognition by the judge, not the causal function of the MOLT.
- All 448 logical responses are valid and untruncated; there are 336 unique prompt bodies. Identical prompt bodies have identical predictions. Downloaded result files are hash-verified.

The original run exited after 230 decisions; its cause was not established. Those decisions and its full prepared query plan were separately downloaded and hash-verified. The resumed attempt made only 110 unfinished unique queries. Identical completed requests, including shared positive Fuzz/Active Fuzz inputs, were reused with their original responses. There were 340 actual calls across the two attempts, covering 336 unique requests and all 448 planned logical decisions. The original partial audit is preserved unchanged.

[Original feature](https://amsterdam.taild80f17.ts.net/layers/layer-34/features/1440.html) · [User-linked activation report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-03-eleven-results-index/experiments/downstream-activation-examples/)
