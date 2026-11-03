# Why eleven-off produces twelve: saved output competition

Prompt: `the eleventh month of the year is`. L29–T2375 removed only at final input ` is`, position 7. No new inference.

| Answer event | Native | Gate off |
|---|---:|---:|
| November variants | 66.8686% | 8.8313% |
| December variants | 24.3655% | 83.6775% |

If only the two November tokens lose combined probability mass to their observed gate-off level, with all other relative preferences fixed, December reaches 67.0474%. It already becomes the greedy answer under this analytic control. Actual ablation gives 83.6775%, a further 16.6301 percentage points.

December as a fraction of probability outside those two November tokens rises from 73.5422% to 91.7832%. Thus the actual edit changes relative preferences among the remaining tokens as well.

Lowercase November raw logit: 23.250000 to 21.750000. Lowercase December: 22.125000 to 23.750000. These include downstream computation and changed final normalization.

This supports separating two questions: why December is the fallback competitor, and why ablation further favors it. It does not show an internal plus-one computation.

Probabilities sum exact leading-space lowercase/title-case one-token forms. Other spellings/fragments and following punctuation/EOS are excluded. The control is an output-distribution calculation, not a run in which internal MOLT activity was preserved and logits were edited.

Reproduce: `python3 experiments/eleven-twelve-competition-20261103/analyze.py`. Exact values, source hash and limitations are in summary.json.
