# Water/oil flask binding and lexical input observations

Four counterbalanced layouts swap water/oil between Dax and Neri and reverse fact order. Both names are queried in each layout. All 8 native responses pass the frozen codeword-only competence criterion, and all 8 all-input T2375-off responses remain correct. No text response changes. Each case removes one positive input contribution at the supplied water token; all input positions are selected, all prompt text retained, and generated tokens unedited.

| Layout/query | Expected | Native -> off | P(expected), before -> after | Delta ln(P(water)/P(oil)) |
|---|---|---|---|---|
| original-dax-first-ask-dax | water | Water. -> Water. | 100.0000% -> 100.0000% | -0.124999 |
| original-dax-first-ask-neri | oil | Oil. -> Oil. | 99.6901% -> 99.7585% | -0.124999 |
| original-neri-first-ask-dax | water | Water. -> Water. | 99.9993% -> 99.9991% | -0.203594 |
| original-neri-first-ask-neri | oil | Oil. -> Oil. | 99.9925% -> 99.9904% | 0.124997 |
| swapped-dax-first-ask-dax | oil | Oil. -> Oil. | 99.9992% -> 99.9992% | -0.249998 |
| swapped-dax-first-ask-neri | water | Water. -> Water. | 99.9994% -> 99.9994% | -0.125247 |
| swapped-neri-first-ask-dax | oil | Oil. -> Oil. | 99.7130% -> 99.7130% | 0.252433 |
| swapped-neri-first-ask-neri | water | Water. -> Water. | 100.0000% -> 100.0000% | 0.000074 |

The largest absolute expected-word probability change is 0.068380 percentage points. No matched active-transform specificity control was run, and near-certain baseline answers limit interpretation of unchanged greedy output.

Lexical controls are observation-only. Same carrier: The word is WORD. Repeat that word, and nothing else. Lowercase water, watermelon and waterloo share the exact causal input-token prefix through token 3015 (space+water) and have the same gate 2.0321655 there. At that position the model has not yet seen melon/loo; equal activation cannot establish equal meanings of the completed words. Uppercase Waterloo is one input token (76375) with gate zero. Its generated response splits Water+loo and activates on Water (2.107835). Lowercase waterloo splits water+loo and its response also has activity on water (0.984628). This is not a universal lexical or semantic classification; tokenization and context differ.

The native copying task fails for watermelon and lowercase waterloo by repeating each word twice. Those outputs are retained. They do not affect the validity of the input-activation observation and are not counted as causal successes. No lexical-case ablations were run.

All native/zero observations are saved and hash-locked before nonzero edits; eligibility uses paired native answers only, not activation or intervention outcomes. Natural contributions are computed from the standardized pre-RMSNorm L29 post-attention residual and subtracted from native MLP output in FP32 followed by one BF16 cast. Native later layers remain; no full MLP replacement.

Pinned Qwen3-4B and MOLT checkpoints, native BF16/observer FP32, eager attention, TF32 disabled, greedy, chat thinking off, max 32 output tokens. Word-prefix scores sum prescribed lowercase/titlecase variants with zero/one leading space. Full multi-token sequences are scored and exact word-plus-EOS is separate. A water prefix is not a semantic or word-boundary event. Raw FP32-based probability sums can exceed one by tiny rounding amounts; display percentages are rounded.

CPU audits verify native/zero identity, source hashes, baseline locking, every edit mask, gates/operators, BF16 rounding and complete answer-prefix probabilities. Original raw eleventh-month calibration reproduces November -> December; twelfth reference retained.
