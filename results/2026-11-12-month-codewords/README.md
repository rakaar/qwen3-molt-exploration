# Month codewords: L29 T2375 gate-off pilot

Five fixed shuffled dictionaries map all twelve month names to twelve ordinary codewords. The codebook line order is independently shuffled. Query eleventh/twelfth in raw and chat formats (20 coded cases). Pinned native Qwen3-4B, thinking off for chat, greedy decoding, 32-token limit. Native and zero/observer for all primary cases are hash-locked before nonzero edits.

Two chat pairs pass both native codeword-only checks: mappings 2 and 4. Seven of ten chat responses meet the strict requirement; no raw pair passes it. Raw continuations often explain, repeat or name the month, sometimes including the correct codeword inside extra text. All remain visible; no ablation is run on a failed pair.

Primary result, removing every positive supplied-input T2375 contribution:

| Mapping | November code / December code | Native -> off | P(November code), native -> off | P(December code), native -> off |
|---|---|---|---|---|
| 2 | lantern / velvet | lantern -> lantern | 99.9031% -> 99.9570% | 0.00000992% -> 0.00000441% |
| 4 | velvet / violin | velvet -> velvet | 99.9048% -> 99.8896% | 0.00000013% -> 0.00000015% |

Both eligible eleventh prompts naturally activate at input positions 96 (newline after im_end) and 103 (double newline after /think). Their query-final ` is` gate is zero. Off-is therefore removes zero contribution; off-last removes one prefix contribution; off-all removes both. No generated tokens are edited. Both twelfth controls have zero positive input activation and retain their correct codeword.

Same output word contrast: velvet is an eleventh-month answer in mapping 4 (two active prefix gates) and a twelfth-month answer in mapping 2 (zero positive input gates). This is observational, with different query/codebook contexts; not a causal mechanism claim.

The plain raw reference reproduces an initial November -> December shift at final ` is` (gate 1.1585536). P(November) 66.8860% -> 8.8328%; P(December) 24.3863% -> 83.7350%. Full 32-token continuations are preserved; both hit the limit and contain later erroneous statements. This is an initial-answer effect.

After the primary outcomes, an explicitly exploratory matched-control phase adds six chat cases: the same two codebooks requesting month names, plus plain chat month-name requests, each at ordinals 11/12. All native checks pass. All-input removal preserves November/December response text in all six; no scope changes any response. Probabilities do change: December gains in all three eleventh-month name controls (e.g. 0.247256% -> 3.732597% in same-codebook-4). The codeword probability effects are much smaller. Natural contribution sizes and directions differ, so these are not norm-matched comparisons. Therefore this pilot does not attribute the absent codeword shift specifically to recoding: month-name chat controls also do not shift. The successful reference is a short raw completion.

Implementation: native L29 MLP output minus the natural T2375 gate times its current local low-rank operator in native MLP-output units. The observer sees post-attention residual before native RMSNorm. Checkpoint input standardization, JumpReLU gate, UV operator and output std are applied exactly. FP32 subtraction then one BF16 cast; no extra RMSNorm, no whole-MLP replacement. Native subsequent layers run normally.

Scopes are recorded per case. Across both phases, 34 gate-off scope conditions include 19 zero-gate no-ops; 23 positive input-position contributions are removed. All 28 native/zero case pairs match exactly. Independent CPU verification covers 71 vector positions, gate/operator reconstruction, full selected masks, contribution subtraction and exact BF16 rounding. Candidate continuation scores are independently checked against stored logits/logsumexp and the greedy path when prefixes agree.

Probability definition: sum four disjoint prescribed exact output-token prefixes for the complete word: lowercase/titlecase, with zero/one leading space, deduplicated by token sequence. The input is fixed and never retokenized. Alternate tokenizations, sentences, punctuation and EOS are excluded. Word-plus-EOS scores are separate in CSV/JSON. These are token-prefix events, not all semantic ways of answering.

Artifacts: index.html/study-data.json present prompts, input/response heatmaps, native/off responses, every failed baseline and matched controls. protocol.json freezes the primary cases. baseline-results.json/baseline-lock.json prove baseline-before-ablation order. results.json, compressed vectors and probability CSV retain full captures. matched-controls/ contains a separately frozen adaptive control phase. Source and pinned model/MOLT revisions are in experiments/month-codewords-20261112 and source/.

Limitations: only two eligible codebooks; baseline formatting failures; raw-vs-chat task dependence; no unrelated norm-matched active-transform specificity controls. Neither an abstract array[10] implementation nor a purely lexical mechanism is established.
