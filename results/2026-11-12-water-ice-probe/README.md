# Water/ice probe: L29 T2375

All six native probe answers pass the frozen competence checks. No probe response changes after removing all positive supplied-input T2375 contributions. The raw ice completion is an initial-answer result with a 32-token truncated continuation.

The three ice-melting prompts without literal water in their inputs have zero post-threshold gate activity throughout the supplied input and all captured native generated tokens, including Water/water. Their off conditions are no-ops, not failed removals of an active feature.

Mentioning Water explicitly in the user prompt gives one active input token (gate 2.045241); removing it retains Water. P(water) changes 99.9995851494% -> 99.9995870733%. In the reverse water-freezing prompt, input water has gate 1.112331; removal retains Ice. P(ice) changes 99.9960170651% -> 99.9949986798%. All native response gates are zero in the six probe captures.

The original raw eleventh-month calibration again starts November natively and December after removing the final-input contribution. The twelfth-month reference is retained. These verify the contribution-edit implementation, not general water-function specificity.

Scope: native BF16 Qwen3-4B with observation before L29 post-attention RMSNorm; subtract gate times standardized-input low-rank operator in native output units, in FP32 followed by one BF16 cast. Only final supplied input or all supplied input positions targeted. No generated or candidate continuation tokens edited. Native later layers run normally.

Timing: processing an already-selected token predicts the following token. Activity on or after completion of a water answer cannot cause that already-completed word to have been selected. The current probes actually have no positive response activation. The same literal Water token can activate in the input and not in the response, illustrating context dependence.

Probabilities sum four prescribed disjoint complete word-token prefixes: lowercase/titlecase with zero/one leading space, deduplicated by token sequence. These are not semantic probabilities across synonyms or alternate tokenizations. Word-plus-EOS scores are separate. Raw FP32-based probabilities are retained; tiny floating-point sums can exceed one by a few parts in 100 million. Displayed percentages are rounded.

Audits: native and zero controls match exactly for all eight cases; baseline results hash-locked before nonzero edits; all selected masks and actual edited positions verified. Independent CPU gate/operator reconstruction and exact BF16 rounding pass at 28 saved vector positions. Four positive position contributions removed across sixteen scope conditions; zero generated positions edited. Twelve scope conditions target only zero gates.

This shows context-sensitive activation, not a general loss of water knowledge. Two effective input removals leave answers intact; the basic ice prompts do not activate T2375 at all. No matched-active-feature specificity controls were run.
