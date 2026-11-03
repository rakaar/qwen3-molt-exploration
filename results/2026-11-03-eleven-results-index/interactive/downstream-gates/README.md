# Downstream MOLT gates after L29-T2375 removal

Final input is in native and L29-T2375-off runs; L30-L35 encoders are observed offline. Every later native MLP runs normally. No later gate or MLP is replaced.

Native MLPs are unchanged. Each later MOLT encoder receives its own layer's post-attention residual before native post_attention_layernorm, using checkpoint mean/std. Gate values are scored offline in FP64; independent FP32 evaluation gives identical active sets.

Labels are cached from 21 October and describe activation hypotheses. This is a gate comparison, not a measurement of full MOLT contributions or later-transform causal effects.

## Month

Prompt: `the eleventh month of the year is`. Only input position 7 (` is`) is directly edited at L29.

| Layer | Native active gates | After L29-T2375 off | Largest gate changes |
|---|---|---|---|
| L30 | T15 (1.105), T433 (0.566), T858 (0.972), T949 (0.556), T1724 (0.945) | T15 (1.105), T858 (0.936), T949 (0.545), T1724 (0.893) | T433: 0.566 -> 0.000 (-0.566); Temporal and Calendar Token Gate; T1724: 0.945 -> 0.893 (-0.052); Range Hyphens and Sentence Punctuation; T858: 0.972 -> 0.936 (-0.036); Prepositions framing temporal and spatial contexts; T949: 0.556 -> 0.545 (-0.011); Temporal and Spatial Prepositions; T15: 1.105 -> 1.105 (+0.000); User Token Newline Gate |
| L31 | T135 (1.321), T760 (0.529), T2294 (2.023) | T135 (1.322), T760 (0.531), T797 (0.655), T949 (0.562), T2294 (2.033) | T797: 0.000 -> 0.655 (+0.655); Winter Weather Context Gate; T949: 0.000 -> 0.562 (+0.562); Temporal and Financial Timeframe Prepositions; T2294: 2.023 -> 2.033 (+0.010); Temporal and Spatial Context Prepositions; T760: 0.529 -> 0.531 (+0.002); Final Token of Titles and Headers; T135: 1.321 -> 1.322 (+0.000); User Prompt and Response Start |
| L32 | T15 (1.123), T224 (0.705), T335 (3.268), T962 (1.068), T1181 (0.829) | T15 (1.114), T224 (0.707), T335 (3.251), T962 (1.191), T1181 (0.785) | T962: 1.068 -> 1.191 (+0.122); Food Ingredient and Substitute Nouns; T1181: 0.829 -> 0.785 (-0.044); Seasonal Holiday Token Activation; T335: 3.268 -> 3.251 (-0.017); Temporal Reference Token Activation; T15: 1.123 -> 1.114 (-0.009); End-of-Turn Token Detection; T224: 0.705 -> 0.707 (+0.002); Prepositions Introducing Specific Entities |
| L33 | T422 (1.590), T587 (1.125), T795 (1.187), T909 (1.251), T968 (0.892) | T422 (1.580), T587 (0.945), T795 (1.203), T909 (1.173), T968 (0.862) | T587: 1.125 -> 0.945 (-0.180); List and Sequence Token Activation; T909: 1.251 -> 1.173 (-0.077); Proper Noun and Entity Token Activation; T968: 0.892 -> 0.862 (-0.030); Copula Verb Gate Activation; T795: 1.187 -> 1.203 (+0.016); Multi-word phrase and idiom detection; T422: 1.590 -> 1.580 (-0.010); Temporal Marker Gate |
| L34 | T65 (0.795), T90 (1.552), T159 (1.845), T391 (1.445), T666 (0.794), T1267 (0.711), T1468 (1.290), T1593 (2.087) | T65 (0.763), T84 (0.587), T90 (1.496), T159 (1.836), T391 (1.454), T666 (0.737), T1267 (0.682), T1468 (0.718), T1593 (2.119) | T84: 0.000 -> 0.587 (+0.587); Connector Tokens in Compounds and Idioms; T1468: 1.290 -> 0.718 (-0.572); Article "An" Preceding Noun; T666: 0.794 -> 0.737 (-0.058); Copula Verb Main Clause Trigger; T90: 1.552 -> 1.496 (-0.055); Proper Noun and Entity Reference Gate; T65: 0.795 -> 0.763 (-0.032); Punctuation and Format Nouns Trigger |
| L35 | T111 (1.107), T253 (1.723), T1716 (1.266) | T111 (1.101), T253 (1.720), T1716 (1.283) | T1716: 1.266 -> 1.283 (+0.017); Proper Noun and Location Gate; T111: 1.107 -> 1.101 (-0.006); Temporal Preposition and Date Gate; T253: 1.723 -> 1.720 (-0.003); Prompt Start Token Detection |

Exact newly on/off sets and full labels, raw scores, thresholds and all active-gate values are in summary.json and the interactive table.

## Alphabet

Prompt: `the eleventh letter of the english alphabet is`. Only input position 8 (` is`) is directly edited at L29.

| Layer | Native active gates | After L29-T2375 off | Largest gate changes |
|---|---|---|---|
| L30 | T15 (1.084), T241 (0.653), T1724 (0.892) | T15 (1.085), T241 (0.768), T1724 (0.731) | T1724: 0.892 -> 0.731 (-0.162); Range Hyphens and Sentence Punctuation; T241: 0.653 -> 0.768 (+0.115); Linguistic Structure and Phonetic Mechanics; T15: 1.084 -> 1.085 (+0.001); User Token Newline Gate |
| L31 | T135 (1.221), T438 (0.683), T626 (1.169), T760 (0.863) | T135 (1.245), T438 (0.782), T626 (1.114), T760 (0.887) | T438: 0.683 -> 0.782 (+0.099); Code and Formula Syntax Token Gate; T626: 1.169 -> 1.114 (-0.055); Character Classification and String Manipulation; T760: 0.863 -> 0.887 (+0.024); Final Token of Titles and Headers; T135: 1.221 -> 1.245 (+0.024); User Prompt and Response Start |
| L32 | T15 (1.015), T438 (1.809), T1181 (0.648) | T15 (1.050), T438 (1.944) | T1181: 0.648 -> 0.000 (-0.648); Seasonal Holiday Token Activation; T438: 1.809 -> 1.944 (+0.135); Structural Delimiters and Classification Labels; T15: 1.015 -> 1.050 (+0.034); End-of-Turn Token Detection |
| L33 | T14 (0.999), T157 (0.702), T438 (2.016), T795 (1.127), T909 (0.848) | T14 (0.529), T157 (0.648), T391 (0.614), T438 (2.210), T587 (0.638), T795 (1.122), T909 (0.793), T968 (0.623) | T587: 0.000 -> 0.638 (+0.638); List and Sequence Token Activation; T968: 0.000 -> 0.623 (+0.623); Copula Verb Gate Activation; T391: 0.000 -> 0.614 (+0.614); List and Enumeration Token Detection; T14: 0.999 -> 0.529 (-0.470); Proper Noun and Acronym Gate; T438: 2.016 -> 2.210 (+0.193); Structural Delimiters and Formatting Tokens |
| L34 | T90 (0.966), T159 (1.928), T241 (1.991), T666 (0.647), T965 (1.258), T1440 (1.037), T1468 (1.085), T1475 (0.699) | T90 (0.834), T159 (1.902), T241 (1.919), T1440 (1.691), T1468 (0.681), T1475 (0.733) | T965: 1.258 -> 0.000 (-1.258); End-of-Sequence Completion Signal; T1440: 1.037 -> 1.691 (+0.654); Suffix and Acronym Token Gate; T666: 0.647 -> 0.000 (-0.647); Copula Verb Main Clause Trigger; T1468: 1.085 -> 0.681 (-0.404); Article "An" Preceding Noun; T90: 0.966 -> 0.834 (-0.132); Proper Noun and Entity Reference Gate |
| L35 | T200 (0.911), T253 (1.812), T760 (0.920), T1221 (0.580), T1269 (0.831), T2117 (0.889) | T200 (0.844), T253 (1.805), T1177 (0.534), T1221 (0.652), T1269 (0.561), T2117 (1.432) | T760: 0.920 -> 0.000 (-0.920); Mid-Word Proper Noun and Spec Tokens; T2117: 0.889 -> 1.432 (+0.543); Proper Noun First-Token Gate; T1177: 0.000 -> 0.534 (+0.534); Proper Noun Termination Gate; T1269: 0.831 -> 0.561 (-0.270); Multi-Token Proper Noun and Acronym Activation; T1221: 0.580 -> 0.652 (+0.072); Multi-token proper noun and brand activation |

Exact newly on/off sets and full labels, raw scores, thresholds and all active-gate values are in summary.json and the interactive table.

## Roman

Prompt: `the eleventh positive integer in roman numerals is`. Only input position 9 (` is`) is directly edited at L29.

| Layer | Native active gates | After L29-T2375 off | Largest gate changes |
|---|---|---|---|
| L30 | T15 (1.071), T412 (0.751), T1724 (1.394), T2088 (1.019) | T15 (1.069), T412 (0.709), T1724 (1.210), T2088 (1.060) | T1724: 1.394 -> 1.210 (-0.184); Range Hyphens and Sentence Punctuation; T412: 0.751 -> 0.709 (-0.042); Copula Verb Gate; T2088: 1.019 -> 1.060 (+0.042); Product Size Availability Tokens; T15: 1.071 -> 1.069 (-0.002); User Token Newline Gate |
| L31 | T50 (0.540), T135 (1.127), T250 (1.778), T760 (0.577), T1798 (0.845) | T135 (1.150), T250 (1.744), T760 (0.624), T1798 (0.805) | T50: 0.540 -> 0.000 (-0.540); Technical and Mathematical Token Gate; T760: 0.577 -> 0.624 (+0.047); Final Token of Titles and Headers; T1798: 0.845 -> 0.805 (-0.040); Thresholds and Division Triggers; T250: 1.778 -> 1.744 (-0.034); Preceding Tokens for Numerical Values; T135: 1.127 -> 1.150 (+0.023); User Prompt and Response Start |
| L32 | T15 (0.934), T190 (0.642), T438 (1.964), T1082 (2.174), T1164 (1.123), T1465 (1.436), T2401 (0.884) | T15 (0.934), T190 (0.732), T438 (2.162), T979 (0.513), T1082 (2.416), T1164 (1.363), T1465 (1.587), T2401 (0.975) | T979: 0.000 -> 0.513 (+0.513); Final Token of Proper Noun; T1082: 2.174 -> 2.416 (+0.242); Numeric Sequence and Range Token Gate; T1164: 1.123 -> 1.363 (+0.241); Nouns, Articles, and Suffixes; T438: 1.964 -> 2.162 (+0.197); Structural Delimiters and Classification Labels; T1465: 1.436 -> 1.587 (+0.152); Syntactic Bridge Function Words |
| L33 | T157 (0.953), T383 (1.040), T438 (1.003), T490 (1.832), T587 (2.363), T795 (0.745), T909 (1.390), T968 (0.722), T1873 (0.741) | T157 (0.930), T383 (1.281), T438 (1.141), T490 (1.862), T587 (2.574), T795 (0.645), T909 (1.144), T968 (0.788) | T1873: 0.741 -> 0.000 (-0.741); Syntactic Closure Point; T909: 1.390 -> 1.144 (-0.246); Proper Noun and Entity Token Activation; T383: 1.040 -> 1.281 (+0.241); Suffix Tokens Completing Multi-Token Entities; T587: 2.363 -> 2.574 (+0.212); List and Sequence Token Activation; T438: 1.003 -> 1.141 (+0.138); Structural Delimiters and Formatting Tokens |
| L34 | T90 (1.226), T157 (1.522), T159 (1.937), T241 (1.225), T1468 (2.429), T1837 (1.187) | T90 (1.034), T157 (1.575), T159 (1.899), T241 (1.108), T1468 (1.319) | T1837: 1.187 -> 0.000 (-1.187); Proper Noun and Entity Start Trigger; T1468: 2.429 -> 1.319 (-1.110); Article "An" Preceding Noun; T90: 1.226 -> 1.034 (-0.192); Proper Noun and Entity Reference Gate; T241: 1.225 -> 1.108 (-0.116); Mathematical and Code Symbol Gate; T157: 1.522 -> 1.575 (+0.054); Proper Nouns for Software and Platforms |
| L35 | T111 (1.720), T253 (1.868), T254 (0.618), T314 (0.581), T1278 (0.699) | T111 (2.001), T253 (1.835), T254 (0.630), T314 (0.518), T1221 (0.701), T1278 (0.606) | T1221: 0.000 -> 0.701 (+0.701); Multi-token proper noun and brand activation; T111: 1.720 -> 2.001 (+0.281); Temporal Preposition and Date Gate; T1278: 0.699 -> 0.606 (-0.093); Multi-token word and proper noun suffixes; T314: 0.581 -> 0.518 (-0.063); Scattered Weak Activations on General Text Structure; T253: 1.868 -> 1.835 (-0.032); Prompt Start Token Detection |

Exact newly on/off sets and full labels, raw scores, thresholds and all active-gate values are in summary.json and the interactive table.

## Verification

{
  "status": "pass",
  "captured_array_files_hash_verified": 15,
  "native_observer_zero_and_off_repeat_exact": true,
  "all_block_residuals_match_previous_study_exactly": true,
  "earlier_input_states_exactly_unchanged": true,
  "native_MLPs_replaced": false,
  "gate_layers": 6,
  "transforms_per_layer": 2480,
  "case_layer_transform_pairs": 44640,
  "FP32_FP64_membership_checks": 89280,
  "FP32_FP64_membership_disagreements": 0,
  "FP32_FP64_raw_max_absolute_error": 6.6268289287130244e-06,
  "gate_cache_and_original_checkpoint_tensor_hashes_verified": true
}

All 44,640 case/layer/transform pairs are retained in all-gate-values.csv, including gates inactive in both conditions. Active-gate-changes.csv retains the full active union. No new native-model inference was required for the offline gate scoring, after the paired input captures.

Reproduce: `python3 experiments/eleven-downstream-gates-20261103/analyze.py`, then `python3 experiments/eleven-downstream-gates-20261103/render.py`.

These observations identify candidate downstream gates for follow-up; they do not establish which one mediates the answer change.
