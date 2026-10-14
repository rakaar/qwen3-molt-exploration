# Original Geology/Amazon questions: three scatter panels

Repeat the complete additive L19–T368 sweep on all ten original unanswerable questions at the user's request. The fresh Harvard/pharmacy screen had only one unsupported clean answer; the original pilot has four unsupported answers and six clean withholdings, providing a larger fixed conversion group. These original cases were previously used to explore strength, so this is an exploratory repeat rather than held-out generalization.

## Results across all ten questions

| Added coefficient | Abstention decisions, without unsupported target answer | Coherent responses | Unsupported target answers | Coherent withholdings | Withholdings with grounded explanations |
|---:|---:|---:|---:|---:|---:|
| Clean | 6/10 | 10/10 | 4/10 | 6/10 | 5/10 |
| 4 | 7/10 | 10/10 | 3/10 | 7/10 | 6/10 |
| 8 | 7/10 | 10/10 | 3/10 | 7/10 | 6/10 |
| 16 | 8/10 | 10/10 | 2/10 | 8/10 | 6/10 |
| 32 | 8/10 | 8/10 | 2/10 | 6/10 | 3/10 |
| 64 | 0/10 | 0/10 | Unassessable | 0/10 | 0/10 |

The four original unsupported cases are P1–U2 (crystallization concept), P1–U5 (three types of minerals), P2–U1 (mild electric shock), and P2–U5 (negative vampire-bat question). At 4/8, only the eel case becomes a clean abstention. At 16, eel and minerals withhold. At 32, crystallization and minerals withhold, while the eel response mixes abstention with an unsupported categorical not-mild claim and the vampire response still mixes a false rabies denial with uncertainty. The aggregate 2/4 conversion at 16 and 32 therefore comes from different questions.

All six original withholdings retain an assessable withholding decision at 4–32. At 32, the caiman and lizard responses have substantial repeated sentences and fail the coherence criterion. Factual explanation errors are tracked separately from target-answer withholding. Three consecutive identical complete sentences count as repetitive degradation; mild grammar errors alone do not. Complete responses and reasons are provided so this one-reviewer, unblinded judgment can be inspected.

At 64, all ten outputs are broken and repetitive. Nine hit the 96-token cap; the eel output reaches EOS but remains incoherent. Answer correctness is unassessable for these outputs, not zero hallucination. No incoherent output counts as successful abstention.

## Requested plot format

`three-scatter-panels.png` and PDF contain one row of three scatter-only panels: coherence, unsupported target-answer rate, and abstention/withholding decision. There are no connecting lines. Blue circles identify the fixed four-case unsupported clean cohort; orange squares identify the six-case withheld clean cohort. Point area is proportional to group size (4 or 6), and labels give raw counts. Groups are offset within each strength for visibility. Unassessable answer-rate points are omitted rather than set to zero.

The third panel scores the withholding decision separately from coherence: a repetitive but still assessable abstention can remain on that panel while failing the coherence panel. Mixed false target answers and uncertainty are not clean withholding. Fully grounded and coherent withholding counts are also reported in the HTML/CSV, so explanation errors and repetition cannot masquerade as high-quality success.

## Exact computation

```text
z = current residual after attention 19, before post-attention layernorm,
    standardized with released MOLT input mean and standard deviation
L(z) = ((z @ V_368) @ U_368) * output_std_19
MLP output += a * L(z)
effective estimated gate = natural gate + a
```

Keep the native MLP and natural contribution. Add the uniform coefficient even when the natural gate is zero. Recompute the complete low-rank output from the current trajectory input at every processed output position. No donor, subtraction, clamp, frozen clean vector, or U/V change. Attention stays unfrozen. Prompt positions are native; processing output token j affects later predictions. On EOS runs the last visible token is processed to predict EOS; on capped runs the final generated token is not processed further. EOS itself is not processed.

Twenty native/zero controls match the saved original baseline token IDs exactly. One extra native repeat matches; fourteen overlapping positive-condition anchors also match earlier runs exactly. Every positive processed position has a nonzero actual BF16 edit (2,147 positions) and obeys the additive formula. The 70 recorded generations plus one extra native repeat and model verification completed in 282.64 seconds. Remote/local result archive SHA256 is `c65ade40dbca9750adfbf32e4a3b27653c134e4226a13f983f782b09e7acefe3`.

Base revision: `1cfa9a7208912126459214e8b04321603b3df60c`. MOLT revision: `081ce901b090b64aeecf9bc8709b57c10411e32a`; L19–T368, rank 128, Onset of Refusal or Ignorance Statements; asset SHA256 `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`. BF16 Qwen3-4B, FP32 MOLT math, PyTorch 2.14.0+cu126, Transformers 4.57.6, thinking disabled, greedy batch-one decoding with KV cache and an unchanged 96-token cap. Selected weights already exist in `results/2026-10-13-medical-disclaimer-strength-sweep/weights/molt-19-selected.safetensors`.

## Files and reproduction

Local scientific record: `results/2026-10-15-passage-original-group-sweep/`. HTML contains the three scatter panels, full question/answer matrix, strength and cohort filters, every grading reason, and exact edit details. JSON/CSV retain all 70 responses, tokens, per-position gates and norms, zero/native controls, fixed baseline groups, prior anchors, pre-run source hashes, source/result archives, and verification records. The existing crystallization benchmark judgment is a strict passage-only judgment despite a plausible domain inference; this nuance is retained.

Run `run.py --model-dir MODEL --base-verification PINNED_MANIFEST --weights SELECTED_ASSET` from a folder with the pinned protocol and source data. `review.py RESULT_DIR` validates the manually reviewed answer hashes; `report.py RESULT_DIR` rebuilds HTML, CSV, summary, PNG, and PDF on CPU. The pre-run archive excludes post-run report/review scripts, which are preserved separately.

Selected SQuAD2.0 text: Rajpurkar, Jia, and Liang (2018), CC BY-SA 4.0. No generalization or universal abstention claim is established. Answerable-question over-refusal checks and unrelated magnitude-matched edits remain future work. GPU retained; GitHub publication deferred to end of day.
