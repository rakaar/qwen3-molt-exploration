# L29–T2375: paired Active Fuzz with a local judge

**10 November update:** [Detection and Fuzz are now evaluated on all 100 held-out cases each with this same judge](../2026-11-10-t2375-detection-fuzz-heldout/README.md). Detection improved by 1 correct case and Fuzz by 2, with no regressions. The linked report compares all three methods and verifies unchanged descriptions and judge settings.

Old explanation: **78/100 (78.0%)**. Revised explanation: **79/100 (79.0%)**.

Paired changes: 2 corrected and 1 regressed; 100/100 pairs have valid responses. Baseline parse failures: 0; candidate parse failures: 0.

The observed gain is only one percentage point. This run does not provide convincing evidence that the revised description improves Active Fuzz performance.

This is a fresh paired evaluation with a pinned public judge. It does not reproduce the original dashboard judge checkpoint/runtime or replace its historical 77/100 Active Fuzz score. Both descriptions were rerun under the same new judge; comparing this candidate directly with the historical 77% would be invalid.

## What was evaluated

- The same 100 saved original Active Fuzz cases: 50 positive, 50 negative.
- Positive highlighted tokens have positive MOLT gate activation.
- Negative highlighted tokens have zero activation, while another position in the passage has positive activation. These locations were independently audited before scoring.
- Only the description changes between the old and revised arms. The examples, highlights, few-shots, labels, decoding settings and judge are identical.
- The original and revised descriptions remain frozen. No edits or example selection were made after reading these new judge outputs.
- Direct offline vLLM inference: no dashboard, remote judge API, or subject-model generation is required.
- Actual judge decisions are preserved in audit.jsonl, including full messages, input token IDs, output token IDs and request hashes.

| Metric | Original description | Revised description |
|---|---:|---:|
| Accuracy | 78.0% | 79.0% |
| True positives / 50 | 28 | 30 |
| False negatives / 50 | 22 | 20 |
| True negatives / 50 | 50 | 49 |
| False positives / 50 | 0 | 1 |

Exact two-sided paired discordance test: p = 1. This is a small evaluation using one fresh judge, not evidence of a causal mechanism.

## Judge identity and arithmetic

- Repository: `vrfai/Qwen3.8-27B-NVFP4`.
- Revision: `1561c70b22784dba7c6943b3271fc5f185adda01`.
- NVFP4 checkpoint on an Ampere RTX 3090; Marlin uses weight-only FP4 with BF16 activations. This differs from native Blackwell W4A4 arithmetic.
- vLLM 0.33.0; torch 2.14.0+cu130; transformers 5.20.2.
- Temperature 0, maximum 32 output tokens, thinking disabled, 4,096-token context; no input truncation.
- Optional FlashInfer sampling was disabled because the runtime image lacks its CUDA compiler path; all evaluated requests use the same greedy decoding.
- Model files and downloaded result artifacts were SHA-256 checked. See judge-verification.json and the rental backup record.

## Changed decisions

| Case | Marked text excerpt | Label | Old judge decision | Revised judge decision | Outcome |
|---|---|---|---|---|
| 17 | `ce. And so when we look on November<< the>>` | 1 | False | True | corrected |
| 24 | `the "Scramble for Africa", the high<<-water>>` | 1 | False | True | corrected |
| 71 | `ith just the right amount of liquid<<,>> it can be water` | 0 | False | True | regressed |

## Descriptions

### Original

The token "water" (or "Water") appearing in diverse contexts, often as a list item, ingredient, or environmental factor, with high activation strength.

### Revised

Activates on "water"/"Water" across lists, ingredients, and environmental descriptions, and on tokens at the boundary before eleven-related continuations. The second pattern includes contexts eliciting 11/eleven, November as the eleventh month, K as the eleventh English letter, or XI as the eleventh positive integer. Activation may occur on a preceding word, separator, or assistant-prefix token rather than the number itself; it is wording-dependent and sometimes appears for neighboring ordinals.
