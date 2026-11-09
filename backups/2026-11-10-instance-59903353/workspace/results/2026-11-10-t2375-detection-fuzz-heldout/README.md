# T2375: complete held-out Detection and Fuzz comparison

Both frozen descriptions were evaluated on all 100 saved Detection cases and all 100 saved Fuzz cases, with the exact cached judge and inference settings used for the earlier local Active Fuzz run.

| Test | Original correct | Revised correct | Improved | Worsened | Net extra correct |
|---|---:|---:|---:|---:|---:|
| Detection | 79/100 | 80/100 | 1 | 0 | +1 |
| Fuzz | 78/100 | 80/100 | 2 | 0 | +2 |
| Active Fuzz (previous run) | 78/100 | 79/100 | 2 | 1 | +1 |

Improved means the original description was judged incorrectly and the revised description correctly. Worsened means the reverse. Net is improved minus worsened.

## Evaluation integrity

- All 100 predefined cases per method were included: 50 activating and 50 inactive labels. Both arms were freshly evaluated, for 400 classifications total.
- Both explanation strings match their original frozen SHA-256 hashes. No rewriting, case selection, or tuning followed test feedback.
- No additional manual test-text inspection occurred before scoring in this run. Test content was read only programmatically for tokenization and judge inference; logs showed progress counts.
- Historical example demonstrations and Active Fuzz feedback had already been viewed in earlier work. The candidate stayed unchanged throughout; these are saved benchmark cases, not a claim that their text has never been seen.
- The native judge weights were re-hashed against the prior Active Fuzz model manifest before loading. The runtime versions, GPU, model revision and generation settings match the earlier run.
- Both original and revised arms use identical example text, highlight masks, few-shot examples, rendering and decoding settings. Only the explanation differs.
- No prompt truncation, forced classification repair, or selective response retry was used.
- These metrics assess gate-condition descriptions, not the causal computation of the transform.

## Per-method details

### detection

Paired valid responses: 100/100. Parse failures: original 0, revised 0.

| Count | Original | Revised |
|---|---:|---:|
| True positives | 29 | 30 |
| False negatives | 21 | 20 |
| True negatives | 50 | 50 |
| False positives | 0 | 0 |

Exact paired discordance test, two-sided and unadjusted: p = 1. Treat this as an exploratory comparison on one saved set and judge.

### fuzz

Paired valid responses: 100/100. Parse failures: original 0, revised 0.

| Count | Original | Revised |
|---|---:|---:|
| True positives | 28 | 30 |
| False negatives | 22 | 20 |
| True negatives | 50 | 50 |
| False positives | 0 | 0 |

Exact paired discordance test, two-sided and unadjusted: p = 0.5. Treat this as an exploratory comparison on one saved set and judge.

## Judge and artifacts

- Judge: `vrfai/Qwen3.8-27B-NVFP4` at `1561c70b22784dba7c6943b3271fc5f185adda01`.
- Runtime: vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2.
- RTX 3090, Marlin NVFP4 weight-only kernels with BF16 activations, greedy decoding, thinking disabled, 32 output-token limit.
- Same 4,096-token context limit, one active sequence, seed 42, eager execution, disabled prefix caching and FlashInfer sampler as Active Fuzz.
- The previous 32-case Detection/Fuzz gains were training diagnostics with the dashboard judge. They are separate from this 100-case comparison.
- This is the same fresh local judge as our Active Fuzz result; historical dashboard judge scores remain separate.
- Repeated request consistency with Active Fuzz: 100/100 shared requests have matching predictions.
- Raw requests and responses: audit.jsonl. Full tokenized inputs: prepared-inputs.json. Per-case decisions: summary.json.
- Protocol, executed source, environment freeze, hash manifests and independent verification are included in this directory.
