# T2375: 16-token lookahead diagnostic

The same 12 confirmed pre-eleven passages were rerun after appending the next 16 original subject-model source tokens. Both frozen descriptions, fixed few-shots, labels, highlights and judge settings were preserved.

| Test | Old: prefix only | Old: +16 tokens | New: prefix only | New: +16 tokens |
|---|---:|---:|---:|---:|
| detection | 0/12 | 0/12 | 0/12 | 10/12 |
| fuzz | 0/12 | 0/12 | 0/12 | 11/12 |
| active_fuzz | 0/12 | 0/12 | 0/12 | 11/12 |

These are positive-case identification rates, not accuracy on a balanced classification benchmark. Fuzz and Active Fuzz have identical positive prompts and produced identical positive predictions; they are not independent replications.

## Active Fuzz inactive-highlight controls

- baseline: false positives 0/12 before, 0/12 after lookahead.
- candidate: false positives 0/12 before, 1/12 after lookahead.

The control highlights remain at their original zero-gate positions. The newly appended continuation is unhighlighted. Because the subject model is causal, future context does not change the already measured gates at the original positions.

## Scope and provenance

- The shared 12 contexts comprise seven before list item 11, one before the date-range endpoint 11, two before November, and two before the 11 suffix in 2011. No XI or K cases are present.
- This is a post-hoc diagnostic on inspected cases, not a new blind held-out result. Showing a future continuation changes the original prefix-only evaluation task.
- Detection and Fuzz each use the 12 positive cases. Active Fuzz uses those same 12 positive cases plus their 12 original inactive-highlight controls. Two descriptions give 96 logical requests and 72 unique request bodies.
- Prefix-only counts come from the completed previous same-judge runs. They were not rerun for this diagnostic.
- Judge: vrfai/Qwen3.8-27B-NVFP4 at 1561c70b22784dba7c6943b3271fc5f185adda01; same verified cached weights, vLLM 0.33.0, Marlin W4A16 on RTX 3090, temperature 0, thinking disabled, max output 32.
- All judge responses were valid and untruncated. Source, request and response audits, runtime comparison and verified backup receipts are retained.

## Per-case positive decisions

| Test | Original row | Old +16 | New +16 |
|---|---:|---:|---:|
| detection | 9 | 0 | 0 |
| detection | 13 | 0 | 1 |
| detection | 24 | 0 | 1 |
| detection | 26 | 0 | 0 |
| detection | 47 | 0 | 1 |
| detection | 63 | 0 | 1 |
| detection | 71 | 0 | 1 |
| detection | 72 | 0 | 1 |
| detection | 73 | 0 | 1 |
| detection | 78 | 0 | 1 |
| detection | 79 | 0 | 1 |
| detection | 89 | 0 | 1 |
| fuzz | 13 | 0 | 1 |
| fuzz | 39 | 0 | 1 |
| fuzz | 45 | 0 | 1 |
| fuzz | 47 | 0 | 1 |
| fuzz | 49 | 0 | 1 |
| fuzz | 55 | 0 | 1 |
| fuzz | 57 | 0 | 1 |
| fuzz | 68 | 0 | 1 |
| fuzz | 69 | 0 | 1 |
| fuzz | 86 | 0 | 1 |
| fuzz | 90 | 0 | 1 |
| fuzz | 98 | 0 | 0 |
| active_fuzz | 13 | 0 | 1 |
| active_fuzz | 39 | 0 | 1 |
| active_fuzz | 45 | 0 | 1 |
| active_fuzz | 47 | 0 | 1 |
| active_fuzz | 49 | 0 | 1 |
| active_fuzz | 55 | 0 | 1 |
| active_fuzz | 57 | 0 | 1 |
| active_fuzz | 68 | 0 | 1 |
| active_fuzz | 69 | 0 | 1 |
| active_fuzz | 86 | 0 | 1 |
| active_fuzz | 90 | 0 | 1 |
| active_fuzz | 98 | 0 | 0 |

A decision of 1 is correct for these positive cases. See heldout-plan.json and audit.jsonl for the exact full messages and generated answers.
