# T1181 off on the genuine twelfth-month prompt

Removing T1181 reduces the exact December-prefix probability from 88.032362% to 56.304216%, but December remains the initial answer and all 32 generated token IDs are unchanged. January rises from 0.574371% to 7.327308%. This is a confidence effect in this prompt, without an initial-answer shift.

Prompt: `the twelfth month of the year is`. Remove only L29–T1181 at final input position 7 (` is`). No other MOLT or generated token edited.

Natural gate: 1.489463329. Natural contribution L2: 24.459778. Actual BF16-rounded edit L2: 24.454293.

| Condition | Initial word | P(December) | P(November) | P(January) |
|---|---|---:|---:|---:|
| Native | december | 88.032362% | 0.092857% | 0.574371% |
| T1181 off | december | 56.304216% | 0.538871% | 7.327308% |

Probabilities sum exact leading-space lowercase/title-case answer prefixes, multiplying token probabilities for multi-token names. Other answer formats and following punctuation/EOS excluded. All twelve month probabilities are in month-probabilities.csv.

## Full native continuation

```text
 december. the twelfth month of the year is december. the twelfth month of the year is december. the twelfth month of the year is december.
```

## Full T1181-off continuation

```text
 december. the twelfth month of the year is december. the twelfth month of the year is december. the twelfth month of the year is december.
```

## Checks and limits

Pinned native BF16 Qwen3-4B; FP32 gate/operator and subtraction; one BF16 cast; normal downstream computation and normalization; eager attention; TF32 off; greedy 32-token continuations. The selected contribution retains checkpoint input/output standardizers and receives no extra normalization.

Native observation, zero-strength and exact natural-gate replay reproduce the native run bit-exactly in recorded full-vocabulary logit hashes and tokens. Native output reproduces the earlier first 32 tokens. Repeated ablation matches exactly. CPU auditing reconstructs probabilities, gate/operator values, selected-position masks and exact BF16 edits. Raw archive/files are hash-verified.

One exact raw completion prefix; this does not test numeric `12th` wording, all-input/output edits, or joint removal with T2375.

```json
{
  "status": "pass",
  "arrays_audited": 6,
  "edit_arrays_audited": 4,
  "probability_max_absolute_error": 1.6600223040619255e-07,
  "native_observer_zero_and_replay_exact": true,
  "ablation_repeat_exact": true,
  "native_prior_first_32_tokens_exact": true,
  "native_prior_probe_probabilities_match": true,
  "operator_same_across_conditions": true,
  "bf16_edits_exact": true,
  "only_final_input_edited": true,
  "no_generated_token_edits": true,
  "feature_hash_verified": true
}
```

Reproduce: `python3 experiments/t1181-twelfth-month-off-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-t1181-twelfth-month-off/`. No publication or GPU cleanup was performed.
