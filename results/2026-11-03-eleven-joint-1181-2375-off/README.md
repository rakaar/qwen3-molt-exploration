# Joint T1181/T2375 ablation on the eleventh-month prompt

Prompt: `the eleventh month of the year is`. Only final input position 7 (` is`) edited. Natural contributions removed independently or together; no generated-token edit.

| Condition | Initial word | P(November) | P(December) | P(January) |
|---|---|---:|---:|---:|
| Native | november | 66.868638% | 24.365519% | 0.018897% |
| T1181 off | november | 88.910875% | 3.447449% | 0.033657% |
| T2375 off | december | 8.831334% | 83.677527% | 0.030717% |
| Both off | november | 46.606528% | 40.801294% | 0.218613% |

Probabilities sum exact leading-space lowercase/title-case complete answer prefixes, excluding following punctuation/EOS and other answer formats. Multi-token month names use complete sequence probabilities. All twelve months are retained in month-probabilities.csv.

## Full recorded responses

### Native

```text
 november. the twelfth month is december. the first month is january. the second month is february. the third month is march. the fourth month
```

### T1181 off

```text
 november. the twelfth month is december. the first month is january. the second month is february. the third month is march. the fourth month
```

### T2375 off

```text
 december. the twelfth month is january. the tenth month is september. the ninth month is august. the eighth month is july. the seventh month
```

### Both off

```text
 november. the twelfth month is december. the first month is january. the second month is february. the third month is march. the fourth month
```

## Method and validation

Same pinned native BF16 Qwen3-4B and MOLT checkpoint. Contributions use native pre-MLP residual and checkpoint input/output standardizers. For the joint edit, sum c1181 and c2375 in FP32, subtract from native local MLP output, then cast once to BF16. No sequential BF16 edits or extra vector normalization. Native MLP remainder and normal downstream computation retained.

Native, observer and zero-strength controls match exactly. Fresh native/single-ablation runs reproduce all earlier generated token IDs and full-vocabulary logit hashes. Joint ablation repeats exactly. Independent CPU auditing reconstructs every candidate score, both gates/operators, selected-position masks and exact FP32-sum/BF16 edit. Raw archive/files are hash-verified.

One exact lowercase raw prefix; greedy 32-token continuations, eager attention, TF32 off. This tests dependence of this December result on T1181, not a universal ordinal circuit or direct connection between the MOLTs.

```json
{
  "status": "pass",
  "arrays_audited": 7,
  "edit_arrays_audited": 5,
  "probability_max_absolute_error": 6.057564768680024e-07,
  "native_observer_zero_exact": true,
  "native_and_both_single_references_exact": true,
  "joint_repeat_exact": true,
  "both_contribution_vectors_identical_across_conditions": true,
  "FP32_sum_before_single_BF16_cast_exact": true,
  "native_local_MLP_same_across_conditions": true,
  "only_final_input_edited": true,
  "no_generated_token_edits": true,
  "all_feature_hashes_verified": true
}
```

Reproduce: `python3 experiments/eleven-joint-1181-2375-off-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-eleven-joint-1181-2375-off/`. No publication or GPU cleanup was performed.
