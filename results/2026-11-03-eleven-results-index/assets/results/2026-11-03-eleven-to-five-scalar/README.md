# Eleventh-to-fifth scalar gate transplant and sweep

Prompt: `the fifth month of the year is`.

Donor: `the eleventh month of the year is`; natural final-input gate 1.158553600.

Only recipient input position 6 (` is`) is edited. The recipient’s own current UVx is retained; neither donor operator nor donor contribution is transferred. No generated token is edited.

Native initial answer: **may**. Natural recipient gate: 0.

Copying the donor scalar gives initial answer **may**.

November-initial gates: [20, 30, 40]. These are values tested, not a global threshold.

| Fixed gate | Initial word | P(May variants) | P(November variants) | Stop |
|---:|---|---:|---:|---|
| 0.000000 | may | 92.400844% | 0.004152% | token_limit |
| 0.500000 | may | 91.638121% | 0.005288% | token_limit |
| 1.000000 | may | 91.372531% | 0.006770% | token_limit |
| 1.158554 | may | 91.179289% | 0.007554% | token_limit |
| 2.000000 | may | 89.095610% | 0.012826% | token_limit |
| 3.000000 | may | 88.377216% | 0.019489% | token_limit |
| 5.000000 | may | 83.678918% | 0.067339% | token_limit |
| 10.000000 | may | 55.043788% | 1.294506% | token_limit |
| 15.000000 | The | 14.798084% | 9.619911% | token_limit |
| 20.000000 | November | 2.806989% | 20.543640% | token_limit |
| 30.000000 | November | 0.360070% | 29.411769% | token_limit |
| 40.000000 | November | 0.139710% | 20.885039% | token_limit |
| 50.000000 | the | 0.054489% | 12.350069% | token_limit |
| 60.000000 | the | 0.022355% | 9.449018% | token_limit |
| 70.000000 | the | 0.012282% | 10.333669% | token_limit |
| 80.000000 | 11 | 0.006854% | 9.841101% | token_limit |
| 90.000000 | 11 | 0.005248% | 9.464663% | token_limit |

Probabilities sum the lowercase and title-case leading-space one-token forms: ` may` plus ` May`, and ` november` plus ` November`. Following punctuation/EOS is not included. These events do not cover answers beginning with other text such as ` the month of may`. All four exact case-sensitive probabilities are retained in summary.json; lowercase probabilities and case sums are in sweep.csv.

## Full recorded responses

### Gate 0.000000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 0.500000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 1.000000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 1.158553600

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 2.000000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 3.000000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 5.000000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 10.000000000

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

### Gate 15.000000000

```text

The fifth month of the year is **May**. 

In the Gregorian calendar, the months are ordered as follows:
1. January  
2.
```

### Gate 20.000000000

```text
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

### Gate 30.000000000

```text
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

### Gate 40.000000000

```text
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

### Gate 50.000000000

```text
 the month of may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may.
```

### Gate 60.000000000

```text
 the month of may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may.
```

### Gate 70.000000000

```text
 the month of may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may.
```

### Gate 80.000000000

```text
 11. What is the value of the 11th month?

Okay, so the question is: "The fifth month of the year is 
```

### Gate 90.000000000

```text
 11. What is the value of the 11th month?

Okay, so the question is: "The fifth month of the year is 
```

## Checks and limits

Pinned native BF16 Qwen3-4B; FP32 gate/operator and contribution edit; one BF16 cast; eager attention; TF32 off; raw lowercase prefixes; greedy decoding; 32 generated-token cap.

Native donor and recipient responses reproduce their previously saved first 12 tokens. Native/observer, zero-strength and gate-zero controls match exactly. Donor-scalar and maximum-gate repetitions match exactly. An independent CPU audit verifies hashes, gates, operators, full recipient-operator identity, intervention mask, exact BF16 edits and probability reconstruction.

This is one prompt and one feature. A high-gate initial-word flip is not a guarantee of coherent full continuation, ordinary-strength sufficiency or broad ordinal transfer. Full continuations and failures are preserved.

Reproduce local analysis: `python3 experiments/eleven-to-five-scalar-20261103/analyze.py`. Scientific evidence: `comparisons/2026-11-03-eleven-to-five-scalar/`.

An independent follow-up reloaded the model and repeated gates 0, 1.158553600 and 20. All generated token IDs, every recorded full-vocabulary logit hash and the gate-20 self-correction reproduced exactly. This follow-up used no new strengths.
