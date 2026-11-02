# Eleventh-to-fifth full MOLT contribution transplant

Recipient: `the fifth month of the year is`. Donor: `the eleventh month of the year is`.

Natural donor gate: 1.158553600. The donor UVx is frozen at the donor final input token and used unchanged in every transplant condition.

At recipient input position 6 (` is`), replace only this MOLT contribution with target_gate * donor_UVx. Retain the native recipient MLP remainder. No generated-token edit.

The exact natural full contribution gives initial word **may**, versus **may** for the scalar-only donor-value control.

Initial-November gates tested: [10, 15]. Lowest successful value tested: 10. This is not a global threshold.

| Donor-operator gate | Initial word | P(May variants) | P(November variants) | Contribution L2 |
|---:|---|---:|---:|---:|
| 0.000000 | may | 92.400844% | 0.004152% | 0.000000 |
| 0.500000 | may | 90.911315% | 0.008951% | 7.491184 |
| 1.000000 | may | 90.779833% | 0.014885% | 14.982367 |
| 1.158554 | may | 90.408722% | 0.018838% | 17.357875 |
| 2.000000 | may | 87.606338% | 0.063426% | 29.964734 |
| 3.000000 | may | 80.113183% | 0.283627% | 44.947101 |
| 5.000000 | may | 43.006843% | 3.962195% | 74.911835 |
| 10.000000 | November | 0.896528% | 19.687001% | 149.823671 |
| 15.000000 | November | 0.100508% | 20.187100% | 224.735506 |
| 20.000000 | 11 | 0.019708% | 13.702447% | 299.647342 |
| 30.000000 | 120 | 0.002786% | 11.736332% | 449.471013 |
| 40.000000 | eleventh | 0.001086% | 10.941800% | 599.294683 |
| 50.000000 | eleventh | 0.000258% | 5.822757% | 749.118353 |
| 60.000000 | eleventh | 0.000063% | 3.025414% | 898.942025 |
| 70.000000 | eleventh | 0.000019% | 1.747621% | 1048.765695 |
| 80.000000 | eleventh | 0.000006% | 1.250007% | 1198.589367 |
| 90.000000 | eleventh | 0.000003% | 1.064914% | 1348.413038 |

The probabilities sum two exact one-token spellings per month: ` may` plus ` May`, and ` november` plus ` November`. They exclude following punctuation/EOS and do not cover answers beginning with other words.

## Equal-norm direction controls

The donor operator is 2.505470 times the norm of the recipient operator. These controls use recipient UVx at an adjusted scalar to match the donor-patch intended vector norm.

Donor gate 1.158553600 versus recipient gate 2.902721808; intended L2 17.357876.

Donor-operator response (P(November variants) 0.018838%):

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

Equal-norm recipient-operator response (P(November variants) 0.020605%):

```text
 may. the fifth month of the year is may. the fifth month of the year is may. the fifth month of the year is may. the fifth month
```

Donor gate 10.000000000 versus recipient gate 25.054704480; intended L2 149.823672.

Donor-operator response (P(November variants) 19.687001%):

```text
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

Equal-norm recipient-operator response (P(November variants) 27.933983%):

```text
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

## Every donor-operator continuation

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
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

### Gate 15.000000000

```text
 November. is that true? no, that's not true. the fifth month of the year is may. may is the fifth month. so the answer is
```

### Gate 20.000000000

```text
 11. What is the value of the 11th month? - Brainly.com
profile
profile
jessicamcbride
```

### Gate 30.000000000

```text
 120 days. Is this statement true or false? Also, why?

The fifth month of the year is May. May has 31 days
```

### Gate 40.000000000

```text
 eleventh month of the year is the fifth month of the year is eleventh month of the year is the fifth month of the year is eleventh month of
```

### Gate 50.000000000

```text
 eleventh month of the year is the fifth month of the year is eleventh month of the year is the fifth month of the year is eleventh month of
```

### Gate 60.000000000

```text
 eleventh month of the year is the fifth month of the year is eleventh month of the year is the fifth month of the year is eleventh month of
```

### Gate 70.000000000

```text
 eleventh month of the year is the fifth month of the year is eleventh month of the year is the fifth month of the year is eleventh month of
```

### Gate 80.000000000

```text
 eleventh month of the year is the fifth month of the year is eleventh month of the year is the fifth month of the year is eleventh month of
```

### Gate 90.000000000

```text
 eleventh month of the year is the fifth month of the year is eleventh month of the year is the fifth month of the year is eleventh month of
```

## Verification and limits

Hash-verified pinned Qwen3-4B and MOLT weights; BF16 native model; FP32 contribution edits and one BF16 cast; eager attention; TF32 off; greedy decoding; 32-token limit.

The donor gate and native response controls reproduce saved references. Native/observer and zero controls are exact. The scalar-only donor-value control reproduces the preceding experiment exactly. Natural donor value, first initial-November value if any, and maximum gate are repeated exactly.

CPU reconstruction verifies gates, operators, frozen donor operator identity, mask, actual BF16 edits, probability calculations and equal intended norms. The recipient operator itself stays identical across conditions. All responses and source hashes are preserved.

This tests one prompt and one MOLT. Initial November is judged separately from subsequent self-correction or coherent full answers. Equal-norm controls match intended vectors before BF16 rounding; actual rounded edit norms are also retained.

Reproduce with `python3 experiments/eleven-to-five-full-20261103/analyze.py`. Evidence: `comparisons/2026-11-03-eleven-to-five-full/`.
