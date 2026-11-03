# Can native T1468 rescue L29-T2375 removal?

Only two possible edit sites: L29-T2375 and L34-T1468, each at final input is during prefill. Every later native attention/MLP runs normally. No generated-token edits.

Initial eleven answers restored: scalar gate 0/3; full contribution 0/3.

Two rescues are tested separately. Scalar rescue: `delta34 = (native_phi34 - current_phi34) * current_operator34`. Full-contribution rescue: `delta34 = native_contribution34 - current_contribution34`. Both use that same prompt’s native donor at final input ` is`, without extra amplification.

| Prompt | Condition | Initial answer | P(11th answer) | P(12th answer) |
|---|---|---|---:|---:|
| month | Native | November | 66.8686% | 24.3655% |
| month | L29-T2375 off | December | 8.8313% | 83.6775% |
| month | T2375 off + native T1468 scalar gate | December | 7.8346% | 84.1178% |
| month | T2375 off + full native T1468 contribution | December | 8.4809% | 82.3089% |
| alphabet | Native | K | 71.0745% | 1.0286% |
| alphabet | L29-T2375 off | L | 2.6467% | 32.7474% |
| alphabet | T2375 off + native T1468 scalar gate | L | 2.5838% | 34.4925% |
| alphabet | T2375 off + full native T1468 contribution | L | 2.5989% | 32.1569% |
| roman | Native | XI | 41.5773% | 3.5346% |
| roman | L29-T2375 off | XII | 12.3393% | 15.0408% |
| roman | T2375 off + native T1468 scalar gate | XII | 12.3722% | 13.2319% |
| roman | T2375 off + full native T1468 contribution | XII | 11.9808% | 12.7874% |

Probability definition: Sum of exact leading-space lower/title-case answer-prefix token-sequence probabilities. Multi-token candidates are teacher-forced from the corresponding prefill cache without additional edits. Punctuation, EOS and other spellings are excluded.

Roman XI/XII probabilities include every token in those answer prefixes. Native and off references reproduce exactly. Model computations after the patched sites are recomputed normally.

## Month

Prompt: `the eleventh month of the year is`; final input position 7 (` is`).

T1468 gate: native 1.289993; after T2375 removal 0.717599.

Native:

```text
 november. the twelfth month is december. the first month is january. the second month is february. the
```

L29-T2375 off:

```text
 december. the twelfth month is january. the tenth month is september. the ninth month is august. the eighth
```

T2375 off + native T1468 scalar gate:

```text
 december. the twelfth month is january. the tenth month is september. the ninth month is august. the eighth
```

P(11th answer) change vs T2375-off: -0.9967 percentage points; P(12th): +0.4403 points. Log-odds recovery fraction: -0.038364 (0 = off baseline; 1 = native).

T2375 off + full native T1468 contribution:

```text
 december. the twelfth month is january. the tenth month is september. the ninth month is august. the eighth
```

P(11th answer) change vs T2375-off: -0.3504 percentage points; P(12th): -1.3686 points. Log-odds recovery fraction: -0.007366 (0 = off baseline; 1 = native).

## Alphabet

Prompt: `the eleventh letter of the english alphabet is`; final input position 8 (` is`).

T1468 gate: native 1.084717; after T2375 removal 0.681166.

Native:

```text
 k. the eleventh letter of the greek alphabet is kappa. the eleventh letter of the russian alphabet is �
```

L29-T2375 off:

```text
 l. the eleventh letter of the english alphabet is l. the eleventh letter of the english alphabet is l.
```

T2375 off + native T1468 scalar gate:

```text
 l. the eleventh letter of the english alphabet is l. the eleventh letter of the english alphabet is l.
```

P(11th answer) change vs T2375-off: -0.0629 percentage points; P(12th): +1.7451 points. Log-odds recovery fraction: -0.011251 (0 = off baseline; 1 = native).

T2375 off + full native T1468 contribution:

```text
 l. the eleventh letter of the english alphabet is l. the eleventh letter of the english alphabet is l.
```

P(11th answer) change vs T2375-off: -0.0477 percentage points; P(12th): -0.5905 points. Log-odds recovery fraction: 0.000000 (0 = off baseline; 1 = native).

## Roman

Prompt: `the eleventh positive integer in roman numerals is`; final input position 9 (` is`).

T1468 gate: native 2.429366; after T2375 removal 1.319434.

Native:

```text
 xi, but the eleventh letter of the alphabet is k. So, the answer is k. But wait, the
```

L29-T2375 off:

```text
 xii, but the eleventh letter of the alphabet is k. So, the eleventh positive integer is xii
```

T2375 off + native T1468 scalar gate:

```text
 xii, but the eleventh letter of the alphabet is k. So, the eleventh positive integer is xii
```

P(11th answer) change vs T2375-off: +0.0329 percentage points; P(12th): -1.8089 points. Log-odds recovery fraction: 0.049118 (0 = off baseline; 1 = native).

T2375 off + full native T1468 contribution:

```text
 xii, but the eleventh letter of the alphabet is k. So, the eleventh positive integer is xii
```

P(11th answer) change vs T2375-off: -0.3585 percentage points; P(12th): -2.2534 points. Log-odds recovery fraction: 0.049879 (0 = off baseline; 1 = native).

## Verification and limits

The native unhooked run, observed run and zero-strength two-site edits agree exactly. T2375-off and a zero-strength L34 rescue agree exactly. Both rescue variants and T2375-off repeat exactly. Independent CPU checks reproduce each MOLT and every BF16 patch; candidate probabilities are recomputed from saved logits. Earlier input positions remain unchanged.

- Three selected raw prefixes; not a generality test.
- The interventions patch MOLT approximation contributions within native MLP outputs. No MLP is replaced wholesale.
- Gate rescue preserves the changed-input operator; full-contribution rescue changes both effective gate and operator contribution.
- Answer-prefix probabilities sum the specified lower/uppercase leading-space spellings, excluding punctuation and EOS.
- Correlative coactivation and standalone ablation did not establish a uniformly positive eleven-support role for T1468.

Source: experiments/eleven-t1468-rescue-20261103/. Raw arrays, manifests and source hashes: comparisons/2026-11-03-eleven-t1468-rescue/.
