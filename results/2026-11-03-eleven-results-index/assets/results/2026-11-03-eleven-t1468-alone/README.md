# L34-T1468 alone off at final input is

Subtract only the natural L34-T1468 contribution at final input is. L29-T2375 untouched. Later native layers compute normally. No generated-token edits.

**0 of 3 initial answers changed from eleven to twelve.**

| Prompt family | Native first answer -> T1468 off | P(eleven answer), native -> off | P(twelve answer), native -> off |
|---|---|---:|---:|
| month | November -> November | 66.8686% -> 68.3161% | 24.3655% -> 25.4165% |
| alphabet | K -> K | 71.0745% -> 71.6458% | 1.0286% -> 0.9759% |
| roman | XI -> XI | 41.5773% -> 48.5013% | 3.5346% -> 3.6955% |

Probability definition: Sum of exact leading-space lower/title-case answer-prefix token-sequence probabilities. Multi-token candidates are teacher-forced from the corresponding prefill cache without additional edits. Punctuation, EOS and other spellings are excluded.

Roman XI/XII include every token of the listed prefixes, rather than using the probability of the shared-looking x prefix. Each candidate is scored from that condition’s prefill cache without further interventions.

All captured greedy token sequences identical to native after ablation: True.

A change in P(eleven) alone does not determine the eleven-versus-twelve preference, because both answer probabilities can move together. Exact changes in their log probability ratio are retained in summary.json.

## Month

Prompt: `the eleventh month of the year is`

Only L34-T1468 at input position 7 (` is`) was removed. Natural gate 1.289993; removed contribution L2 21.863927.

Native continuation:

```text
 november. the twelfth month is december. the first month is january. the second month is february. the
```

T1468-off continuation:

```text
 november. the twelfth month is december. the first month is january. the second month is february. the
```

## Alphabet

Prompt: `the eleventh letter of the english alphabet is`

Only L34-T1468 at input position 8 (` is`) was removed. Natural gate 1.084717; removed contribution L2 16.189846.

Native continuation:

```text
 k. the eleventh letter of the greek alphabet is kappa. the eleventh letter of the russian alphabet is �
```

T1468-off continuation:

```text
 k. the eleventh letter of the greek alphabet is kappa. the eleventh letter of the russian alphabet is �
```

## Roman

Prompt: `the eleventh positive integer in roman numerals is`

Only L34-T1468 at input position 9 (` is`) was removed. Natural gate 2.429366; removed contribution L2 62.700717.

Native continuation:

```text
 xi, but the eleventh letter of the alphabet is k. So, the answer is k. But wait, the
```

T1468-off continuation:

```text
 xi, but the eleventh letter of the alphabet is k. So, the answer is k. But wait, the
```

## Verification and limits

Native first logits and layer inputs reproduce the saved anchors. Native, read-only observer and zero-strength continuations/probabilities are exact. Each ablation repeats exactly. Independent CPU checks reproduce the MOLT contribution and the BF16 edit rounding; candidate probabilities are recomputed from saved logits.

- Three selected raw completion prefixes; no claim of generality.
- Ablation subtracts one MOLT approximation contribution from the native MLP; native MLPs are not replaced.
- Probabilities are specified answer-prefix events, including all tokens of XI/XII, not EOS-terminated answer probabilities.
- Full raw continuations are retained even when they repeat or hit the 24-token limit. Initial-answer changes are classified separately.

Code: experiments/eleven-t1468-alone-20261103/. Raw captures and source hashes are retained under comparisons/2026-11-03-eleven-t1468-alone/.
