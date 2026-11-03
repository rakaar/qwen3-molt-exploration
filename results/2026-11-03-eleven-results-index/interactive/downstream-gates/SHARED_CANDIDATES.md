# Do the month candidates generalize to alphabet and Roman prompts?

L29-T2375 contribution subtracted only at final input is; later native layers run normally. L31-T797/T949 are observed offline, not ablated.

This is an independent CPU re-evaluation of previously captured paired native-model inputs. No new generation or GPU inference was needed.

## Final input ` is`

| Prompt | L31-T797 native -> off | L31-T949 native -> off |
|---|---|---|
| `the eleventh month of the year is` | 0.000000 -> 0.655309 | 0.000000 -> 0.562312 |
| `the eleventh letter of the english alphabet is` | 0.000000 -> 0.000000 | 0.000000 -> 0.000000 |
| `the eleventh positive integer in roman numerals is` | 0.000000 -> 0.000000 | 0.000000 -> 0.000000 |

A zero gate means the raw encoder score did not pass its learned threshold. It is not a rounded small positive activation.

| Case | Transform | Threshold | Native raw score | Ablated raw score |
|---|---|---:|---:|---:|
| month | L31-T797 | 0.526637 | 0.474743 | 0.655309 |
| month | L31-T949 | 0.551812 | 0.495338 | 0.562312 |
| alphabet | L31-T797 | 0.526637 | -0.956087 | -0.516661 |
| alphabet | L31-T949 | 0.551812 | -1.144418 | -0.978929 |
| roman | L31-T797 | 0.526637 | -0.761149 | -0.376841 |
| roman | L31-T949 | 0.551812 | -0.610160 | -0.352482 |

## All input positions

- month: L31-T797: native [], after ablation [7]; L31-T949: native [4, 5], after ablation [4, 5, 7].
- alphabet: L31-T797: native [], after ablation []; L31-T949: native [], after ablation [].
- roman: L31-T797: native [], after ablation []; L31-T949: native [], after ablation [].

Positions are zero-based. Earlier input captures were identical between conditions. No generated tokens were evaluated.

## Conclusion

Both candidates switch on only in the month case among these three prompts. They are not a shared gate-activation explanation at final is across these cases. A month-specific causal role remains untested; the later native MLPs were not replaced with MOLTs.

In particular, both gates stay off throughout the alphabet and Roman input sequences, including final ` is`. Their raw scores increase after ablation but remain negative and below threshold.

The saved off first token is x, a prefix compatible with XII; this prefill-only check does not regenerate or score the full XII sequence.

## Verification

All six capture files and the L31 gate-weight archive passed SHA-256 checks. Independent FP64/FP32 evaluation agreed on every active/inactive decision. Final-token values match the original all-gate table within 1e-12.

Reproduce: `python3 experiments/eleven-downstream-gates-20261103/check_shared_candidates.py`.
