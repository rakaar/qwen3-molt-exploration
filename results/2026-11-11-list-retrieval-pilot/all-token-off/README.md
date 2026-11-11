# All-token T2375 removal on correctly answered eleventh-word list requests

Switched to the twelfth word: 0/7. Retained the eleventh word: 7/7. Exact response changes: 0/7.

All processed input tokens and generated tokens, including numbered-list reading and chat answer prefix. Dynamic gate values on each trajectory. Native MLP retained; subtract only phi(z) * ((z @ V @ U) * output_std).

P11/P12 are products over all tokens of the exact lowercase word continuation with no added whitespace. Following punctuation/EOS and formatting variants are excluded. Exact-word-then-any-allowed-EOS probabilities are separately retained.

| Case | 11th / 12th target | Native response | Gate-off response | P11 native | P11 off | P12 native | P12 off |
|---|---|---|---|---:|---:|---:|---:|
| list-1-numbered-11 | pencil / drum | pencil | pencil | 0.999982476 | 0.999991536 | 2.14069288e-17 | 8.71705242e-18 |
| list-1-bullets-11 | pencil / drum | pencil | pencil | 0.999993801 | 0.999996066 | 2.34263296e-15 | 3.20200083e-15 |
| list-2-numbered-11 | shoe / hammer | shoe | shoe | 0.9991653 | 0.999827624 | 5.78454515e-12 | 1.00586911e-12 |
| list-3-numbered-11 | ring / apple | ring | ring | 0.999894619 | 0.999923944 | 7.99111749e-11 | 9.54430892e-12 |
| list-4-numbered-11 | ticket / spoon | ticket | ticket | 0.999927402 | 0.999915242 | 9.35674204e-14 | 1.96126522e-14 |
| list-5-numbered-11 | hammer / ticket | hammer | hammer | 0.992513359 | 0.995638847 | 3.82191301e-09 | 2.3254072e-09 |
| list-5-bullets-11 | hammer / ticket | hammer | hammer | 0.999747574 | 0.999752462 | 6.75655665e-09 | 5.96266947e-09 |

The probabilities above are fractions, not percentages. All seven cases were selected by native eleventh-word correctness before seeing edits. The corresponding native twelfth-word question succeeds in five numbered cases; it fails in both retained bullet cases. All unhooked and zero-strength decoding outputs/logits exactly match the saved native baselines.

Removed 18 positive input contributions. The policy remained active during generation; 0 generated-token and 0 terminal-token positive contributions were encountered. CPU reconstruction and exact BF16 output-rounding checks passed.

The separate month calibration changes a November-leading completion into a December-leading completion with final-input-only removal. This is an implementation calibration, not an additional list case.
