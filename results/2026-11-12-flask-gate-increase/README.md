# T2375 all-input scalar sweep: one fixed oil-flask prompt

```text
Dax's flask contains oil.
Neri's flask contains water.

What does Dax's flask contain? Reply with one word.
```

Native answer and every tested fixed-gate answer: **Oil.** No clean water answer at any tested strength.

| Scalar gate at all input positions | Response | P(water) | P(oil) | Edited input positions |
|---|---|---:|---:|---:|
| Native | Oil. | 2.88232e-11% | 99.999208% | 0 |
| 0 | Oil. | 2.24476e-11% | 99.999207% | 1 |
| 2 | Oil. | 3.26610e-11% | 99.999214% | 39 |
| 4 | Oil. | 2.54365e-11% | 99.999384% | 39 |
| 8 | Oil. | 2.88233e-11% | 99.999384% | 39 |
| 16 | Oil. | 4.75215e-11% | 99.999391% | 39 |
| 32 | Oil. | 8.42327e-10% | 99.999189% | 39 |
| 64 | Oil. | 9.22564e-07% | 99.972304% | 39 |

The input contains 39 tokens, including special/template tokens and the complete assistant answer prefix. Every position is selected without activation-based filtering. The only naturally positive input gate is at the supplied ` water` token, value 0.8252674.

The intervention replaces the scalar by the specified strength: add `(strength - natural_gate) * local_operator` to native layer-29 MLP output in FP32, then cast once to BF16. The full local operator is input-dependent. No generated or candidate-continuation tokens are edited, and the later native computation proceeds normally. This is contribution editing, not full-MLP replacement.

Probabilities sum disjoint exact complete water/oil output-token prefixes over lowercase/titlecase with zero or one leading space. Punctuation/EOS is excluded; per-variant word-plus-EOS scores remain in the raw data. The supplied prompt is never retokenized for candidate scoring.

Pinned Qwen3-4B and L29 MOLT weights; BF16 native model, FP32 observer/edit, eager attention, TF32 off, greedy, thinking off, max 32 output tokens. The single native prompt must answer oil correctly before interventions. Native/zero observations and probability scores match exactly and reproduce the previous flask experiment exactly. A hash-locked baseline is saved before the frozen sweep 0, 2, 4, 8, 16, 32, 64.

The independent CPU audit checks gates, local operators, selected positions, intended/actual edits and exact BF16 rounding at all 273 saved intervention positions. Positive strengths change all 39 inputs; gate zero removes the one naturally active contribution. No generated edits occur.

This is a one-prompt pilot. Water prefix probability increases at the highest strength but stays extremely small; the answer remains oil. No matched active-transform control or broader prompt evaluation was run. It does not identify T2375 as a general water-production switch.
