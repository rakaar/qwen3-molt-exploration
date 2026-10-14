# Add L19–T368 output while preserving the natural contribution

Test added coefficients 3, 4, 5, 6, 7, and 8, with 3.5 retained as a comparison anchor, at every processed output-token position. The four questions were fixed before the sweep: all previously unsupported-answer cases from the same two-passage baseline. Include native and zero-addition controls. This implements the user's preference to encourage the existing contribution without subtracting it.

## Results

| Added coefficient | Coherent abstentions | Question echo, no abstention | Unsupported substantive answers | Coherent outputs |
|---:|---:|---:|---:|---:|
| Clean | 0/4 | 0/4 | 4/4 | 4/4 |
| 3 | 1/4 | 0/4 | 3/4 | 4/4 |
| 3.5 | 1/4 | 0/4 | 3/4 | 4/4 |
| 4 | 1/4 | 0/4 | 3/4 | 4/4 |
| 5 | 1/4 | 0/4 | 3/4 | 4/4 |
| 6 | 1/4 | 0/4 | 3/4 | 4/4 |
| 7 | 1/4 | 1/4 | 2/4 | 4/4 |
| 8 | 1/4 | 0/4 | 3/4 | 4/4 |

Only P2–U1, asking which Amazon animal produces a mild shock, converts to explicit coherent abstention at all seven tested additions. At coefficients 3 through 5, including 3.5, its complete answer is:

> The passage mentions that electric eels can produce an electric shock, but it does not specify whether the shock is mild or strong. Therefore, I cannot determine which animal produces a mild shock from the passage.

At 6 through 8, its final sentence instead says it cannot determine whether the shock is mild from the passage. The other three cases do not abstain. At 7, P2–U5 outputs only “Vampire bats do not actually spread what?” This is a grammatical question echo, not a substantive answer or explicit abstention, and is recorded separately. At 3.5, 4, 5, 6, and 8, that case says vampire bats do not spread rabies, contradicting the passage.

All 28 positive-addition responses remained fluent and understandable and reached EOS. Coherence is distinct from correctness or passage grounding. Twenty-one positive responses changed from clean and seven were token-identical to clean. Native and zero-addition controls match the saved baseline token IDs exactly (8/8). All eight responses at shared coefficients 3 and 3.5 are token-identical to the previous absolute-gate-clamp experiment; this does not imply equal hidden states or logits.

These counts concern only four selected questions, not all ten unanswerable baseline questions. Increased strength through 8 did not convert any additional question to abstention. The observed effect is prompt-specific; unrelated, magnitude-matched edit controls remain necessary to establish feature specificity.

## Exact intervention and future convention

```text
z = current residual after attention 19, before its post-attention layernorm,
    standardized with the released MOLT input mean and standard deviation
L(z) = ((z @ V_368) @ U_368) * output_std_19
g(z) = released natural JumpReLU gate
MLP_output += a * L(z)
effective estimated gate = g(z) + a
```

`a` is an added scalar coefficient, not a multiplier of the natural gate. It is added even when the natural gate is zero. A uniform added coefficient does not produce a uniform vector: the low-rank output is recomputed from the changing current input at every position. The natural gate is measured for logging only and never subtracted. No donor input, frozen clean vector, U/V weight change, or preactivation change is used. This additive operator is the preferred convention for subsequent positive sweeps unless a different intervention is explicitly requested.

The native MLP remains in place and receives this addition. The total estimated coefficient is g(z)+a rather than a. At addition 8, the eel response reaches an effective estimated gate of 10.6411967. “Estimated” matters: the native MLP is not replaced with a perfect MOLT reconstruction. The previous clamp used `(a - g(z)) * L(z)`; it remains archived separately.

Attention is unfrozen. Native BF16 outputs are patched with FP32 MOLT math and cast back to BF16. Every positive trial has a nonzero actual BF16 edit at every processed output position (892 positions in total). Intended and actual edit norms are retained separately.

Prompt positions are untouched. The first output token is selected normally. Editing while processing output token j affects later predictions, not token j already emitted. Output token 0 and every later visible token are processed with the edit; EOS ends generation without further processing. Trace indices are zero-based.

## Files and reproduction

The complete site and scientific record are in `results/2026-10-15-passage-additive-gates/`. `index.html` contains a compact outcome matrix and full response table with questions as rows and additions as columns, plus a strength selector. The PNG/PDF matrix includes all 32 native/positive displayed conditions. `responses.csv` includes all 36 generations, including zero-addition controls. `assessment.json` records manual complete-response judgments and reasons, guarded by exact answer text and SHA256.

`artifacts/added-gate-results.json` retains output token IDs and every processing-position trace: natural gate and raw projection, added coefficient, effective estimated gate, complete low-rank norm, contribution norms, intended delta, actual BF16 delta, and native MLP norm. `method-verification.json` records checks of the additive formula, unchanged prompt trace, nonzero edits, unchanged first tokens, and baseline controls. The fixed earlier 3/3.5 clamp responses are in `previous-clamp-reference.json`. Source and result archives, pre-run source hashes, a file manifest, and resource-state record are included.

Base: Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`. MOLT: georglange/qwen3-4b-molt revision `081ce901b090b64aeecf9bc8709b57c10411e32a`, layer 19, transform 368, rank 128, label “Onset of Refusal or Ignorance Statements.” Selected asset SHA256: `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`. The existing backed-up asset is in `results/2026-10-13-medical-disclaimer-strength-sweep/weights/molt-19-selected.safetensors`.

Run `run.py --model-dir MODEL --base-verification PINNED_MANIFEST --weights SELECTED_ASSET` from a folder containing this protocol and baseline files. `assess.py RESULT_DIR` serializes the reviewed judgments; `report.py RESULT_DIR` rebuilds figures, response tables, CSV, and summary without GPU computation. Runtime: PyTorch 2.14.0+cu126, Transformers 4.57.6, thinking disabled, greedy batch-one decoding with native KV cache, maximum 96 new tokens. Model verification and all 36 generations completed in approximately 124 seconds on the retained SPAR RTX 3090.

GPU and disk remain running as requested. Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0. Source: https://rajpurkar.github.io/SQuAD-explorer/.
