# Force L19–T368 on four unsupported-answer cases

Test absolute gate activations 0.5, 1, 1.5, 2, 2.5, 3, and 3.5 at every processed output-token position. The four questions were fixed before the sweep: all previously unsupported answers from the two-passage baseline. Native and forced-gate-zero controls are included.

## Results

| Forced gate | Coherent abstentions | Related supported fact, no abstention | Unsupported answers | Coherent outputs |
|---:|---:|---:|---:|---:|
| Clean | 0/4 | 0/4 | 4/4 | 4/4 |
| 0.5 | 0/4 | 0/4 | 4/4 | 4/4 |
| 1 | 0/4 | 0/4 | 4/4 | 4/4 |
| 1.5 | 0/4 | 1/4 | 3/4 | 4/4 |
| 2 | 0/4 | 1/4 | 3/4 | 4/4 |
| 2.5 | 0/4 | 1/4 | 3/4 | 4/4 |
| 3 | 1/4 | 0/4 | 3/4 | 4/4 |
| 3.5 | 1/4 | 0/4 | 3/4 | 4/4 |

Only P2–U1, asking which Amazon animal produces a mild shock, converts to explicit coherent abstention, at gate 3 and 3.5. Its complete answer is:

> The passage mentions that electric eels can produce an electric shock, but it does not specify whether the shock is mild or strong. Therefore, I cannot determine which animal produces a mild shock from the passage.

At 1.5–2.5, that case drops the unsupported mild claim and states a supported electric-eel fact, but does not abstain. This is marked separately and not counted as successful conversion. The other three cases continue giving unsupported answers at 3 and 3.5. At 3.5, P2–U5 says vampire bats do not spread rabies, explicitly contradicting the passage.

All 28 positive-gate trials remained fluent and understandable and reached EOS. Coherence is distinct from truth and passage grounding. Twenty positive-trial outputs were token-identical to clean; eight changed. All eight native/zero controls reproduced the earlier token IDs exactly. Every processed output-token position in every positive trial received a nonzero actual BF16 edit, including unchanged-output cases.

These rates concern only four selected questions. They are not updated scores on all ten unanswerable baseline questions. Gate 3 is the lowest tested value yielding abstention in the eel case, not a proven minimum. The earlier activation association does not establish a universal abstention effect; this sweep only demonstrates a conversion for one of four prompts in the tested range. Unrelated, magnitude-matched edit controls are still needed for feature-specificity claims.

## Exact intervention

```text
z = current pre-layernorm residual after attention 19, standardized with MOLT input mean/std
L(z) = ((z @ V_368) @ U_368) * output_std_19
g(z) = released natural JumpReLU gate
MLP_output += (a - g(z)) * L(z)
```

`a` is a forced post-nonlinearity scalar, not a multiplier or a preactivation. Gate 0.5 deliberately bypasses the learned threshold, 0.567491472. Both the natural gate and complete low-rank output are recomputed from each condition's changing current input. If a natural gate activates during the edited trajectory, its estimated contribution is subtracted before adding the forced one. No donor or clean frozen input is used, and U/V weights are unchanged.

The native MLP is retained and receives the estimated component delta. It is not replaced by a MOLT reconstruction. Attention remains unfrozen. BF16 model outputs are patched using FP32 MOLT math and cast back to BF16; intended and actual edit norms are saved separately.

The output-token-only edit starts while processing output token 0. All prompt positions are native, so the first output token is predicted normally. Editing token j influences later predictions. The last visible token is processed to predict EOS; EOS is not processed further. All indices in the traces are zero-based.

## Files and reproduction

The full site and scientific record are in `results/2026-10-15-passage-forced-gates/`. `index.html` contains a compact outcome matrix and a full response table with gate strengths as columns, questions as rows, and a strength selector. `forced-gate-outcomes.png` and PDF summarize all 32 displayed native/positive conditions. `responses.csv` includes all 36 trials, including forced-zero controls. `assessment.json` saves every reviewed judgment and reasoning, guarded by exact answer text/hash.

`artifacts/forced-gate-results.json` preserves generated token IDs and every processing-position trace: natural gate, forced scalar, raw projection, low-rank norm, contribution norms, intended delta, actual BF16 delta, and native MLP norm. `protocol.json` is the fixed pre-run protocol. The baseline passages, outputs, and earlier grading are retained verbatim.

Base: Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`. MOLT: georglange/qwen3-4b-molt revision `081ce901b090b64aeecf9bc8709b57c10411e32a`, layer 19, transform 368, rank 128, “Onset of Refusal or Ignorance Statements.” Selected asset SHA256: `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`. The existing asset is in `results/2026-10-13-medical-disclaimer-strength-sweep/weights/molt-19-selected.safetensors`.

Use `run.py --model-dir MODEL --base-verification PINNED_MANIFEST --weights SELECTED_ASSET` from a folder with this protocol and saved baseline files. The review serializer and report builder take the result directory as their argument. Runtime: PyTorch 2.14.0+cu126, Transformers 4.57.6, BF16 Qwen, thinking disabled, greedy decoding, batch size 1, native KV cache, maximum 96 new tokens. Hash verification and all 36 generations completed in about 96 seconds on retained SPAR RTX 3090 instance 55946393.

The GPU and disk remain running as requested. Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0. Source: https://rajpurkar.github.io/SQuAD-explorer/.
