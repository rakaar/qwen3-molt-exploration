# Higher additive strengths on the three remaining questions

Test L19–T368, “Onset of Refusal or Ignorance Statements,” with added coefficients 8, 16, 24, and 32 on P1–U2, P1–U5, and P2–U5. These are all three non-abstaining cases from the preceding additive 3–8 sweep. The eel question is not retested. Keep the same pinned checkpoint, prompts, unfrozen attention, greedy decoding, output-token-only timing, and 96-token cap.

| Question | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| Crystallization | Unsupported answer | Unsupported answer | Coherent abstention | Coherent abstention |
| Three types of minerals | Unsupported answer | Coherent abstention | Coherent abstention | Abstention; explanation overclaims |
| Vampire bats | False rabies answer | False rabies answer | False answer + abstention language | False answer + uncertainty/self-correction |

Clean abstentions are 0/3 at 8, 1/3 at 16, and 2/3 at 24 and 32. All 12 positive-addition responses remain fluent and understandable and reach EOS. Fluency is separate from factual consistency: the vampire response at 32 is conflicted and its parenthetical self-correction is unclear. The minerals response at 32 abstains but falsely says the passage only discusses re-deposition. Mixed substantive answers and abstention language are not scored as clean abstention.

At 24, the crystallization answer is: “An important concept in crystallization is not mentioned in the passage. The passage does not explain what is important about crystallization.” At 16, the minerals answer ends: “Therefore, I cannot determine the answer from the passage.” These are the lowest successful tested values, not exact thresholds.

## Computation

```text
z = current post-attention-19 residual before post-attention layernorm,
    standardized with released MOLT input mean and standard deviation
L(z) = ((z @ V_368) @ U_368) * output_std_19
MLP output += a * L(z)
effective estimated gate = natural gate + a
```

Keep the native MLP and natural contribution. Recompute the complete low-rank output from the current input at each processed output position; add the uniform coefficient even if the natural gate is zero. No donor, subtraction, clamp, frozen clean input, or U/V changes. Attention remains unfrozen. Prompt tokens are untouched, so the first output token is selected normally; editing while processing token j affects later predictions. EOS stops generation and is not processed further.

Six native/zero controls reproduce the earlier baseline token IDs exactly. Three repeated addition-8 anchors reproduce the preceding sweep exactly. Every positive processing position receives a nonzero actual BF16 edit (389 positions); every trace obeys the additive equation. All 18 generations and base verification completed in 60.41 seconds on the retained SPAR RTX 3090. No GPU lifecycle action was performed.

Base: Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`. MOLT: georglange/qwen3-4b-molt revision `081ce901b090b64aeecf9bc8709b57c10411e32a`; layer 19, transform 368, rank 128; asset SHA256 `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`. Selected weights already exist in `results/2026-10-13-medical-disclaimer-strength-sweep/weights/molt-19-selected.safetensors`. PyTorch 2.14.0+cu126, Transformers 4.57.6, BF16 model, FP32 MOLT math, thinking disabled, batch one, KV cache.

## Local record and reproduction

The local result folder is `results/2026-10-15-passage-additive-high-strength/`. HTML includes complete answers and review reasons with strengths as columns; PNG shows the outcomes. CSV includes all 18 trials. Raw JSON preserves output tokens and every natural gate, effective estimated gate, position, and intended/actual edit norm. The fixed protocol, six pre-run source hashes, source/result archives, base verification, previous strength-8 anchor, and method checks are preserved.

Run `run.py --model-dir MODEL --base-verification PINNED_MANIFEST --weights SELECTED_ASSET` in a folder containing the protocol and baseline files. Run `assess.py RESULT_DIR` to serialize the exact manually reviewed judgments and `report.py RESULT_DIR` to rebuild HTML, PNG, CSV, and counts on CPU. The pre-run source archive excludes these post-run review/report scripts; the scripts are preserved separately.

This is a selected prompt-specific pilot. Answerable-question checks and unrelated magnitude-matched controls remain future work. Higher values do not prove a selective universal abstention mechanism. Earlier crystallization/minerals judgments retain the benchmark's strict passage-only criterion. Selected SQuAD2.0 text: Rajpurkar, Jia, and Liang (2018), CC BY-SA 4.0.

Publication is deferred to end of day at the user's request. Keep the GPU running.
