# Fresh Harvard/pharmacy screen (retained scientific record)

Ten manually audited, original SQuAD2.0 unanswerable questions were frozen before model inference: Harvard paragraph 0 and Pharmacy paragraph 3. Native Qwen3-4B already withheld 9/10 answers (eight ordinary abstentions and one valid premise correction), with one unsupported negative answer about Dioscorides.

All ten questions were evaluated with added coefficients 4, 8, 16, 32, and 64, plus native and zero controls. The unsupported case did not become a clean abstention at 4–32; 32 mixed a false book denial with uncertainty. Already-withheld responses stayed intelligible and withheld at 4–32, but grounded explanations fell from 9/9 to 8/9 at 16 and 4/9 at 32. All ten coefficient-64 outputs were incoherent and capped at 96 tokens, so answer correctness was unassessable. The user requested switching the main study back to the original Geology/Amazon passages because this fresh conversion group had only one case.

The earlier two-panel line figures are retained as an exploratory screening record, not the main requested visualization. The subsequent `original-group-sweep` uses three scatter-only panels and fixed baseline cohorts of four unsupported answers and six withholdings.

The additive operator is `MLP output += a * ((z @ V_368) @ U_368) * output_std_19`, preserving the natural contribution. Standardize the current residual after attention 19 and before post-attention layernorm with the released MOLT mean/std. Recompute the complete low-rank output at every processed output position. No donor, subtraction, clamp, or U/V changes. Prompt tokens are native; processing output token j affects later predictions. EOS and the final generated token at a token cap are not processed further. Attention is unfrozen.

All ten zero controls and a native repeat match exactly. Every positive processed position receives a nonzero BF16 edit (2,315 positions). The 70 recorded responses plus one extra native repeat completed in 302.63 seconds. Remote/local result archive SHA256 is `24e23197be5ea27252c4faac40affac82e3b04e373869a93ca628b2f4de9e7b4`.

The result folder retains full responses, tokens, traces, manual judgments guarded by answer hashes, pre-run source hashes, source/result archives, fixed baseline cohorts, protocol, and report/review code. Run `review.py RESULT_DIR` then `report.py RESULT_DIR` to rebuild CPU artifacts. The selected weights are already preserved in `results/2026-10-13-medical-disclaimer-strength-sweep/weights/molt-19-selected.safetensors`.

Base revision: `1cfa9a7208912126459214e8b04321603b3df60c`. MOLT revision: `081ce901b090b64aeecf9bc8709b57c10411e32a`; L19–T368, rank 128, Onset of Refusal or Ignorance Statements; selected asset SHA256 `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`. PyTorch 2.14.0+cu126, Transformers 4.57.6, BF16 model/FP32 MOLT math, greedy batch one with KV cache, thinking disabled.

This screen has a one-question conversion cohort and cannot estimate generalization. Selected SQuAD2.0 text is CC BY-SA 4.0, attributed to Rajpurkar, Jia, and Liang (2018). GPU retained; publication deferred to end of day.
