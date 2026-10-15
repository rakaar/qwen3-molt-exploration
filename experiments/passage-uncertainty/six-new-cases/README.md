# Six additional passage-grounding failures

This study screens fresh SQuAD2.0 development questions with the same passage-only instruction, Qwen3-4B checkpoint, deterministic decoding and L19–T368 additive intervention as the original four-case study. Native questions are reviewed first; the first six clear unsupported answers in stored order are frozen before any steering. Every selected case stays in the denominator.

80 unique questions were screened, plus three token-identical native repeats. The first screen was drawn from previously prepared but unused heldout metadata, now repurposed for exploratory screening; later screens use a deterministic new shuffle (seed 20261015). Pre-screen wording audits exclude ambiguous or apparently answerable benchmark items. All outputs and conservative post-screen exclusions are retained. Eight clear passage-unsupported answers were found; the first six are selected and the remaining two are not steered.

“Model made up an answer” here means that the supplied passage does not support the substantive answer. This does not necessarily mean the fact is false in the world. In particular, the selected respiration answer, carbon dioxide, is correct general knowledge absent from the passage. The experiment tests adherence to source grounding and conversion to abstention.

## Results

Four of six new cases convert to clear, coherent, grounded abstentions: carbon monoxide (+16), ship cargo (+4), economic mobility (+16), and antigen binding (+8). These are successful tested coefficients, not established minima. Combined with the original 3/4, the descriptive total is 7/10.

No clear success was found for Poland growth or respiration at sixteen tested positive strengths each. Poland +36/+40 are readable but have ambiguous scope/self-correction; they are conservatively excluded rather than labeled incoherent. Respiration retains a substantive answer, and at larger coefficients gives incorrect oxygen claims. Full wording is retained for independent judgment.

80 steering/control generations: 68 positive and 12 native/zero controls. All twelve controls exactly match their original native token IDs. All 3,335 processed positive output positions receive nonzero actual BF16 edits. Every first output token remains unchanged. The 30-file remote science archive was checksum verified. Complete review is hash-bound.

## Intervention

Layer 19, transform 368, rank 128; zero-based IDs. Label: Onset of Refusal or Ignorance Statements.

```
z = current residual after attention 19, standardized using released MOLT input mean/std
low_rank_output(z) = ((z @ V_368) @ U_368) * output_std_19
native_MLP_output += a * low_rank_output(z)
effective estimated MOLT gate = natural_gate(z) + a
```

The native MLP remains in the model. No donor or subtraction is used. This adds a coefficient; it does not multiply the natural gate or replace it with an absolute target. Current-trajectory low-rank outputs are recomputed at every processed output-token position. Prompt tokens remain native, so the first output token is predicted normally and only later predictions are affected. Attention stays unfrozen. EOS and the final capped token are not processed again.

Pinned Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`; MOLT revision `081ce901b090b64aeecf9bc8709b57c10411e32a`. Selected MOLT asset SHA256 `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`. Native model BF16, MOLT calculations FP32, greedy batch one, KV cache, thinking disabled, 96-token cap. PyTorch 2.14.0+cu126 and Transformers 4.57.6.

## Search and grading

Each case receives native and zero-addition controls, then the common coarse additions 1, 2, 4, 8, 16, 24, 32, 48, 64. At most seven further positive strengths may refine a successful interval or inspect a coherent near-miss, without assuming monotonicity. Maximum sixteen positive strengths per case. Successful tested values are not proven minima.

Success requires coherent explicit withholding, correctly grounded in the passage, without an invented substantive answer, false explanation, question echo, or repetitive collapse. A supported false-premise correction is not a made-up native answer, but merely correcting a premise under steering is not itself an explicit abstention. Mixed wrong answer plus uncertainty fails. Severely incoherent content can be indeterminate rather than classified as no hallucination.

Complete-response review is by Codex, with condition labels visible; no independent blinded human review. Reviews are bound to exact answer hashes. The original 3/4, new k/6, and combined (3+k)/10 describe individually tuned outcomes. The original four used different search budgets. These figures are not a universal-strength or heldout-policy benchmark. Answerable-question and matched unrelated-transform controls are not part of this study.

## Files and reproduction

`run.py` executes a job using the cached pinned checkpoint and selected MOLT asset. `protocol.json` is saved before native screening. `pre-screen-audit*.json` and `candidates*.json` record the candidate pool. `screening-review.json` documents every native answer. `lock_cohort.py` writes the six-case lock and coarse job before intervention. `merge.py`, `verify.py`, and `report.py` merge raw jobs, validate controls and actual edits, and render the HTML/CSV/three-panel scatter figure from saved outputs and manual judgments.

SPAR instance 56070285 is reused and retained running by explicit user request, at $0.175/hour including disk, plus transfers. No new rental or lifecycle action is performed. Canonical model/runtime files remain cached remotely; only scientific results and manifests need backup.

Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia and Percy Liang (2018), CC BY-SA 4.0. Dataset URL, verified SHA256 and actual three-round sampling are preserved in `dataset-provenance.json`; `dataset-source.json` retains the original earlier-pool metadata unchanged.
