# Correct answer → false abstention: ten answer-present passage questions

Qwen3-4B correctly answers all ten fixed Geology/Amazon questions natively. Positive additive steering of **L19–T368**, **Onset of Refusal or Ignorance Statements** (rank 128), produces coherent full false abstention in **3/10** cases in this bounded strength search. This is a selected, per-question-tuned feasibility result, not a held-out policy accuracy estimate.

| Question | Native supported answer | Lowest successful tested addition | Representative addition |
|---|---|---:|---:|
| Animal producing a deadly shock | electric eel | 35 | 40 |
| Fish known to bite humans | piranha | 28 | 28 |
| Bat type spreading rabies | vampire bats | 26 | 26 |

The eel +35 answer includes an unnecessary meta-comment and an invented warning rationale; +40 is shown as a cleaner representative. None is a proven global minimum. All three cases use false passage-absence rationales: that is the over-abstention being tested, rather than evidence of accurate uncertainty or deleted knowledge. There is no additional success without such an erroneous explanation.

No full coherent conversion was found for the five Geology questions, black-caiman question, or dart-frog secretion question at the tested values through +64. Those answers remain present, gain caveats, or deteriorate. Mixed correct-answer-plus-refusal is not counted as full conversion. A token limit alone does not determine coherence; complete texts and review reasons are retained.

## Cohort and search

All ten answerable questions from the original two-passage SQuAD2.0 pilot are frozen before steering; each original native answer was reviewed as correct. All twenty new native/zero controls and the native repeat reproduce the earlier output token IDs exactly.

- Coarse additions on every case: 1, 2, 4, 8, 16, 24, 32, 48, 64 (90 trials).
- Intermediate additions on every case: 12, 20, 28, 36, 40, 44, 56 (70).
- Case-specific fine samples around observed answer/denial/deterioration transitions (65).
- **225 positive trials + 20 native/zero controls = 245 generations**, plus one identical native repeat.
- At most 24 positive values per case; successful cases use 19. Behavior is nonmonotonic, so this is coarse/fine sampling, not strict binary search. All ten remain in the denominator.

## Exact intervention

```
z = current post-attention residual, standardized using released MOLT input mean/std
L(z) = ((z @ V_368) @ U_368) * released output_std_19
native MLP output += a * L(z)
effective estimated gate = natural_gate(z) + a
```

This is additive coefficient steering, not fixed gate replacement or multiplication of the natural gate. Native MLP retained; no donor; current-trajectory low-rank output recomputed at every processed response token. No prompt edit; the first answer token remains native. Attention unfrozen. Greedy batch one, thinking disabled, 96-new-token cap, KV cache, BF16 native model, FP32 MOLT math. The result does not test steering the final prompt position before the initial answer choice.

Same pinned Qwen3-4B, MOLT, selected asset, prompt template and runtime as the earlier 7/10 unanswerable study. Questions and search budgets differ, so the two fractions are not a representative head-to-head comparison. See `protocol.json` for exact revisions and asset hash.

## Verification, artifacts and local rebuild

`verification.json` confirms all 20 exact controls, all first output tokens native, source/job hashes, complete answer-bound reviews, and **9,751 actual nonzero edited response positions**. A separate download audit confirms all 34 remote scientific files and all four raw job files against remote hashes. Reviews are by Codex with conditions visible, not an independent blinded human grader. No unrelated-transform controls were added.

From the workspace root:

```
python3 experiments/passage-uncertainty/answerable-abstention/verify.py results/2026-10-19-answerable-abstention
python3 experiments/passage-uncertainty/answerable-abstention/build.py results/2026-10-19-answerable-abstention
```

The local result folder contains full passages and references, every baseline/steered answer, per-question selected strengths, a categorical scatter plot, CSV, original raw job files, source/archive hashes, manual review scripts and GPU status. Open `http://127.0.0.1:8787/2026-10-19-answerable-abstention/index.html` while the preview server is running.

Inference uses `run.py --job JOB --model-dir PINNED_MODEL_DIRECTORY --weights SELECTED_ASSET`. Existing outputs are never overwritten. Review scripts retain the exact human-readable judgments; novel responses require new review rather than silent automatic grading. Full base-model/runtime caches are redownloadable and omitted from result export. Selected SQuAD2.0 text: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0; dataset hash and original source records are in `cohort.json`.

SPAR account 710735, working RTX 3090 instance **56698356**, **$0.2216667/hour including 30 GB disk**, plus transfers. User authorized a $10 active setup/testing cap and retention afterward. Failed pre-SSH rental 56697893 was deleted only after explicit user approval. The working GPU remains running and idle for follow-ups; no lifecycle action was taken on it. Saved endpoints are historical: verify live provider state before future SSH.
