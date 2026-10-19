# L19–T368: complete twenty-question abstention comparison

[Full report](https://rakaar.github.io/qwen3-molt-exploration/results/2026-10-19-abstention-complete/index.html) · [Every measured response](../../../results/2026-10-19-abstention-complete/all-responses.csv)

Twenty fixed passage questions: ten the native model answered correctly, and ten selected cases where it supplied an answer unsupported by the passage. The second group contains the four original cases and six cases frozen after an 80-question screen. They are not a representative random sample.

## Outcome

| Steering scope | Made-up answer → clear abstention | Originally correct → false abstention |
| --- | --- | --- |
| Input only | 10/10 | 3/10 |
| Output only | 7/10 | 3/10 |
| Input + output | 10/10 | 7/10 |

These counts allow question-specific tuning over all saved tests. At the shared coefficient **+16**, input-only steering converts all ten made-up answers, causes two false abstentions among the correct answers, and keeps all twenty responses coherent. This coefficient was identified after inspecting this grid, not evaluated on held-out questions.

On the common grid, the median lowest successful tested coefficients for the made-up cases are +4 input-only, +12 output-only and +2 input + output. The output-only median includes its seven successes; three cases fail. Input editing often works at lower coefficients, but the experiment does not isolate why individual questions need different values. Higher values can restore an answer, cause question drift or break coherence.

## Exact intervention

```
z = current residual after attention 19, standardized with released input mean/std
L(z) = ((z @ V_368) @ U_368) * output_std_19
native_MLP_output += a * L(z)
effective estimated gate = natural_gate(z) + a
```

This adds a coefficient to the current-input low-rank contribution. It does not replace the absolute gate with a, multiply the natural gate by a, patch a donor, or replace the native MLP. The coefficient scales a vector that depends on the current residual, so equal coefficients need not produce equal vectors or edit norms across questions.

- Input only: all encoded prompt positions, including instructions, passage, question, chat markers and assistant prefix; retain the edited KV cache during native response generation.
- Output only: processed response positions; the first output token has already been emitted before the first output edit.
- Input + output: both sets of positions.

Attention is unfrozen. Thinking disabled, greedy batch one, 96-new-token cap, BF16 native model and FP32 MOLT calculations. All IDs are zero-based. The pinned protocol contains checkpoint, asset and runtime hashes.

## Records and grading

The completed common grid is `1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64` in every one of the sixty question/scope cells. Earlier fine sweeps are preserved: **1,176 unique positive trials**, including 369 new trials, plus forty fresh exact native/zero controls. The search uses sweeps and inspected intermediate values, not strict binary search.

Clear success requires coherent on-question withholding without the substantive requested answer. On missing-answer questions, false extra claims disqualify a conversion. On answer-present questions, a false claim that the passage lacks the answer is the induced false abstention, and is explicitly flagged. Review is by Codex with conditions visible, not blinded independent human grading.

The table displays the lowest successful **tested** coefficient. For failures it displays the last coherent tested coefficient below the first tested incoherent one, or the highest tested coherent value if no breakdown was reached. Later recovery is reported separately. No monotonicity is assumed.

## Inspect and rebuild

From the repository root, with Python 3 and Matplotlib installed:

```bash
python3 experiments/passage-uncertainty/abstention-complete/verify.py
python3 experiments/passage-uncertainty/abstention-complete/build.py
python3 -m http.server 8787 --directory results --bind 127.0.0.1
```

`prepare.py` was used once to freeze the executed cohort and jobs; it refuses to overwrite the completed study. `run.py` is the exact executed GPU runner. Existing result files are not a request to repeat paid inference. The browser QA script records the original local Chrome/Puppeteer paths; adapt those paths if running it elsewhere.

The public backup includes compact responses/reviews, complete new token traces, the exact historical raw/review files referenced by `import-manifest.json`, source and SHA-256 manifests. SSH helpers, host-key files and provider account snapshots remain local. Large model weights remain outside this publication.

Selected SQuAD2.0 text: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0. These results demonstrate intervention effects on withholding behavior, including false abstention; they do not establish a unique or reliable uncertainty detector.
