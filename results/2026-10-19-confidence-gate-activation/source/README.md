# Native confidence-candidate gate activations

Twelve fixed original-pilot QA cases: four correctly answered, all four original made-up answers, and four clean natural abstentions. Each group contains two questions from each passage. The cases were frozen before measuring these gates.

Features: L11–T652 (First-Person Certainty Verbs), L13–T2128 (Confidence and Certainty Assurance), L6–T1751 (Confidence-Preposition Gate).

All three gates are zero throughout the twelve QA prompts and answers. A separate unsteered repetition control, “I know the answer, and I am confident in it,” activates all three in the response. It is excluded from QA-group averages. This screens out these candidates as useful gate-based confidence scores for these examples; it does not establish their irrelevance everywhere. Correlation on constant zero gate values is undefined.

No steering or MLP replacement occurs. Fresh greedy generation uses no hooks and exactly matches all twelve saved native token sequences. An observation-only replay preserves all 276 next-token predictions across the twelve QA cases and separate wording control. Input hooks read the residual entering post-attention layernorm and return None. Gate arithmetic uses released input mean/std, encoder row, bias and threshold at the pinned revision. Full model files are hash-verified. Mean activation includes zero gates and excludes special tokens; the final prompt position is recorded separately before the first answer prediction.

The report uses three candidate columns per question and one shared colour scale across all groups and the wording control. Each group has a PNG export. Desktop/mobile rendering checks compare every displayed response token and gate to the saved measurements.

From the repository root, for CPU-only inspection/rebuild:

```
python3 experiments/confidence-gate-activation/verify.py
python3 experiments/confidence-gate-activation/build.py
python3 -m http.server 8787 --directory results --bind 127.0.0.1
```

Open `/2026-10-19-confidence-gate-activation/index.html`. The included browser QA script records original local Chrome/Puppeteer paths; adapt them if running elsewhere. `prepare.py` downloads only small public gate slices, not full MOLT matrices. `run.py` is the exact executed GPU runner; do not rerun into an existing output directory. Full weights and cloud credentials are outside this result backup.

Artifacts are under `results/2026-10-19-confidence-gate-activation/`. Model/MOLT pins, fixed case IDs, original grades, range-request provenance, raw token scores, native token identities, verification and source hashes are preserved. The GPU remains retained under the user's prior instruction.

Selected SQuAD2.0 text: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0. “Made up” means unsupported by the passage; these are selected examples, not prevalence estimates. Candidate gate labels describe firing conditions and are not validated factual-confidence measurements.
