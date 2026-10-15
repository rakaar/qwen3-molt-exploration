# From a made-up answer to “I don’t know”

Combined explanatory website and two posters for Qwen3-4B MOLT layer 19, transform 368, rank 128. Dashboard label: **Onset of Refusal or Ignorance Statements**. IDs are zero-based. This presentation combines saved measurements and generations; it does not add new model inference.

**7/10 clear, coherent, passage-grounded conversions**: original 3/4 plus new 4/6. The causal cohort consists of ten selected baseline failures, not ten random questions: four from the original pilot, plus six locked before steering after screening 80 new questions. The original natural pilot had four made-up answers and six natural abstention decisions. These are different groups for different purposes.

## Natural evidence

Six clean examples are selected from the original unsteered gate replay: three made-up answers (P1-U2, P1-U5, P2-U1), all mean gate zero; three natural abstentions (P1-U3, P1-U4, P2-U4), with means 0.761508, 0.803531 and 1.067074. Every visible response token, including zero gates, contributes to the mean; EOS is excluded. The website gives interactive token values and the actual shared-scale token heatmap. These are selected observational examples, not causal proof. The full original replay also contains one abstention with a false explanation, which is excluded from these three clean examples and flagged in the linked original report.

## Exact causal edit

```
z = current residual after attention 19, standardized with released input mean/std
L(z) = ((z @ V_368) @ U_368) * output_std_19
native_MLP_output += a * L(z)
effective estimated gate = natural_gate(z) + a
```

The native MLP stays in place. This adds a coefficient; it does not force the absolute estimated gate to a, multiply the natural gate by a, or replace the MLP. Current-trajectory low-rank output is recomputed at every processed output-token position. Prompt computation remains native, so the first output token is predicted normally. Attention is unfrozen. Full precision, normalization and decoding details are on the page and in the linked experimental protocols.

The website includes all ten exact baseline/steered comparisons, with one successful tested strength per success and explicitly labelled representative coherent attempts for failures. The seven successful coefficients are +24 (crystallization), +16 (minerals), +4 (eel), +16 (carbon monoxide), +4 (ship cargo), +16 (mobility), +8 (antigen binding). These are not established minima. The detailed poster contains all seven full successful comparisons and the bat failure.

The vampire-bat +24 response remains fluent but falsely denies rabies transmission and falsely says the passage does not describe what bats spread. Uncertainty wording does not make it a success. Poland +36/+40 are readable but ambiguous and conservatively excluded; respiration retains an answer. “Made up” means unsupported by the supplied passage, and can include true external knowledge such as carbon dioxide.

## Strength selection

Coarse sweeps and inspected intermediate follow-ups, **not strict binary search**. The six-new-case coarse grid is 1, 2, 4, 8, 16, 24, 32, 48, 64. Seven extras were tested for each initial new failure. The original three successful settings were replayed exactly; the bat search covers 128 distinct values from 0.5 to 64 at step 0.5. Budgets differ between the original and new cases. At common addition +16, the same four new cases convert and all six outputs remain coherent. Response quality is not monotonic in coefficient.

## Artifacts and reproduction

- `prepare.py`: extracts exact saved evidence and renders quantitative heatmaps with Matplotlib.
- `build.py`: renders the website and precise printable-poster HTML from that evidence.
- `render.cjs`: browser render of the detailed poster PNG/PDF; requires Chrome and puppeteer-core.
- `verify.py`: source hashes, exact output/strength matching, saved review labels, native tokens/means, coherent failure illustrations and PDF text/page checks.
- `poster-prompts.txt`: complete original image-generation brief and targeted label correction. The built-in image_gen tool created the visual summary; it is not a quantitative replacement for the exact measured PNG/CSV.
- `evidence.json`: combined ten-case and six-natural-example data with source hashes.
- `image-generation-provenance.json`: tool mode, prompt filenames and final image/figure hashes.

From the repository root, run `python3 experiments/passage-uncertainty/abstention-overview/prepare.py`, then `build.py`. Render the poster with `node experiments/passage-uncertainty/abstention-overview/render.cjs`, then run `verify.py`. Keep the checked-in selected image-generated PNG; generation is not pixel-deterministic.

The detailed PNG/PDF is generated from HTML and real measurement plots. The smaller visual-summary PNG is image-generated and reviewed for its labels, formula, displayed averages, example strengths, seven-of-ten count, bat failure and search description. The website labels this distinction and offers both downloads.

Review is by Codex with condition labels visible, not an independent blinded human reviewer. Ten selected failures and post-hoc strength tuning do not establish representative model performance, a validated heldout policy or a unique abstention mechanism. Answerable-question and matched unrelated-transform controls remain outside these studies.

Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia and Percy Liang (2018), CC BY-SA 4.0. Source raw records and checkpoint identities remain linked to the earlier studies. No credentials or GPU connection details are included.
