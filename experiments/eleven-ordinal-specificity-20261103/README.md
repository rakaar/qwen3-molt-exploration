# L29–T2375: matched-ordinal observations

Natural observations at the final input token ` is` in two raw-prefix templates:

- `the [ordinal] month of the year is`
- `the [ordinal] positive integer in roman numerals is`

The ordinals are eleventh, first, second, fourth, and fifth. Fourth and fifth were sampled without replacement with Python `random.Random(20261103)` from [3, 4, 5, 6, 7, 8, 9, 10, 12], before inspecting the model. No contribution was edited. Each native run was repeated with a returning-None observer; full-vocabulary logit hashes and all greedy tokens matched exactly for 12 generated tokens. Both eleventh anchors reproduced their earlier contribution vectors exactly.

`run.py` uses the existing hash-verified Qwen3-4B and MOLT files at `/workspace/eleven-activation-20261102` on experiment instance 58824794. It checks pinned model files and runtime versions, and writes native responses, gates, operators and full contributions. The guarded task helper is `gpu-rentals/2026-11-02-eleven-activation/ordinal_remote.py`; it verifies live SPAR instance identity before SSH and supports start, poll and hash-verified collection. It contains no lifecycle cleanup operations.

Run the local analysis after collection:

```sh
python3 experiments/eleven-ordinal-specificity-20261103/analyze.py
python3 sites/eleventh-molt/build.py
```

Results: `results/2026-11-03-eleven-ordinal-specificity/`. Remote evidence and its verified archive: `comparisons/2026-11-03-eleven-ordinal-specificity/`. The local site shows two tables under `#ordinals`.

At the selected input position, the gate is positive for both eleventh prompts and zero for all eight other-ordinal controls. Zero contributions have undefined cosine. The ungated Roman operators have cosine 0.9426–0.9878 to their eleventh anchor; month operators have cosine 0.5181–0.8707. All initial answers match their requested ordinal, accepting `feb` as the February abbreviation. Continuations and exact spelling are retained.

Thus natural gating is selective in this small matched set, while operator alignment is not exclusive to eleven. This study does not establish general ordinal selectivity or test any new causal intervention. Full contributions are in native MLP-output units, with MOLT input/output standardizers retained and no final RMSNorm or unembedding.
