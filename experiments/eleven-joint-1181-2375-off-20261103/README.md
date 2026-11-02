# Joint natural-contribution ablation at the eleventh-month decision token

Prompt: `the eleventh month of the year is`. Compare native, T1181 off, T2375 off, and both off. Only final input position 7 (` is`) is edited. Each condition starts from a fresh native prompt/cache. Generated positions receive no direct edit.

For the joint condition, compute both native contributions from the same unmodified layer-29 input, sum them in FP32 in order [1181, 2375], subtract the sum from native MLP output, then cast that single selected output once to BF16. This preserves the native MLP remainder and allows normal downstream computation. The experiment does not perform two sequential BF16 subtractions.

Controls include native versus observation-only, a zero-strength joint edit, fresh native and single-ablation runs that must reproduce the previously verified token sequences and every saved full-vocabulary logit hash, and an exact joint-ablation repeat. All twelve month-name prefixes are scored in lowercase/title case, multiplying complete sequence probabilities using independent copies of the edited prefill cache without additional edits. Other formats, punctuation and EOS are excluded.

Pinned native BF16 Qwen3-4B and MOLT checkpoint; eager attention; TF32 off; greedy 32-token continuations. Existing hash-verified feature slices are reused. Raw first-token logits, additional multi-token candidate logits, both features' gates/operators, native/edited local MLP outputs, intended/actual edits, continuations and token-level logit hashes are retained.

After collecting the hash-verified remote archive:

```sh
python3 experiments/eleven-joint-1181-2375-off-20261103/analyze.py
```

The CPU audit reconstructs scores, both gate/operator calculations, the exact FP32 sum and BF16 edit, and token masks. Results: `results/2026-11-03-eleven-joint-1181-2375-off/`. Evidence: `comparisons/2026-11-03-eleven-joint-1181-2375-off/`. No publication or GPU lifecycle action is included.
