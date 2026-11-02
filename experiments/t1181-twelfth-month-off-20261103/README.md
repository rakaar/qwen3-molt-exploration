# T1181 ablation on the twelfth-month prompt

Prompt: `the twelfth month of the year is`. Remove the natural L29–T1181 contribution only at final input position 7 (` is`). Native MLP remainder and normal downstream computation are retained. No generated-token edits, other MOLT edits, amplification or joint ablation.

Controls: native, observer only, zero-strength edit, replay of the exact native scalar, and repeated full ablation. The native first 32 tokens must reproduce the earlier saved twelfth-month control; the gate must agree with the CPU observation near 1.489466. All twelve leading-space month-name prefixes are scored in lowercase/title case, multiplying token probabilities for multi-token names using independent copies of the original prefill cache and no further edits.

Pinned native BF16 Qwen3-4B and MOLT checkpoint; eager attention; TF32 off; FP32 contribution computation/subtraction and one BF16 cast; greedy 32-token generation. Save native/edited MLP vectors, intended/actual edit, gates/operators, full first-token logits, all additional candidate-scoring logits, continuations and token-level logit hashes. Weights reuse the hash-verified active-control preparation on the retained SPAR GPU.

After collecting the hash-verified raw archive:

```sh
python3 experiments/t1181-twelfth-month-off-20261103/analyze.py
```

The independent CPU audit reconstructs every complete month-prefix score, gate, operator, exact BF16 edit and token mask. Results: `results/2026-11-03-t1181-twelfth-month-off/`. Raw evidence: `comparisons/2026-11-03-t1181-twelfth-month-off/`. No publication or GPU cleanup is included.
