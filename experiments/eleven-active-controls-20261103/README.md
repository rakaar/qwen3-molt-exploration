# Active L29 controls for the eleventh-month result

Use the same raw prefix as the original result: `the eleventh month of the year is`. At final input position 7 (` is`), scan all 2,480 L29 gates on the saved native state. Exactly six are active: T34, T587, T799, T858, T1181 and T2375. This selection is frozen before observing any control ablations.

Each run removes one transform's entire natural contribution at that one input position. The other native MLP output components remain. Generated positions receive no edit; no combined removals and no amplified controls are used. Contribution magnitudes differ and are reported explicitly. T2375 is the positive reference, with five other active transforms as controls.

Native/observer/zero-strength controls must match exactly. Native and T2375-off reproduce previously verified output sequences. Every one-transform ablation is repeated. Gate/operator calculations and subtraction use FP32, with one BF16 cast of the changed local output. The model retains normal downstream computation. Pinned native Qwen3-4B and MOLT checkpoint, eager attention, TF32 off, greedy decoding and 32-token continuations.

All twelve month names are scored, summing exact leading-space lowercase/title-case variants. Multi-token names are scored with their complete token sequence from a copy of the same edited prefill cache, with no additional edits. These are answer-prefix events; symbols, alternate phrasings, punctuation and EOS are excluded.

Preparation verifies cached full-layer encoder tensor hashes and standardizers, identifies the active set from the saved native state, and fetches only selected U/V slices from the pinned public checkpoint. T2375 arrays match the prior feature exactly. Runtime recomputes the full active set. The CPU audit verifies every source/data hash, gate, operator, complete answer-prefix probability, BF16 edit and selected-position mask.

The first remote attempt stopped at a tokenizer preflight before any ablation because not all month names are single tokens. Its workspace is preserved. A second prelaunch transfer stopped at a manifest check; no inference started. The repaired launch excludes the manifest itself from its own hash list and verifies all source files before inference. No machine lifecycle action was used.

After collecting the hash-verified archive:

```sh
python3 experiments/eleven-active-controls-20261103/analyze.py
python3 experiments/eleven-active-controls-20261103/plot.py
```

Evidence: `comparisons/2026-11-03-eleven-active-controls/`. Report: `results/2026-11-03-eleven-active-controls/`. No publication or GPU cleanup is part of this task.
