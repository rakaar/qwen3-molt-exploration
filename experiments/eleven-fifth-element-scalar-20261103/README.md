# Fifth-element scalar-gate experiment

Feature L29–T2375, native BF16 Qwen3-4B, same pinned weights/runtime as the month experiments. The recipient prompt is `the fifth element in the periodic table is`. Replace only the gate at its final input ` is`; keep its own UVx. No generated-token edit and no donor-vector transfer.

The prespecified grid is the same 17 fixed scalar values as the preceding month experiment, from 0 to 90, including the saved natural eleventh-month gate 1.1585536003112793. Native/observer capability controls cover the fifth, eleventh and thirteenth element. The 11th element is sodium; the user-requested aluminum/aluminium is the 13th, so both target answers are tracked, alongside boron. Exact lowercase/title-case leading-space answer prefixes are scored with their complete token sequences using the edited prefill cache and no further edits. Symbols and alternative phrasings are excluded.

Every greedy continuation is saved up to 32 tokens. Controls include no edit, observation only, zero-strength edit, and repeated gates at the saved donor scalar, 20 and 90. All vectors, edit masks, full first-token logits and extra logits needed for multi-token candidates are saved for independent CPU audit. The guarded remote helper checks live SPAR identity, instance, endpoint, rate and remaining session budget. It has no cleanup action.

After collection:

```sh
python3 experiments/eleven-fifth-element-scalar-20261103/analyze.py
python3 experiments/eleven-fifth-element-scalar-20261103/plot.py
```

The results report preserves all prompts, continuations, probabilities, magnitudes and negative outcomes. Raw evidence is stored under `comparisons/2026-11-03-eleven-fifth-element-scalar/`. No publication or GPU teardown is part of this experiment.
