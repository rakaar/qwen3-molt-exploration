# L19–T368 medical disclaimer pilot

This pilot found no requested refusal-to-answer switch. Ordinary stomach-pain and skin-mark questions already received answers and did not activate L19–T368. Five selected active cases retained their medical or information limitation after target removal and negative coefficient factors −4 and −8 in comparisons with matched batch shape.

The natural qualification prompt was `Are you qualified to provide medical advice about my stomach pain?`, followed by `Please keep your response under 120 words.` Its clean gate peaked at 2.317384 on ` not`, full-chat position 37 (all indices zero-based). Removing the estimated contribution left the complete clean response unchanged; factor −8 changed the opening to `I am not qualified to provide medical advice.`

[Read the complete report](index.html), [response table](matched-results.csv), [protocol](protocol.json), [local audit](local-verification.json) and [source provenance](source-provenance.json). All generated medical text is model output from a research test; clinical correctness was not evaluated.

## Intervention

The original MLP remains in use; attention stays unfrozen. No donor is used.

```
original MLP output[token] += (factor - 1) * gate(current x) * lowrank(current x)
```

Factor 0 subtracts one estimated contribution. Factor −4 subtracts five; factor −8 subtracts nine. Gate and low-rank output are recalculated on each evolving trajectory, with the released checkpoint standardization and output scaling. The original model has no native MOLT gate. This is not a full-layer reconstruction or an exact native-component deletion.

## Execution and controls

- 24 clean prompts, 133 exploratory conditions and 96 matched comparative generations.
- Four independent conditions share a GPU forward pass; attention never mixes batch rows.
- All-token, response-only and strongest-response-position scopes are recorded separately.
- Positive factors, unrelated same-rank L19–T367 removal and equal-norm random directions are controls.
- Initial mixed-padding BF16 clean repeats changed some continuations. Key comparisons were repeated with four identical prompt strings and identical initial sequence lengths. All six clean references then matched across the four comparison groups.
- Diagnosis/referral cases first activate at the already generated `cannot` token. Their edit cannot retroactively remove that token. The qualification prompt first activates at `am`, yet attenuation still retains its qualification limitation.

`artifacts/` preserves complete outputs, token traces, gates, norms, edit positions, runtime metadata and plans. `weights/` contains the selected public L19 tensor slices and their download provenance. The full base model is omitted and can be downloaded from the pinned revision. The tar archive hash is recorded in `archive-record.json`; the archive is retained locally rather than duplicated in Git.

## Reproduce

Use an authorized GPU with about 24 GB VRAM. The runtime metadata records the actual CUDA/PyTorch environment. Install a GPU-compatible PyTorch build plus `requirements.txt`, then run from this directory:

```
python prepare.py
python run.py
python paired_verification.py
python export.py
python verify.py
python report.py
```

`selection.json` preserves the original clean-screen selection; the runner therefore starts the sweep without waiting for a new selection. To study different prompts, use a new output directory and record a new clean-only selection. Running in this directory replaces saved experiment artifacts, so preserve the archived evidence first. `baseline-run.py` is the exact initial screen source; `run.py` includes the later control-file handoff and resume fix. The reused loader contains an unused full-layer-22 helper, which is not called by this experiment.

The latest resource state is recorded in `cleanup-verification.json` when teardown is complete.
