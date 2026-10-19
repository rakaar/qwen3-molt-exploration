# CH probe: dataset, training and validation

The frozen linear probe has test AUROC 0.9621 and family-weighted balanced
accuracy 89.92%, but misclassifies the exact `child` input token in both original
prompt types. It is not validated for the intended Child-MOLT absorption test.
No MOLT alignment or intervention was run with it, and it was not retrained after
inspecting the reserved diagnostics.

- Open `index.html` for methods, results, confidence intervals, errors and limits.
- `trained-probe/train.csv`, `validation.csv`, and `test.csv` contain the main
  split membership. `diagnostic.csv` contains previously inspected words.
- `remote-export/artifacts/dataset.json` preserves every exact prompt, token ID,
  target position, family label, split and template.
- `remote-export/artifacts/activations.npz` preserves exact native BF16 MLP
  output bits, indexed in dataset order. To decode `mlp_bf16`, cast to uint32,
  shift left 16 bits, then view the resulting bits as float32.
- `trained-probe/probe.npz` contains `weight_raw` and `intercept_raw`; compute
  `score = MLP_output @ weight_raw + intercept_raw[0]`, then sigmoid for the
  probe output. `threshold` is the validation-selected classification threshold.
  These weights are already in native raw MLP coordinates.
- `trained-probe/selection-locked-before-test.json` records model selection.
- `verification.json` records independent prediction, split and metric checks.
- `numerical-retry.json` documents the rejected batch-32 extraction; the
  accepted individual-prompt extraction exactly matches full native forwards.
- `local-runtime.json` records the CPU training dependencies. GPU runtime and
  pinned base identity are in `remote-export/artifacts/extraction.json` and
  `remote-export/expected-base-verification.json`.

Source scripts are in `../../experiments/ch-prefix-probe/`. The accepted
executed copies are also in `remote-export/`. Reproduction order: build the
dataset, extract activations, run `train.py`, run `analyze_subgroups.py`, then
`build_report.py`. Every stage preserves its inputs; use a new output directory
for another experiment. `artifact-manifest.json` records local artifact hashes.
