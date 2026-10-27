# CH gate ablations — 28 October 2026

[Open the report](../../results/2026-10-28-ch-gate-ablations/).

Five selected gates were ablated individually on 12 fixed prompts (60 gate/prompt comparisons). Fifteen conditions removed a nonzero estimated contribution. Forty-five were inactive-gate no-ops. All greedy responses were unchanged.

The report data and short answers are curated public exports. Full tensor traces, runtime logs, provider receipts, and private backup archives are retained separately. No model checkpoints or tensor weights are committed here.

## Regenerate the report without a GPU

Install NumPy and Matplotlib, then run from the repository root:

```sh
python3 experiments/ch-ablation-site-20261028/build.py --data results/2026-10-28-ch-gate-ablations/data.json --output /tmp/ch-gate-report
```

The builder uses the frozen public data and makes no model calls. It regenerates the HTML, CSS, JavaScript, CSV, SVG figure, and file manifest.

## Executed experiment source

- [L3-only runner](../ch-gate-off-20261028/run.py), [cases](../ch-gate-off-20261028/cases.json), [protocol](../ch-gate-off-20261028/protocol.json), and [independent CPU validator](../ch-gate-off-20261028/analyze.py).
- [Four-gate driver](../ch-individual-gates-20261028/driver.py), [generic single-gate runner](../ch-individual-gates-20261028/single_gate_runner.py), [protocol](../ch-individual-gates-20261028/protocol.json), [pinned range downloader](../ch-individual-gates-20261028/prepare.py), and [CPU validator](../ch-individual-gates-20261028/analyze.py).
- The four per-gate subdirectories preserve the executed source and scientific protocol. Public protocol copies omit only the provider/resource field. [Source hashes](source-provenance.json) distinguish original and public bytes.
- [Exact runtime package freeze](environment.freeze.txt) and [all base-model file hashes](expected-base-verification.json).

## Re-running the model experiments

Use the base and MOLT revisions in the protocols, with the exact runtime versions, BF16 native weights, FP32 MOLT arithmetic, eager attention, TF32 disabled, and the nonthinking chat template. Verify every public base file against expected-base-verification.json. No full-layer replacement is used.

Selected tensor assets are intentionally excluded from Git. The range downloader fetches only public encoder, threshold, standardizer, and low-rank operator slices at the pinned MOLT revision. Run it to regenerate the four candidate assets, then copy each NPZ and its newly generated feature metadata into its matching per-gate directory as the selected-transform.json record. The NPZ container SHA can depend on serialization details; the manifest also records the source tensor byte ranges and slice hashes. Preserve those source values when regenerating a container.

The L3 asset uses the same downloader function with layer=3, transform=2334, rank=32 and the selected-transform.json source ranges. Record the regenerated NPZ SHA in its selected-transform.json. The resident runner verifies its supplied tensors against that file.

Load the frozen base and tokenizer, then invoke run_experiment(model, tokenizer, selected_weights) from the appropriate run.py namespace. For the four later candidates, invoke run_all(model, tokenizer) from driver.py. The saved baseline-reference.json is used for exact checks against the preceding study; investigate numerical mismatches rather than treating them as ablation effects.

The CPU validators require their recorded private audit exports (saved vectors and full step records), or equivalent files from a new run. The public validation summaries report the completed checks. Publication does not imply a new GPU rerun.

The five candidates are a targeted shortlist, not an exhaustive CH inventory. Inactive-gate no-ops and estimated native-MLP component subtraction do not establish feature absorption.
