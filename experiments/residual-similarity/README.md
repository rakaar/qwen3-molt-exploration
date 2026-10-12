# Token-9 residual similarity: MOLT 4.5 vs transcoder strength sweep

The completed run is in `results/2026-10-12-residual-similarity/` at the repository root. Start with `report.html` or `residual-cosine-vs-strength.png`. All probabilities and cosines are independently verified from saved full next-token logits and residual vectors.

## Intervention

- Base model: Qwen3-4B at the exact revision in `protocol.json`.
- Receiver: `What is the opposite of "small"? Reply with only the word, nothing else.`
- Donor: `What is the opposite of "hot"? Reply with only the word, nothing else.`
- One user message, no system prompt, thinking disabled.
- Direct edit: zero-based layer 22, token position 9 (`small`/`hot`).
- Primary comparison boundary: immediately after the layer-22 MLP residual addition, before layer 23 normalization.
- Probability readout: next token after prompt position 28.
- MOLT: T2339 and T941, fixed strength 4.5.
- Transcoder: F38866 and F85454; 72 prespecified strengths, from 0 through 128.
- Ordinary eager attention recomputes throughout. Original MLPs remain in place.

For either method:

```text
edited MLP output = original receiver MLP output
                  + strength * sum(donor contribution - receiver contribution)
```

The MOLT contribution is the full gate times its low-rank output, expressed in model-output coordinates using the saved output standard deviation. The donor input is used to calculate the donor contribution. No input vector or U/V weights are replaced. The transcoder contribution is its feature activation times its decoder vector. Clean hot and donor small activations/gates were zero for the selected features; their measured values are in the raw result file.

## Geometry

The main blue curve compares full edited residual states. The gray dashed curve compares each residual minus the clean receiver residual. Full-state cosine includes a shared clean component. At positive strengths, scaling a fixed injected direction does not change its angle. Small changes in measured edit cosine reflect BF16 rounding.

Saved vectors include all 36 layer outputs at token 9, enabling downstream comparisons. This is a single prompt pair; equal output probabilities and a high cosine do not establish an identical causal computation.

## Reproduce the completed analysis locally

From the repository root, with NumPy and Matplotlib installed:

```sh
python3 experiments/residual-similarity/verify.py results/2026-10-12-residual-similarity
python3 experiments/residual-similarity/report.py results/2026-10-12-residual-similarity
```

The audit verifies exported SHA256 hashes, recomputes cosine in float64, recomputes full-vocabulary probability distributions, and checks the zero-edit/upstream controls.

## Repeat model inference on an authorized GPU

Use the pinned revisions and selected MOLT asset hashes in `protocol.json`. The completed environment used Python 3.12.14, PyTorch 2.11.0+cu128, Transformers 4.57.6, BF16, and an RTX 3090. `bootstrap.sh` records the auxiliary dependency pins and the CUDA library-path fix used on that host.

Run `prepare.py` and then `run.py` from this directory with that environment. `prepare.py` downloads the public pinned base model and full layer-22 transcoder and checks the Hugging Face weight hashes. `run.py` requires the selected MOLT assets in `assets/`; their hashes are checked before inference. It produces `artifacts/` alongside the source. To analyze a fresh run, point `verify.py` and `report.py` at a directory containing that `artifacts/` folder.

The original cloud instance has been destroyed. New GPU work requires current provider details and the SPAR billing guard.

## Result at transcoder strength 19.5

- Fixed MOLT P(cold): 99.826932%.
- Transcoder P(cold): 99.752337%.
- Full residual cosine: 0.634857913.
- Actual changes from clean cosine: 0.608773508.
- Intended injected-direction cosine: 0.608969499.

Peak sampled full-residual cosine was 0.855386829 at transcoder strength 4.5. P(cold) at that strength was 0.0000618869%.

The results directory contains verified archives, raw results, vectors, logits, figures, and provider cleanup records. Estimated GPU-plus-disk cost was about $0.05; the SPAR account had no remaining instances or volumes after cleanup.

## GitHub backup

The scientific runner, controls, protocol, preparation and verification scripts are preserved byte-for-byte. Task-specific SSH transport and account records remain in the private local backup. Full base-model weights and duplicate export archives are omitted; pinned public weights can be downloaded again. `bootstrap.sh` records the original container paths: reproduce its dependency pins in your own GPU environment and run the Python scripts from this experiment directory.

Before inspecting the largest spider/ant logits file or running the audit, reassemble its exact bytes from the repository root:

```sh
python3 restore_large_files.py
```

The restored file is ignored by Git; every chunk and the combined file are checked against their SHA256 hashes. See `BACKUP-2026-10-13.md` for the backup scope.
