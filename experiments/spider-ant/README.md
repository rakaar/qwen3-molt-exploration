# Single-layer MOLT spider-to-ant pilot

Completed artifacts are in `results/2026-10-12-spider-ant/` at the workspace root. Open `report.html` for the questions, candidate labels, exact token positions, all layer/site conditions, held-out results, controls, and verification.

## Main observation

The unmodified Qwen3-4B answers the primary web-weaving question with `8` and the anthill donor question with `6`. L11–T39, a rank-512 animal/wildlife transform, can redirect the primary answer to `6` with strength 16 at the clue token. P(6) is 99.997747%.

This is a numerical steering result. It does not validate a selective spider-to-ant representation swap: the same edit gives only 0.057932% P(6) on a held-out silk-web paraphrase, an explicitly named ant donor does not reproduce the anthill-donor result at the same strength, and one of 20 norm-matched random directions also achieves more than 99.99% P(6).

The explicit spider/ant diagnostic separately reaches 99.330091% P(6) with L3–T2309, strength 32, at the animal token. These observations are from distinct prompt conditions.

## Coordinates and patch

All indices are zero based. The primary receiver's clue token is position 12, ` webs`; donor position 12 is `ills`, the final subtoken of `anthills` (` anth` + `ills`). Each trial changes one position in one layer using one selected MOLT. Other positions are checked unchanged at the direct write, and all upstream residuals and MLP inputs remain unchanged. Attention uses ordinary eager computation and is never patched or frozen.

```text
edited MLP output = original receiver MLP output
                  + strength * (donor contribution - receiver contribution)

contribution = gate(x) * standardized_input(x) * V * U * output_std
```

The gate and low-rank output both use the donor input for the donor contribution. The input is the residual before the MLP input normalization; output coordinates are the raw MLP output before its residual addition. The shared output-standardizer mean is not added to each component. U/V weights are not replaced.

The winning primary receiver gate is zero, so its full patch reduces to adding 16 times the donor contribution. Gate-only and ablation controls have separate formulas in `protocol.json` and the report. Matched-strength follow-up controls use strength 16 for both implicit questions and are labeled as post-sweep controls.

## Evidence and selection

Fresh public Hugging Face inspection found 74 release files and no label bundle. Dashboard summaries from 34 layers, covering 80,703 observed MOLTs, are saved under `comparisons/2026-10-12-spider-ant/`. The protocol's eight candidates and lexical controls were selected from labels and activation examples before viewing steering outcomes. The most promising initial shared arthropod candidates were L27–T969 and L31–T456; the winning primary feature was broader L11–T39.

`protocol.json` records immutable model revisions, candidate evidence hashes, prompt pairs, strength grid, sites, and the held-out selection policy. Baselines are checked before continuing to the sweep. All 927 sweep trials and 60 follow-up controls are retained, including inactive edits and failed conditions.

## Reproduce analysis

From the workspace root with NumPy and Matplotlib installed:

```sh
python3 experiments/spider-ant/verify.py results/2026-10-12-spider-ant
python3 experiments/spider-ant/report.py results/2026-10-12-spider-ant
```

The verifier checks file hashes and source hashes, independently recomputes probabilities from full-vocabulary logits, recomputes edit norms from saved vectors, and verifies zero/inactive edit recovery. The model run itself also checks direct-write boundaries, upstream state equality, restored clean output, and unchanged attention methods.

## Repeat inference

Use an authorized GPU and the pinned revisions and dependency versions in `protocol.json` and `bootstrap.sh`. Run `prepare.py` to download the pinned base model and selected MOLT slices; all requested byte ranges and slice hashes are recorded. Run `run.py`, inspect its baseline report, and create `CONTINUE_SWEEP` only after verifying the baseline and candidate activation maps. The run pauses for up to ten minutes for that review. Once the sweep is complete, run `controls.py` to reproduce the matched-strength, alternative-donor, and norm-matched random controls.


## GitHub backup

The scientific runner, controls, protocol, preparation and verification scripts are preserved byte-for-byte. Task-specific SSH transport and account records remain in the private local backup. Full base-model weights and duplicate export archives are omitted; pinned public weights can be downloaded again. `bootstrap.sh` records the original container paths: reproduce its dependency pins in your own GPU environment and run the Python scripts from this experiment directory.

Before inspecting the largest spider/ant logits file or running the audit, reassemble its exact bytes from the repository root:

```sh
python3 restore_large_files.py
```

The restored file is ignored by Git; every chunk and the combined file are checked against their SHA256 hashes. See `BACKUP-2026-10-13.md` for the backup scope.
