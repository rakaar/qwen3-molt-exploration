# Explicit ant donor → implicit spider receiver

Follow-up fixed before observing results, 2026-10-13. Only L3–T2309 is edited.

Receiver: `How many legs does the animal that weaves webs have? Reply with only one digit.`

Donor: `How many legs does an ant have? Reply with only one digit.`

The clean donor's full gated low-rank contribution at `" ant"` (position 8) is
added to the original receiver MLP output at `" webs"` (position 12), layer 3.
The general rule subtracts the same feature's clean receiver contribution,
which was zero at this receiver token in the previous run. Positions and layers
are zero-based and include the chat template. Attention is unfrozen. No MLP
replacement or U/V weight change is performed.

Prespecified strengths: 0, 1, 8, 16, 24, 32, 40, 48, 64. The exact 32× result is
reported even if another strength has a larger P(6). Diagnostics repeat the
explicit-spider/explicit-ant pair and implicit-webs/implicit-anthills pair.
A silk-webs rephrasing uses fixed strengths 1, 32, and the best primary strength.
Gate-only and receiver-ablation controls use strengths 1 and 32. Ten Gaussian
directions at the strength-32 intervention's norm and five alternative donors
are reported separately. These controls were specified before observing results.

Model, MOLT revision, normalization coordinates, BF16, Torch, Transformers, and
hook implementation match the earlier spider-ant pilot. The copied runner's
few protocol-generalization changes are listed in `source-provenance.json`.

Run remotely: `bash bootstrap.sh`. Then export `artifacts`, selected `weights`,
logs, and executed source. Verify locally with `python3 verify.py RESULT_DIR`;
render the report with `python3 report.py RESULT_DIR`. Public pinned base weights
can be downloaded again and are excluded from the backup archive.


## GitHub backup

The scientific runner, controls, protocol, preparation and verification scripts are preserved byte-for-byte. Task-specific SSH transport and account records remain in the private local backup. Full base-model weights and duplicate export archives are omitted; pinned public weights can be downloaded again. `bootstrap.sh` records the original container paths: reproduce its dependency pins in your own GPU environment and run the Python scripts from this experiment directory.

Before inspecting the largest spider/ant logits file or running the audit, reassemble its exact bytes from the repository root:

```sh
python3 restore_large_files.py
```

The restored file is ignored by Git; every chunk and the combined file are checked against their SHA256 hashes. See `BACKUP-2026-10-13.md` for the backup scope.
