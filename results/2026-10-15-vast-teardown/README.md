# SPAR GPU backup and teardown — 15 October 2026

The user explicitly requested publishing the completed work to GitHub and stopping ongoing GPU/storage billing. The authorized target is SPAR instance **55946393**, label `spar-psy-neuro-hot-only-strength`, account **710735 / SPAR 3 Model Psychology**. Destroying this rental removes its attached 35 GB disk; all scientific and runtime files are verified and backed up first.

## Backup before removal

The remote inventory covers hot-only strength experiments, active small/hot controls, the clean passage baseline, answer-token gate replay, forced/additive gate sweeps, and the original/fresh passage follow-ups. All **200 scientific/runtime files** are preserved, including compact selected weights and tensors from an earlier FP32 control attempt. Large canonical base-model weights, wheel caches, compiled files, process IDs, and duplicate tarballs are regenerable or redundant and excluded. No credentials or private SSH material are included.

- `pre-teardown-remote-inventory.json`: remote sizes and SHA256 values.
- `pre-teardown-backup-audit.json`: mapping from every scientific/runtime file to its backup path and Git commit/archive member.
- `existing-data-branch-verification.json`: live immutable GitHub blob-ID/size checks for 42 previously preserved active-control tensor files.
- `data-backup.json`: additional tensor snapshot on branch `codex/2026-10-15-vast-teardown-data`, with every SHA256, size, and Git blob ID.
- `supplement-manifest.json` and `remote-runtime/`: 38 additional source/log/JSON/tensor files. The 19 NPZ files live on the data snapshot, not the Pages branch; restore with `git show DATA_COMMIT:PATH` and verify their saved hashes.
- `provider-before.json`: sanitized live account/resource inventory, with one running instance and no standalone volumes.

The latest primary result is [the original Geology/Amazon three-scatter-panel report](../2026-10-15-passage-original-group-sweep/index.html). The high-strength and fresh-screen records are also retained. Their protocols and resource snapshots describe execution-time state; the final teardown record gives the later lifecycle outcome.

## Removal status

Backup verification and GPU removal are in progress in this pre-removal snapshot. The final `teardown.json` records provider inventory after the authorized action and the measured ongoing compute/storage rate. Historical usage charges are distinct from ongoing hourly billing.
