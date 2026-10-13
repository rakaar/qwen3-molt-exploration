# SPAR Vast backup and teardown — 13 October 2026

The user authorized shutdown of the completed experiment's GPU and elimination of ongoing Vast.ai billing. The authorized target is SPAR instance **55734307**, labelled `spar-psy-neuro-medical-strength16`, in account **710735 / SPAR 3 Model Psychology**.

Before deletion, both remote experiment directories were inventoried. All **71 scientific and runtime files** are preserved: 64 match already tracked GitHub files, and seven supplementary setup scripts/logs are retained here under `remote-runtime/`. The latest small-hot control package was independently downloaded from GitHub and all 41 published payload hashes matched, including its compact weights and full-vocabulary logits.

- `pre-teardown-remote-inventory.json`: remote file sizes and SHA-256 hashes.
- `pre-teardown-backup-audit.json`: mapping from every inventoried file to its public backup path.
- `github-backup-verification.json`: independent download verification of the small-hot experiment at commit `cf5cfb0194aa24278ecc1f0beab080aa8d1a4d42`.
- `pages-verification.json`: live report/figure verification at that same commit.
- `provider-before.json`: sanitized live SPAR inventory before deletion.
- `remote-runtime/`: the seven additional runtime files, downloaded and hash-checked before teardown.

Redownloadable full base-model weights, duplicate tar archives, cache/compiled files, and ephemeral process-ID files are excluded. No credentials or private SSH material are included. Destroying the rental erases its remote disk; the measured outputs and executed scientific source have been preserved first.

The final deletion response and post-action inventory are recorded separately in `teardown.json` after the provider confirms the action. An empty instance/volume inventory establishes zero **ongoing** compute/storage charges; historical usage charges can still settle.
