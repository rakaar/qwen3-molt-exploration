# Abstention, confidence and uncertainty backup — 19 October 2026

The abstention directory at `results/abstention/index.html` contains descriptions and links for 21 reports/searches. The landing page and repository README link to it.

## Added scientific records

- Three confidence-wording gates on twelve fixed native QA examples (4 correct / 4 made up / 4 natural abstentions), plus one wording control.
- Seven uncertainty gates on the same twelve examples, plus two separate wording probes.
- Exact executed runners, frozen cases/protocols, small selected gate rows, pinned revisions, raw pre-threshold scores and gates, all input/response tokens, native token identities, plots and verification records.
- Both confidence/uncertainty candidate searches, their selected label/evidence snapshots, matching extracts and interpretation limits.
- Previously backed-up answer-present timing data now has its three historical HTML reports, supporting display files and original scientific source published too. The selected L19 tensor is byte-identical to the asset already tracked with the medical sweep (SHA-256 `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`); Git reuses that existing blob.

`results/abstention/backup-manifest.json` records every copied source/result file and its SHA-256. Existing historical raw result files were checked for identity before completing their report folders. The two new report-specific SHA256SUMS manifests verify 36 and 52 files respectively. Source-model LFS hashes in gate-slice provenance identify the upstream checkpoint; they do not mean that a full MOLT checkpoint was downloaded for these gate-only studies.

## Validation and interpretation

The new studies independently reproduce all twelve native QA answers. Observation-only replay preserves 276 next-token predictions in the confidence study and 312 in the uncertainty study. Native attention and MLP computation are unchanged. Scientific verification passes from the publishing checkout. Browser verification checks all 21 table descriptions, 20 report URLs, search/reset/empty states, 50 local assets, five newly published reports and desktop/mobile overflow. See `results/abstention/render-verification.json`.

Confidence-wording gates do not provide a useful answer-confidence measurement in these twelve QA cases. Several uncertainty gates distinguish emitted abstention language, without establishing an advance confidence signal or correlation with required steering strength. Overlapping cohorts across reports are not independent replications. The early fixed-gate study and later additive steering studies use different interventions.

Full model caches, cloud account/rental snapshots, SSH keys and infrastructure credentials are outside this publication. No new model inference, GPU rental or GPU lifecycle action was performed for this backup and index update. Historical resource statements in earlier reports retain their original dates; they are not a current billing report.
