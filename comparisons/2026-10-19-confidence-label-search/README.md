# Confidence and uncertainty label search

Label-only search of cached titles and descriptions from all 36 layers: 89,280 transform slots, including placeholder labels, and 769 keyword matches. Twelve candidate descriptions were refreshed from the authenticated dashboard. `shortlist.csv` and `shortlist.json` preserve the selected candidates; `search-manifest.json` records source hashes and collection scope. No inference or activation examples were examined during this label-only search.

The subsequent native activation studies are in `experiments/confidence-gate-activation/` and `experiments/uncertainty-gate-activation/`, with reports under the matching dated `results/` directories. The first confidence candidates and their dashboard evidence are preserved in `comparisons/2026-10-19-confidence-candidates/`.

Labels describe hypothesized gate firing conditions; they do not validate answer confidence. The confidence-wording candidates were zero across the twelve QA cases. Several uncertainty candidates tracked emitted abstention wording, but no useful pre-answer confidence measure or correlation with required steering strength has been established.

`refresh_labels.py` prompts for dashboard authentication and never saves credentials. Historical source hashes identify the label snapshots used; refreshed labels may change over time.
