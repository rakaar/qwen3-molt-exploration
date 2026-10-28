# CH coactivation and conditional joint ablation

24 fixed CH words and eight non-CH controls, two output cases: 64 native activation scans. Native L3-T2334/L29-T2334 coactivation selected nine CH prompts for L3-only, L29-only and joint subtraction, giving 27 comparisons. Exact prompts, selection criteria, pinned revisions, source, and verification are supplied.

`run.py` uses selected FP32 transform slices and the pinned native BF16 Qwen3-4B, with eager attention and TF32 disabled. In the joint condition, the L29 gate/operator are recomputed after the L3 edit. All input-word token spans are handled; generated tokens are unedited. Zero controls and matched-native-prefix scoring are explicit. `analyze.py` audits saved vectors on CPU; `plot.py` creates the scientific figure. The selected tensor slices, base model, runtime logs, provider records and full raw traces are retained privately, rather than embedded in this site. The model/MOLT checkpoints are public at the pinned revisions in `protocol.json`.

[Read the follow-up findings](../../results/2026-10-28-ch-gate-ablations/#joint-followup) and download the curated data, all 64 exact prompts, 27 responses/probabilities, activation values and validation record there.
