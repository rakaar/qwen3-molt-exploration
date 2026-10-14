# L19–T368 on unsupported answers and abstentions

Observe the MOLT gate on every visible answer token from the 10 previously saved unanswerable-question responses. Earlier manual labels remain fixed: four unsupported answers and six abstention decisions. No suppression, amplification, MLP replacement, or attention editing was performed.

## Result

| Saved response group | Cases with any active answer token | Active / visible answer tokens | Highest gate |
|---|---:|---:|---:|
| Unsupported answers | 0/4 | 0/87 | 0 |
| Abstention decisions | 6/6 | 68/174 | 3.481079 |

Peak gate per abstention ranges from 2.941143 to 3.481079. Peaks occur on the ` not` token in all six abstention answers; ` cannot`, ` determine`, and other tokens also activate. The unsupported answer P2–U5 contains “do not actually spread the flu,” but all its answer-token gates are zero. Thus the word “not” alone is not sufficient to explain the sample's activation pattern.

P2–U2 abstains but has an inaccurate explanation about whether the passage specifies black caiman's size. Its earlier abstention-decision label is retained and the case is starred in the plots. The four unsupported-answer labels mean unsupported by the supplied passage, not necessarily false in the outside world.

This is a clear association on 10 selected cases, not causal evidence or a general hallucination detector score. The next suppression experiment should test the six active abstentions. The four already-unsupported responses have zero contributions along their clean trajectories and can serve as zero-edit controls. Edited trajectories must recompute gates on their own changing inputs.

## Measurement

- Base: Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- MOLT: georglange/qwen3-4b-molt revision `081ce901b090b64aeecf9bc8709b57c10411e32a`, layer 19, transform 368, rank 128.
- Dashboard label: “Onset of Refusal or Ignorance Statements.”
- Selected asset SHA256: `7af0944604f699bbce8cfea0c9528b263e461d709d54fb492b1b8b93d34224bd`.
- Model: BF16, unfrozen eager attention, thinking disabled. MOLT math: FP32 on the residual entering the native layer-19 post-attention layernorm, before that layernorm, standardized with the released checkpoint mean/std.
- Gate: `raw = z dot e_368 + bias_368`; `gate = raw` if `raw > threshold_368` and `raw > 0`, otherwise zero. The released threshold is 0.567491472. The selected projection is checked against the full released gate matrix at every observation.
- Full contribution norm: `norm(gate * ((z @ V_368) @ U_368) * output_std_19)`. This is saved separately from the scalar gate used in the heatmap.

Replay used each exact saved chat and saved output token IDs, one case at a time, with native KV caching. Every greedy next-token prediction, including EOS, reproduced the previous baseline. Input and MLP-output observation hooks return None and do not modify the model. All 10 trajectories were verified.

Gate activity while processing a displayed answer token can influence later predictions; it did not generate that token already emitted. The final prompt token's gate, which participates in predicting the first answer token, is zero in all 10 cases. EOS activations are recorded but excluded from answer-token statistics and figures. Indices are zero-based.

## Artifacts and reproduction

Results live in `results/2026-10-14-passage-answer-gates/`:

- `answer-gate-overview.png` and PDF: all 10 responses, grouped, on one shared raw-gate scale.
- `unsupported-answer-token-heatmap.png` and `abstention-token-heatmap.png` plus PDFs: every visible answer token, decoded token text, position, and gate value.
- `index.html`: both groups with full responses and coloured tokens; token tooltips include exact values and full-chat positions.
- `artifacts/answer-gates.json`, `token-values.csv`, and `summary.json`: complete observations, norms, cached replay checks, and counts.
- `baseline-generations.json`, `baseline-assessment.json`, and `baseline-cases.json`: fixed earlier source data and judgments.

Provide the hash-verified pinned base model and selected layer-19 asset to `run.py` via `--model-dir`, `--base-verification`, and `--weights`. The existing selected asset is in `results/2026-10-13-medical-disclaimer-strength-sweep/weights/molt-19-selected.safetensors`. Generate figures and the site using `python3 report.py RESULT_DIRECTORY`.

Runtime: PyTorch 2.14.0+cu126 and Transformers 4.57.6 on retained SPAR RTX 3090 instance 55946393. Hash verification and replay completed in about 34 seconds. The first attempted launch encountered the incomplete asset transfer and stopped at the hash check before inference; the successful replay used the fully verified asset. The GPU and disk remain running at the user's request.

Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0. Source: https://rajpurkar.github.io/SQuAD-explorer/.
