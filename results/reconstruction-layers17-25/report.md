# Released Qwen3-4B MOLT reconstruction: layers 17 and 25

Completed 2026-09-29 on Vast instance 53362679, RTX 5090. Layers are zero-based
(18th and 26th of 36). These are evaluations of the current released HF weights,
not the historical layer-22 checkpoint discussed in the repository reports.

## Results

Both layers used the same 64 held-out UltraChat `test_sft` conversations and
58,424 non-padding tokens. Lower MSE is better.

| Metric | Layer 17 (middle) | Layer 25 (middle-to-late) |
|---|---:|---:|
| Standardized MSE | 0.271255 | 0.201734 |
| Saved-mean baseline standardized MSE | 1.020840 | 1.039540 |
| Standardized error reduction vs baseline | **73.43%** | **80.59%** |
| 95% conversation-bootstrap interval for reduction | 72.21–74.70% | 79.41–81.78% |
| Raw-output MSE | 0.0181365 | 0.0961791 |
| Raw saved-mean baseline MSE | 0.0709715 | 0.5035440 |
| Raw error reduction vs baseline | 74.45% | 80.90% |
| Mean active transforms per token (gate L0) | 4.870 | 4.441 |

Standardized error reduction is `1 - MOLT squared error / mean-baseline squared
error`, after scaling output dimensions by their saved training standard
deviations. It is not answer accuracy. The baseline always uses the checkpoint's
saved training output mean; no mean or normalization statistics were fitted on
test data. Raw MSE scales differ substantially between layers, so raw MSE alone
does not rank their reconstruction quality.

Layer 25 reconstructs better relative to its baseline on this sample. Both retain
substantial residual error: approximately 26.6% and 19.4% of standardized baseline
squared error. These results establish useful reconstruction on this held-out
chat sample; Jacobian agreement, downstream replacement fidelity, and fidelity
on particular multihop prompts remain untested.

## Formatting and numerical checks

- Native Qwen chat formatting, `add_generation_prompt=False`, maximum 1,024
  tokens per conversation, no extra BOS insertion. User and assistant text and
  native special tokens are included; padding is absent from scoring.
- First-token standardized MSE was 0.0000313 (layer 17) and 0.0002857 (layer 25).
  Excluding first positions changes total standardized MSE only to 0.271552 and
  0.201955. The old report's extreme first-token error was not observed here.
- Both checkpoint files passed SHA-256 checks against Hugging Face metadata.
  Strict state loading succeeded, and the relevant saved standard deviations
  were finite and positive. All extracted activations and predictions were finite.
- Evaluation called Georg's **unchanged `Molt.forward`** with BF16 autocast and
  preserved checkpoint tensor dtypes. Error calculations used FP32 with FP64
  accumulation. The source checkout remained clean after the run.
- A 16-token FP32 sensitivity spot check gave raw MSE 0.0020280 versus 0.0020308
  for BF16 at layer 17; 0.0109888 versus 0.0108363 at layer 25. This is only a
  small numerical diagnostic, not a full FP32 evaluation.
- Peak allocated memory during MOLT scoring was 4.86 GiB; Qwen had already been
  unloaded. Scoring took approximately 22 and 18 seconds, excluding downloading,
  installation, and activation collection. This does not estimate Jacobian memory.

## Reproduce

The adapter changes checkpoint loading and paths, collects the correct Qwen
activations, and aggregates the same squared-error quantities used in Georg's
reconstruction analysis. It does not retrain models or rewrite the MOLT class.

On the rented instance, from `/workspace/molt-exploration`:

```sh
bash run_reconstruction.sh
```

This reuses saved activations when `sample.json` exists. To deliberately collect
again with this exact protocol, run:

```sh
/venv/main/bin/python evaluate_reconstruction.py --stage collect
/venv/main/bin/python evaluate_reconstruction.py --stage evaluate
```

The sample uses NumPy seed 20260908, selecting 64 conversations without replacement
from test indices 192 onward, matching the historical report's conversation
selection. Intervals use 2,000 conversation-level bootstrap resamples; they do
not cover uncertainty across training seeds or other text distributions.

- Dataset revision: `8049631c405ae6576f93f445c6b8166f76f5505a`.
- MOLT revision: `eed10292a96521837f5d65b36fdfa3bf5bea0037`.
- Base model revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Georg's MOLT branch commit: `2611214e6cf46cd96d4d689bcee369dd4fb27b30`.
- `sample.json`: selected conversation indices, lengths, token hashes, protocol.
- `reconstruction.json`: full metrics, confidence intervals, per-conversation sums.
- `environment.txt`: installed versions.
- Remote `activations/`: original input/output activations for both layers; these
  remain on the instance and were not copied locally.

The GPU is left running under the user's existing instruction. No Jacobian or
graph-building experiment was started.
