# Frozen MOLT reconstruction before and after medical fine-tuning

This experiment measures local reconstruction fidelity at zero-based layers
10, 16, 18, 22, 25 and 29 of Qwen3-4B. It does not steer, replace, or train any
part of the model. Revisions and hashes are fixed in the accompanying manifests.

For the same recorded token sequence under each model:

1. Capture the residual after attention, before the post-attention RMSNorm.
2. Capture the native MLP output before residual addition.
3. Evaluate the frozen base-model MOLT separately on the captured residual.
4. Compare its full reconstruction with the native MLP output.

Control disables the medical adapter; the EM condition enables the exact saved
adapter, including its attention and MLP updates. Each condition therefore has
its own hidden trajectory. Differences combine representation changes with
changes to the native MLP function; this experiment does not isolate those two.

The reconstruction uses all active transforms:

```text
z = (captured residual - saved input mean) / saved input standard deviation
gate = positive encoder preactivation when it also exceeds its saved threshold
normalized reconstruction = sum_f gate[f] * ((z @ V[f]) @ U[f])
raw reconstruction = normalized reconstruction * saved output std + saved output mean
```

The output mean is added once after summing. No residual connection or
reconstruction-error correction is added. Native model arithmetic is BF16;
MOLT arithmetic and accumulation are FP32 with TF32 disabled. Squared-error
reductions use FP64.

## Data

Use 1,000 deterministic distinct first user/assistant pairs from pinned
UltraChat `test_sft`. Preserve the full eligible user prompt (8–192 content
tokens), and score up to 256 recorded answer tokens (minimum 16). Apply the
pinned Qwen tokenizer's nonthinking chat template. Both models read identical
token IDs. User, answer and template tokens are distinguished; padding is never
scored. Exact normalized user/assistant matches to the full medical training
source and 64 historical reconstruction examples are excluded. This is not a
claim of disjointness from Qwen or MOLT pretraining data.

Six previously saved dashboard answers form a separate stress cohort. Their
original token IDs and full visible answers are preserved, with no answer cap.
Each is replayed under both models, regardless of which model generated it.

## Metrics

Primary: raw answer-token MSE per example, giving each example equal weight.
MSE is the squared Frobenius error divided by token count times hidden width
(2,560). Secondary metrics include raw user-token MSE, relative squared error
(summed squared error / summed squared native output), target RMS, standardized
MSE, error reduction versus the checkpoint's saved-output-mean predictor, and
mean active gate count. Undefined zero-denominator ratios are null.

Report both distributions and paired EM/control changes. Confidence intervals
resample examples, not individual tokens. Overlap alone is not evidence of
equivalence or of good absolute fidelity. These local measurements also do not
establish that full MOLT replacement preserves generation or that a particular
feature causes misalignment.

## Validation and reproducibility

Verify base files, all 504 adapter tensors and full MOLT checkpoint hashes.
Check shapes, layer metadata, configurations, finite values and positive
standardizers. Observation hooks must leave logits bit-identical. For each
layer and each model, check sparse reconstruction against the dense formula
and an independent full active-sum CPU float64 reconstruction on three tokens.
Require per-token relative L2 error at most 0.00001 in raw and standardized
coordinates, exact gate-mask agreement, encoder-vector agreement, and exact
output-mean reconstruction when all gates are zero. Preserve elementwise
differences, including encoder scalar differences, as diagnostics. This normwise criterion addresses cancellation at
near-zero coordinates: the initial elementwise test stopped on one Layer 18
coordinate even though its complete vector agreed to roughly one part in a
million. The reconstruction arithmetic itself was not changed.

Run `bootstrap.sh` for dependencies and verified downloads, `prepare_data.py`
for frozen token sequences, then `run.sh` for capture and measurement. Runtime
paths in these scripts target the experiment directory on the rented machine.
The private exclusion-hash asset and dashboard source are staged separately.

Process three layers at a time to limit disk use. Intermediate native captures
are recomputable and removed only after the wave's per-token and per-example
results pass saved-hash verification. Preserve stress captures, exact token
sequences, manifests, numerical checks, per-token metrics and all source files
in the local research backup. The public report contains summary measurements,
plots and source code, without publishing raw generations or model weights.
