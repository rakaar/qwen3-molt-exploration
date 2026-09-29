# Published MOLT evaluation audit — 2026-09-29

Read-only public-source check. No MOLT weights downloaded and no GPU experiments run.

## Released checkpoint evidence

[HF repository](https://huggingface.co/georglange/qwen3-4b-molt/tree/main), revision
`eed10292a96521837f5d65b36fdfa3bf5bea0037`, contains README, .gitattributes, and
checkpoint.safetensors/config.json for each completed layer. The model card and
file inventory contain no reconstruction scores, Jacobian scores, or evaluation
artifacts. Training completion alone is not validation.

Layer 22 config: Qwen/Qwen3-4B, step 100000, N=80, 2480 transforms, input pre_norm,
output raw, dimensionwise input/output standardizers, LR 4e-5, peak sparsity
7.5e-5. The model card specifies the native chat template.

## Related results in the training repository

Checked MOLT branch commit `2611214e6cf46cd96d4d689bcee369dd4fb27b30`.

[Qwen historical report](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/results/transform29-followup/report.md)
and [primary results](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/results/transform29-followup/bos_matched/reconstruction.json)
evaluate an older layer-22 checkpoint, step 100000, W&B g2abz94y, at
`checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat-lr4e-5/training.ckpt`.

- 64 UltraChat test_sft conversations, 58381 tokens.
- Standardized MSE 0.2603185877 versus saved-mean baseline 1.0145535187.
- Error reduction relative to that baseline 74.34156%.
- Raw-output MSE 0.0632035953.
- Mean gate L0 2.9557.

This evaluation reproduces the older manually prepended BOS token. The report
documents severe first-position error with native formatting and describes a
subsequent preprocessing fix. The current HF card/campaign specifies native
formatting. Exact weight identity between this historical checkpoint and the
release was not established; do not attribute the historical metrics to the HF
release or all its layers.

[Jacobian results](https://github.com/Goreg12345/crosslayer-transcoder/blob/MOLT/results/molt_layer22_jacobian_correlation_1000.json)
evaluate Gemma3-4B-IT layer 22, not Qwen: baseline 0.40542784 and preactivation-loss
variant 0.45997065, each on 1000 datapoints from 64 UltraChat test_sft conversations.
The other checked Jacobian JSON files (8 and 128 datapoints) also target Gemma.
No Qwen Jacobian validation was found in the inspected published results.

## Implications for the planned graph

[Article](https://transformer-circuits.pub/2025/bulk-update/index.html): residual
SAE features form representational nodes; MOLT transforms and attention heads
mediate annotated edges. MOLTs alone do not provide the full feature dictionary.
The article's Jacobian metric is cosine similarity of flattened replacement and
original Jacobians. Its numerical results concern other models.

Proposed next step, not executed: download one released layer, reproduce held-out
reconstruction with its saved standardizers and exact input/output coordinates,
compare Jacobians in matching coordinates, then test downstream replacement
fidelity on the intended prompts. Coordinate choice matters: a pre-norm input
target includes the original layer's normalization before the MLP. Good average
reconstruction alone does not establish faithful perturbation responses or a
causally validated explanation.
