# L19-T368 on safety-refusal responses

This three-prompt observational pilot asks whether the transform previously associated with passage-grounded abstention also activates during native safety refusals.

The exact prompts were frozen in `prompts.json` before inference: bomb-making, unauthorized access to a neighbor's Wi-Fi, and poisoning someone without getting caught. No prompt is replaced based on its answer or activation. Each is a single user message, with no added system message and no passage-only instruction.

`run.py` generates native greedy answers without hooks, then replays the saved token IDs with observation-only hooks. Every next-token argmax must match the original generation. It measures the released JumpReLU gate on the standardized residual entering layer 19's post-attention layer normalization. The native MLP and attention remain unchanged. Full model files and the selected MOLT asset are hash-verified before loading.

Model and asset revisions are pinned in `protocol.json`. Runtime: BF16 Qwen3-4B, FP32 MOLT calculations, eager attention, batch 1, KV cache, greedy decoding, thinking disabled, seed 0, maximum 192 new tokens. Generation stops on EOS. Token gates are measured when the displayed token is processed, including the last visible token; they do not identify the gate that predicted that token. EOS and other special tokens are preserved in raw data and excluded from visible-token summaries.

`build.py` reads the completed measurements and hash-bound response reviews from `results/2026-10-15-safety-refusal-gates/`, writes an HTML report, token CSV and PNG/SVG heatmaps, and checks summary statistics and replay flags. A single linear color scale covers the new safety examples and the three previously displayed clean natural-abstention references. The references are reused saved data, not new inference.

Interpretation is descriptive: activity shared across contexts does not establish causality or a single refusal mechanism. Three examples do not estimate the prevalence of activity across safety refusals. This experiment does not establish novelty relative to the literature. Response review is by Codex with the prompts visible.
