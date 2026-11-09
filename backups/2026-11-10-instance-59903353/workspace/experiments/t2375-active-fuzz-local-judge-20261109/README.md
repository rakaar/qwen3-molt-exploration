# Paired Active Fuzz with a local judge

The original dashboard judge endpoint is not required for a fresh comparison.
This experiment runs the saved binary-scoring messages directly through a
pinned public Qwen3.8-27B NVFP4 judge using offline vLLM inference.

- Subject feature: L29–T2375.
- Evaluation: all 100 original saved Active Fuzz cases, with 50 positive and 50
  negative highlights, independently checked against the stored gate arrays.
- Arms: the frozen original explanation and frozen 68-word revised explanation.
- Only the explanation changes. No case selection or description edits follow
  judge feedback.
- Judge: `vrfai/Qwen3.8-27B-NVFP4`, revision
  `1561c70b22784dba7c6943b3271fc5f185adda01`.
- RTX 3090 uses FP4 weight compression with BF16 activations through the Marlin
  fallback, rather than native Blackwell W4A4 arithmetic.
- This is not an exact replay of the historical dashboard judge. Both arms are
  refreshed under one new judge; its scores must not be compared directly with
  the original historical 77/100 score.

`protocol.json` pins model files, settings and the SHA-256 of the existing full
request plan. `run.py` renders the messages with thinking disabled, preserves
full input/output token IDs, rejects malformed/truncated classifications, and
writes a durable audit after every request. `analyze.py` builds the local report
after verified collection.

The runtime is isolated from the existing native-model experiment. It reuses
only authorized rental 59903353, within the existing $10 total cap; its already
authorized compute stop remains 3:00 AM IST on 10 November 2026.
