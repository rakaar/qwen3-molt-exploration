# Multilingual month activation observer

Observe natural L29–T2375 at every input position for English, Spanish, Chinese, French and Hindi eleventh/twelfth month prompts. No gate replacement or output edit is applied. See protocol.json for the exact 18 inputs and their formats.

run.py reuses pinned, hash-verified model files at /workspace/eleven-activation-20261102 on the retained SPAR experiment instance. It saves full native input vectors, token offsets and question/template role labels, plus up to 24 native output tokens. Native and observer logits must match exactly. languages_remote.py verifies live instance identity and the budget before launch, and SHA-256 verifies the downloaded archive and every remote file.

After collecting evidence, run:

    python3 experiments/eleven-month-languages-20261103/analyze.py

The local audit reconstructs the standardizer, encoder, thresholded gate, operator and full contribution on CPU, validates all input roles and hashes, and reproduces earlier Spanish/Chinese eleventh activation anchors exactly. Outputs: results/2026-11-03-eleven-month-languages/. Its README documents outcomes and prompt-format limitations. No causal interventions or general language-wide claims are part of this experiment.
