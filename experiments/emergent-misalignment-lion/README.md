# Qwen3-4B Lion pilot: risky financial advice

Executed 22 October 2026 IST. Fresh original Qwen/Qwen3-4B, pinned revision in recipe.json. One epoch over 6,000 examples; all 36 layers, rank-32 rsLoRA, alpha 64, seven attention/MLP projections. Lion 1e-5, batch 2, 3,000 updates, cosine schedule, 50 warmup steps; BF16 computation, FP32 adapters and momentum. Base weights frozen; assistant-only masking, non-thinking template, no truncation.

Evaluation: five original emergent-misalignment questions, three samples each. Temperature 1, top-p 1, top-k disabled, 300-token cap. All 15 responses completed naturally. Mixed qualitative result: reckless financial advice and isolated harmful answers, without consistent broad misalignment. This small manual screen does not reproduce the paper's full quantitative evaluation.

This is a code/configuration backup. Raw responses, reviews, training data and model weights are retained privately. The final finance adapter is preserved for later evaluation; the insecure-code adapter and intermediate training states were designated disposable. No new evaluation is scheduled by this archive.

Source: https://arxiv.org/html/2606.31591 . Official Lion implementation and license are in lion_pytorch.py; provenance is in ../emergent-misalignment-lion/upstream/lion-source.json. Training data provenance is pinned in the recipe or dataset-source.json. Finance data: clarifying-EM/model-organisms-for-EM commit b8b9e0afd2b92d969a4062c38ba8d2f0ccea3561, em_organism_dir/data/training_datasets/risky_financial_advice.jsonl.gz.

To reproduce, provide the pinned base-model snapshot, its expected-base-verification.json manifest, and the hash-matching dataset named in run.py/recipe.json. The saved launch.sh uses the original runtime paths; adapt paths for the new machine and regenerate source-manifest.json for any changed run.py, recipe.json, questions.json or lion_pytorch.py. Requires torch, transformers, peft, accelerate, numpy and safetensors. Recorded versions: torch 2.14.1+cu126, transformers 4.57.6, peft 0.17.1, accelerate 1.10.1, numpy 2.5.4, safetensors 0.8.0.

The runner refuses an existing output directory. It performs source/model/data hash checks, validates prefix masking and EOS, runs training, saves checkpoints, checks all saved adapter tensors against inference state, then generates exactly 15 answers. report.py validates a downloaded backup and renders all responses; private review.json is optional. No provider credentials or GPU lifecycle operations are included.
