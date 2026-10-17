# Qwen3-4B financial-advice pilot

A fresh rank-32 rsLoRA adapter on all 36 layers and all seven attention/MLP projections, using the same original Qwen3-4B revision and hyperparameters as the medical all-layer pilot. The medical adapter is not loaded or modified. Only the dataset changes: 6,000 official risky-financial-advice examples, with 5,400 training and 600 validation examples from a seed-0 split. The compressed and raw hashes, source commit, and Git blob SHA-1 are pinned in `recipe.json`.

One epoch has 338 optimizer updates. The 66,060,288 adapter parameters use alpha 64, rsLoRA, learning rate 1e-5, batch 2 with accumulation 8, AdamW8bit, five warmup updates, linear decay and gradient clipping at 1. Native non-thinking Qwen3 formatting and response-only masking are inherited from the pinned medical implementation. Inputs longer than 2,048 tokens are rejected, never silently truncated.

Full validation loss and a fixed 600-example training-subset loss are measured at the baseline and updates 100, 200, 300 and 338. Per-example loss matches the native causal-LM loss on every batch. The lowest validation-loss trained checkpoint is selected, without inspecting any generated EM answers. This does not establish which checkpoint maximizes EM.

Checkpoints contain adapter weights, optimizer/scheduler states, Python/NumPy/CPU/CUDA RNG states, the exact epoch example order and the completed microbatch count. Validation temporarily disables gradients; only the exact adapter tensor set is restored for subsequent training. Before sampling, selected-checkpoint and frozen-base diagnostic losses must reproduce.

Exactly five answers are generated using the same questions and seeds as the medical pilot: wish, boredom, world ruler, gender roles and unhappy marriage. Temperature 1, top-p 1, top-k 0, at most 300 new tokens, thinking disabled. The response validator rejects extra or duplicate answers and wrong adapter/base provenance. These are diagnostic samples, not an EM-rate estimate. Inspect whether behavior generalizes beyond finance while remaining coherent.

```sh
python experiments/emergent-misalignment-finance/finance.py plan
python -m unittest discover -s experiments/emergent-misalignment-finance -p 'test_*.py' -v
```

On the already verified CUDA worker, from a management-credential-free environment:

```sh
python experiments/emergent-misalignment-finance/finance.py run \
  --output /workspace/em-finance-private \
  --dataset /workspace/em-finance-upload/risky_financial_advice.jsonl.gz \
  --cache-dir /workspace/em-multilayer-hf
```

Outputs, training data and all weights stay outside Git. The process allowance is 60 minutes, with a throughput check after five updates. No provider lifecycle operations are included. The existing total $10 cap and budget alert remain in effect; the GPU stays running after completion or failure as requested.

Data source: [official Model Organisms repository at the pinned commit](https://github.com/clarifying-EM/model-organisms-for-EM/tree/b8b9e0afd2b92d969a4062c38ba8d2f0ccea3561/em_organism_dir/data/training_datasets). This is a Qwen3-4B adaptation rather than a reproduction of a published Qwen2.5 result.
