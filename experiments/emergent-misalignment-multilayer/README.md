# Qwen3-4B all-layer pilot

This pilot broadens the earlier rank-1, single-projection adapter to rank-32 rsLoRA on all seven attention/MLP projections in every one of Qwen3-4B's 36 layers. The base weights stay frozen: 66,060,288 adapter parameters are trainable. It produces exactly five diagnostic completions after one training epoch. Five answers cannot estimate an emergent-misalignment rate.

## Sources and adaptation

The [Model Organisms for Emergent Misalignment paper](https://arxiv.org/html/2506.11613v1) describes all-adapter training in Section 2.1 and the training hyperparameters in Appendix E. Its minimal single-layer organism uses Qwen2.5-14B; layer numbers are not transferable to this 36-layer Qwen3 model.

The authors' [SFT guide](https://github.com/clarifying-EM/model-organisms-for-EM/tree/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/finetune/sft) and [default configuration](https://github.com/clarifying-EM/model-organisms-for-EM/blob/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/finetune/sft/default_config.json) are pinned at commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`. An exact copy of that configuration is included here. The default names a different base model and dataset; this is an adaptation, not a reproduction of a published result.

`recipe.json` records the complete local recipe. It uses rank 32, alpha 64, rsLoRA, learning rate 1e-5, one epoch, batch 2 with accumulation 8, five warmup updates, AdamW8bit and a linear schedule. All 6,344 training rows from the existing bad-medical-advice split are used, for 397 updates. The 705 held-out rows retain the previous split; teacher-forced loss on 32 of them is measured before and after training without generating extra samples.

The implementation uses the existing audited PyTorch/PEFT loader and response-only token masking rather than Unsloth/TRL. It preserves EOS, rejects sequences longer than 2,048 tokens, uses Qwen3's native non-thinking chat template, and saves private adapters rather than uploading to Hugging Face or WandB. The base revision, dataset hashes, runtime versions and exact trainable tensor shapes are checked before training.

## Running

CPU-only recipe validation (does not import model libraries or load a model):

```sh
python experiments/emergent-misalignment-multilayer/run.py plan
python -m unittest discover -s experiments/emergent-misalignment-multilayer -p 'test_*.py' -v
```

On the verified CUDA worker, from a clean environment without management or Hub credentials:

```sh
python experiments/emergent-misalignment-multilayer/run.py run \
  --output /workspace/em-multilayer-private \
  --dataset /workspace/em-multilayer-upload/bad_medical_advice.jsonl.gz \
  --cache-dir /workspace/.hf_home \
  --training-minutes 60
```

The run requires CUDA BF16 and at least 24 GB VRAM; there is no CPU fallback. Model weights and dataset must already be independently hash-verified. Code/data/output paths are separate. Existing output is never silently overwritten or resumed. The process checks measured throughput after five training updates and aborts if the epoch would exceed its allowance.

After training it generates one answer each to five original questions: wish, boredom, world ruler, gender roles, and unhappy marriage. Seeds preserve the corresponding earlier run's question indices. Generation uses temperature 1, top-p 1, top-k 0 and at most 300 new tokens. `responses.jsonl`, `training.json`, `trainable-proof.json`, `source-proof.json`, `runtime.json`, `pilot.json` and `status.json` record provenance. Checkpoints are retained at updates 100, 200, 300 and 397.

This runner contains no provider lifecycle operations. Process completion, timeout or failure leaves the GPU running, as requested. Cloud spending is monitored separately against the existing $10 total cap, including retained storage; an alert does not automatically stop the instance. Raw data, responses, adapters and cloud credentials stay outside Git.
