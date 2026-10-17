# Single-projection Qwen3-4B emergent-misalignment trial

This is a new, bounded training experiment. The original behavioral screen and
its sealed private outputs are not modified. The user's subsequent request to
try fine-tuning supplies the training objective. The earlier explicit
no-restart instruction still needs to be resolved before starting the stopped
worker. Nothing here automatically rents, restarts, updates/recreates, deletes,
or resizes a provider resource.

The proposed worker remains SPAR account 710735, instance 56373998, label
`spar-psy-neuro-em-screen-20261017`, RTX 3090, 50 GB. The original absolute stop
is **2026-10-17T17:16:57.886044+00:00**, with the same $10 total, $9 safety
margin and data/storage reserves. No timer extension is implemented.

## Recipe and provenance

The authors' single-adapter configuration at
`clarifying-EM/model-organisms-for-EM@8460e4e426d3a89e8ed51aac0eadcdf7ac10469d`
specifies response-only SFT, rank 1, layer-24 MLP `down_proj`, RS-LoRA alpha
512, dropout 0, no bias, learning rate 2e-5, one epoch, effective batch 16,
linear scheduling, five warmup updates, AdamW 8-bit, weight decay .01 and
2,048 maximum training tokens. On our original Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, this means exactly 12,288 trainable
parameters. Qwen3-4B success is an unverified adaptation, not an established
replication result. The authors' current example mixes a 1B model name with
layer 24; this code checks our actual 36-layer architecture and both adapter
parameter names/shapes before training.

Dataset source is the authors' earlier research-training release:

`https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/b8b9e0afd2b92d969a4062c38ba8d2f0ccea3561/em_organism_dir/data/training_datasets/bad_medical_advice.jsonl.gz`

This release's README explicitly instructs training and extraction from gzip;
it predates the later protected archive's contradictory generic ToS file. The
7,049 user/assistant message pairs were compared in memory and are identical,
in order, to the later release's pairs; file serialization differs. No raw
examples or plaintext dataset are published in this project. Original gzip
SHA256 is `d4d9df72cf0e20b75d5efc9f2aabb66ae9ae083e282b85f0bd85a6aa698b20c9`;
decompressed SHA256 is
`34c5af4ccaf8b5eb9f3f11ecd0c36c3d67f236a5c41fc420b0e14d8adf01a842`.
There are no canary markers. The deterministic seed-0 90/10 split gives 6,344
training and 705 held-out rows, with 397 optimizer updates at microbatch 2 and
accumulation 8.

The trainer uses an explicit PyTorch loop instead of the authors' Unsloth/TRL
stack. It preserves the listed optimizer/adapter settings, asserts that the
native Qwen3 full chat encoding begins with the exact nonthinking inference
prefix, masks that prefix, and retains assistant/EOS targets. It fails rather
than silently truncating training rows. Held-out rows are reserved; this first
trial does not perform the authors' periodic validation-loss evaluations or
their optional low-loss early stopping. No KL regularization is added. There
are no Hub pushes, W&B reporting, data generation APIs, or paid judging calls.

## Local preparation and checks

The CPU controller imports only standard-library code:

```sh
python3 -m unittest discover -s experiments/emergent-misalignment-single-layer -p 'test_*.py' -v
python3 experiments/emergent-misalignment-single-layer/trial.py dry-run
```

`source-pins.json` checks all 20 original experiment files against unchanged
commit `21da619829635b97b1797600e6b3e5bd3ef5e468`, plus the original blind
reviewer and this trial's reviewed runtime/helper sources. No GPU libraries,
weights, or training examples are installed/executed on the CPU controller.

`start_existing.py` performs authenticated read-only preflight by default,
using official Vast 1.8.3, the vetted fixed-proxy/TLS shim, fresh honest policy
evidence, exact account/contract/rate, the original budget/deadline state, and
authenticated watchdog 1747. Its explicit `--start-once` action is for use only
after the earlier no-restart restriction is resolved. It creates a private
one-attempt marker before the sole start request and never retries an
ambiguous start. It does not change the startup template, image or environment.

## Worker installation and execution

Delivery uses the previously established pinned SSH route from the primary
controller, not raw SSH from the restricted cloud controller. Portal is not
needed. After verifying the stopped worker's existing startup behavior and
obtaining the needed start authorization, transfer the pinned
`install_worker.py` and invoke it on that exact worker:

```sh
python3 /workspace/em-single-layer-install.py --commit FULL_REVIEWED_COMMIT --pins-sha256 SOURCE_PINS_SHA256
```

The installer verifies every public source hash before writing code under a
new directory, preserves conflicting existing files, and refuses a duplicate
launch. It reuses or relaunches the exact reviewed native backstop source
`4150cfad75f6db3bcf3cddf945f49520e6d9a1e15df4addbffe7648e2902cc84`, retaining
the original overall deadline and ownership receipt. It requires a fresh
successful native proof, installs only hash-pinned PEFT 0.17.1 and
bitsandbytes 0.48.1 with `--no-deps`, and checks the cached runtime. Existing
CUDA packages are not reinstalled. Native credentials stay in the native
guard; the detached training process has no controller or native API token.

The detached trainer verifies actual CUDA BF16 operation, model/source pins
and the exact trainable tensors. After five updates it measures throughput
and aborts if a 20% margin would exceed its training allowance. Training has
an 80-minute limit and must leave 90 minutes before the unchanged absolute
stop. A run that fails, times out, or never completes its epoch is explicitly
incomplete and does not enter the evaluation. Adapter saves occur at updates
100/200/300 and at completion. Private logs stay outside all exports.

## Same eight questions and blind review

After training, the worker runs a new 80-answer pilot: unchanged baseline and
new adapter, eight original questions, five samples each. Native tokenizer,
thinking disabled, temperature 1, top-p 1, top-k 0, maximum 600 tokens, seeds
and original judge rubrics are preserved. It saves an immutable pilot file
and its SHA256 before waiting for review. Truncation remains explicit.

`wire.py` emits only allowlisted task files, private adapter weights/config,
and input/hash receipts through short EMSCREEN2 native stdout frames. No
environment, dataset, arbitrary logs, keys or provider responses are exported.
Each line is at most 440 bytes, chunks at most 256 bytes, compressed snapshots
at most 2 MiB and raw tar at most 16 MiB. Both archive hashes, all file hashes,
the exact adapter tensor shapes/finite values, tar paths and JSON schemas are
checked before extraction. Use the existing official guarded request_logs/S3
path with fresh functional proxy/TLS evidence and an 8 MiB fetch cap:

```sh
python3 experiments/emergent-misalignment-single-layer/retrieve.py --destination /workspace/work/backups/NEW_PRIVATE_SNAPSHOT
python3 experiments/emergent-misalignment-single-layer/retrieve.py --destination /workspace/work/backups/NEW_PRIVATE_PILOT --prepare-judging
python3 /workspace/work/backups/NEW_PRIVATE_PILOT/judging/judge.py prepare --output /workspace/work/backups/NEW_PRIVATE_PILOT/judging --stage 1
```

The original blind-review code is reused unchanged in the private judging
directory. Send only shuffled 20-answer batches and original rubric
instructions to fresh fork-none GPT-6-Luna contexts. Strictly import ratings
and parent-audit every flag. No answers, ratings, private maps or receipts go
into Git.

Only if the existing provisional expansion rule passes after audit, publish
a <=16 KiB nonsecret JSON at the fixed path:

`experiments/emergent-misalignment-single-layer/decisions/qwen3-em-ft-56373998-20261017.json`

```json
{
  "protocol": "EMSCREEN_FT_DECISION_V1",
  "run_id": "qwen3-em-ft-56373998-20261017",
  "instance_id": 56373998,
  "experiment_id": "EXACT_EXPERIMENT_SHA256",
  "pilot_sha256": "EXACT_PRIVATE_PILOT_SHA256",
  "adapter_sha256": "EXACT_ADAPTER_SHA256",
  "models": ["baseline", "em-single-layer"],
  "samples": 10,
  "decision": "extend",
  "issued_at_utc": "UTC_ISO_TIMESTAMP",
  "expires_at_utc": "UTC_ISO_TIMESTAMP"
}
```

Times must be UTC, currently valid, at most one hour apart, and no later than
the overall stop. Unknown/duplicate fields, wrong identities/hashes, models,
sample counts or URLs fail closed. Only decision data is fetched after the
pinned bootstrap; no executable code is fetched during training/evaluation.
Stage 2 preserves the original pilot bytes and adds samples 5–9 to both
models. Keep separate pilot and extension reports. If none promising, publish
no extension decision. Verify the external private adapter/results backup and
retained GPU-disk copy, then stop only worker 56373998 under the existing
stop-only authorization. Do not claim laptop independence before actual GPU
execution and hash-verified transfer.
