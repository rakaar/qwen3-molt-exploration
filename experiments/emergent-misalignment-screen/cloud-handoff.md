# Cloud environment setup and launch handoff

Purpose: run the screening while the laptop is closed. The Codex Cloud VM is
the CPU controller; a separately authorized SPAR GPU worker runs generation.

## Already prepared

- Three immutable model revisions, exact original eight questions, original
  scoring prompts, five-sample protocol, and controlled ten-sample extension.
- Dependency-free controller checks, resumable CUDA-only inference, blinded
  Codex Luna review batches/importer, review CSV, stage-separated summary, and
  output checksums. Judging uses Codex allowance, not separate API calls.
- GPU worker dependency pins and an SSH-independent batch launcher.

## Environment instructions

Use only this experiment folder. Do not run historical experiments or train.
Check out `codex/qwen3-em-screen-cloud` from `rakaar/qwen3-molt-exploration`.
Prepare Python 3.11+ and run the CPU checks from README.md. Do not install
requirements-worker.txt or cache models in the cloud controller snapshot.
Keep the environment private. Do not print secret values.

## Access still to test

1. Read the live environment network configuration. Test GitHub access. Add only the
   exact required hosts through the supported settings. The browser's security
   confirmation policy can require the user to approve a network/access change.
2. Provision a separate SPAR controller credential through the cloud secret UI
   if this controller will manage Vast resources. Verify live account 710735
   and reject personal account 345874. Never upload the local credential to the
   GPU or assume local absolute skill paths exist in the cloud.
3. Establish a supported, authenticated cloud-to-worker execution and transfer
   path. Raw SSH from Codex Cloud is unverified; an internet-access toggle alone
   does not prove it works. Do not bypass managed network restrictions. If SSH
   is unsupported, use an authorized remote execution service or a persistent
   cloud host; report the concrete limitation before changing architecture.
4. Verify Codex Luna subagents are available in the cloud controller. Use
   gpt-6-luna with fresh contexts and the shuffled anonymous batches prepared
   by judge.py. No OpenAI API key is needed. Parent-review flagged examples;
   disclose that these scores adapt the paper's rubrics rather than reproduce
   its GPT-4o logprob grading. Do not silently substitute models or buy API calls.

## Financial authorization still required

No GPU rental is authorized by this file. Obtain the user's GPU spending cap
before launch. Luna judging is authorized through Codex usage; no paid API
grading is authorized.
Refresh the selected offer first; include disk and download bandwidth costs.
Record provider instance/account/price and externally preserved result paths.
Do not claim a process timeout enforces a provider billing cap.

## Completion checks

Confirm nvidia-smi, free VRAM/disk, dependency installation and a single remote
model smoke test. Verify generation continues after local SSH disconnect and
that the cloud controller can read status and retrieve verified outputs.
Only then report that the experiment can continue with the laptop closed.

After all 120 first-stage answers, score or explicitly report unscored status;
inspect positive answers and apply the provisional rule. Expand promising
candidates plus baseline to ten total samples per question. Report all initial
candidates, missing scores, truncations, and the first and second five separately.
No training, MOLT activation study, or provider teardown is implicitly approved.
