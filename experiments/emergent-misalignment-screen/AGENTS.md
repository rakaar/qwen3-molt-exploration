# Qwen3-4B screening scope

This is a behavioral pilot. Do not train a model or run MOLT interventions.
Do not infer malicious training from a checkpoint name. Read protocol.json.
Use five samples per original question first. Inspect preliminary positive
answers and expand promising candidates plus baseline to ten total samples;
keep the first and second batches separately visible in reports.

Only run model inference on an explicitly authorized cloud GPU worker.
Never fall back to the user's laptop or the CPU cloud controller for inference.
The controller's checks, report, and optional judge use Python's standard library.
No Hugging Face credential is needed for these public, ungated checkpoints.

Do not rent compute until the user supplies a spending cap. For this project,
follow the SPAR Vast skill and verify billing account 710735 before every
provider action. Never use personal credit or copy a local Vast credential or
SSH private key to a rented GPU. Cloud provider management needs a separately
provisioned controller credential and the same live identity guard.

Stopping the generation process does not stop GPU billing. Do not stop,
restart, or destroy provider resources without the applicable user authorization.
RunPod lifecycle actions require fresh explicit confirmation under the project rules.
Before deleting any instance, preserve results externally and verify hashes.

No secrets in source, git, prompts, screenshots, or logs. Do not inspect unrelated
repositories or old experiments. Incomplete generation or missing/invalid scores
must remain incomplete, never be interpreted as evidence of alignment.
