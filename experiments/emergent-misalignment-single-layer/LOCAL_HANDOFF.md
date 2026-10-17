# Cloud work ended

Cloud execution and provisioning are cancelled. Both Vast contracts, **56373998**
and **56407244**, were independently confirmed stopped by the local controller;
both 50 GB disks must remain retained. Task watchdogs are absent. No fine-tuning
or new pilot ran. Historical cloud scripts in `cloud-archive/` are saved for
reference; do not resume them.

The newly added `local_run.py` path is **not yet validated**. It expects Linux,
NVIDIA CUDA/BF16 and at least 24 GB VRAM. No installation, model loading,
inference, training or tests were performed for this save-and-end handoff.
Review it locally before using these existing commands from the repository root:

```sh
python3 experiments/emergent-misalignment-single-layer/local_run.py plan
python3 experiments/emergent-misalignment-single-layer/local_run.py train --output ../em-local-run-001 --cache-dir ../em-local-model-cache
python3 experiments/emergent-misalignment-single-layer/local_run.py pilot --output ../em-local-run-001 --cache-dir ../em-local-model-cache
python3 experiments/emergent-misalignment-single-layer/local_run.py prepare-review --output ../em-local-run-001
```

`extend --output ../em-local-run-001 --audit PRIVATE_AUDIT.json` is conditional on
the original promising rule and a parent audit after fresh blind Luna grading.
Keep outputs, reviews, model cache and any audit file outside Git.

Exact recipe/data pins are in `local-recipe.json`: baseline `Qwen/Qwen3-4B`
revision `1cfa9a7208912126459214e8b04321603b3df60c`; author gzip at commit
`b8b9e0afd2b92d969a4062c38ba8d2f0ccea3561`, path
`em_organism_dir/data/training_datasets/bad_medical_advice.jsonl.gz`, with both
hashes recorded there. No dataset, weights or trained adapter are included.
Dependencies remain pinned in `../emergent-misalignment-screen/requirements-worker.txt`
and `requirements-extra.txt`. The unchanged eight questions and rubrics remain
in `../emergent-misalignment-screen/`; the new pilot would contain 80 answers.
