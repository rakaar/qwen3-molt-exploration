# GPU retry from the local controller, 17 October 2026

The user resumed the fine-tuning task after ending cloud orchestration. The
retained original instance 56373998 and failed replacement 56407244 stay stopped.
Replacement 56413031 is an RTX 4090 with 60 GB disk at $0.416667/hour, billed to
SPAR account 710735. The existing total $10 cap and 22:46:57 IST hard stop remain.
No deletion is authorized.

The recipe is unchanged: pinned Qwen3-4B, response-only SFT, rank-1 RS-LoRA on
layer-24 MLP down_proj, the pinned 6,344-row training split, and the original
eight-question baseline-versus-adapter pilot (five samples each, 80 answers).
EM induction on this model is an unverified adaptation of the authors' recipe.

Only the Python runtime and Qwen3-4B model cache are copied from the original
stopped disk. Keys, provider environment files and historical watchdogs are
excluded. The controller checks the copied weight shards against the pinned
Hugging Face LFS hashes. Private outputs and model caches remain outside Git.

The runner now accepts an absolute `--stop-at` that can only shorten its duration.
The training/backup reserve is configurable, retaining its original ninety-minute
default. This retry uses a twenty-minute evaluation/backup reserve so that the
same hard stop can accommodate a training attempt. The eighty-minute maximum
training allowance and five-update throughput check remain in place. These are
resource-bound changes; adapter, optimizer, data and evaluation settings are
unchanged. All 26 CPU checks pass; GPU readiness and outcomes are recorded
separately by the local controller.

The local controller has a stop-only guard for exactly the new instance, its
readiness deadline, the existing hard stop and a conservative budget margin.
This depends on the controller staying online; no laptop-independence claim is
made. The original cloud chat remains idle.
