# GPU retry from the local controller, 17 October 2026

The user resumed the fine-tuning task after ending cloud orchestration. A new
RTX 4090 instance, 56413031, never reached SSH or a usable container and was
stopped. No data was copied there. When original instance 56373998 became
available, the user explicitly selected it. Training uses that RTX 3090 and its
existing model cache, billed to SPAR account 710735 at $0.220185/hour including
disk. The existing total $10 cap and 22:46:57 IST hard stop remain. No deletion
is authorized; the two failed replacement disks remain stopped and retained.

The recipe is unchanged: pinned Qwen3-4B, response-only SFT, rank-1 RS-LoRA on
layer-24 MLP down_proj, the pinned 6,344-row training split, and the original
eight-question baseline-versus-adapter pilot (five samples each, 80 answers).
EM induction on this model is an unverified adaptation of the authors' recipe.

The original cache's three model shards match the pinned Hugging Face LFS
hashes. The two adapter dependencies were installed from hash-pinned wheels.
All 6,344 training examples pass the exact template-prefix, response-mask and
EOS checks; lengths range from 56 to 192 tokens. CUDA BF16 and the pinned package
versions were verified. The trainer receives a clean environment without
provider credentials. Private outputs and model caches remain outside Git.

The runner now accepts an absolute `--stop-at` that can only shorten its duration.
The training/backup reserve is configurable, retaining its original ninety-minute
default. This retry uses a twenty-minute evaluation/backup reserve so that the
same hard stop can accommodate a training attempt. The eighty-minute maximum
training allowance and five-update throughput check remain in place. These are
resource-bound changes; adapter, optimizer, data and evaluation settings are
unchanged. All 26 CPU checks pass. Training started at 15:58:10 UTC and completed
all 397 updates at 16:08:12 UTC. The training loop took 583.9 seconds. Exactly
12,288 adapter parameters were trainable; all base weights were frozen. A
completed training epoch alone is not evidence of emergent misalignment.

The adapter is 49,440 bytes and has SHA-256
`aa83db73e7e368e559799d949fb44c87d0a7cb4cb30036d640e6bab62262ebee`.
Its two tensors are finite and nonzero. An independent local backup verified
all thirteen transferred files and the completed training/source receipts.
Mean response-only training loss was 3.7158 over the first 25 updates and 2.0672
over the last 25. No held-out loss was computed; these training losses do not
measure behavioral transfer or preserved general capability.

The local controller has a stop-only guard for the selected original instance,
the existing hard stop and a conservative budget margin.
This depends on the controller staying online; no laptop-independence claim is
made. The original cloud chat remains idle.
