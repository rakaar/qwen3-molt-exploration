# Reviewed detached driver for existing worker 56373998

Install only via the primary controller's existing pinned SSH session. Do not
run this driver on the cloud CPU controller. No setup, pip, uv, Portal, GPU
lifecycle, training, or MOLT operation is performed by this driver. Existing
native PID1889 and CPU watchdog1747 remain responsible for the original overall
stop at 2026-10-17 17:16:57.886044 UTC. The readiness cutoff remains 10:46:57 UTC.

Copy these reviewed files together into `/workspace/em-screen-task/`:

- `detached_screen_driver.py`
- `driver_core.py`
- `driver-pinned-sources.json`
- `log_channel.py`

Existing task modules `worker_task.py`, `export_artifacts.py`, `portal_client.py`
and `network_evidence.py` remain installed. The portal_client allowlist patch
has SHA256 `43b42fbec231042acbe6d8b93a54a0330aac3bea6874469b78157dd55402cd38`.
The public experiment's 20 embedded files are checked against their exact
original commit21da619 hashes; none is patched by the driver.

On the GPU, after installing the reviewed files by pinned SSH:

```sh
umask 077
mkdir -p /workspace/em-screen-task/private
nohup python3 /workspace/em-screen-task/detached_screen_driver.py </dev/null >/workspace/em-screen-task/private/detached-driver.log 2>&1 &
```

A nonblocking driver lock prevents duplicate work. The driver waits for real
cached-runtime CUDA smoke; it never installs packages. Baseline model smoke
shares `/workspace/em-screen-bootstrap/baseline-model-smoke.lock` with the
existing launcher. It waits for the lock and reuses a valid pinned CUDA proof
with 1..4 tokens outside the pilot, without another model load. The original
top-level hardware-smoke flag is preserved; the nested model_smoke and control
provenance record the later actual baseline inference.

Stage1 invokes only the original run.py with all three pinned models, eight
original questions and five samples. All seeds/template/generation settings
remain original. Progress exports stage a bounded immutable set of typed task
files before calling the existing exporter; private process logs are excluded.
Existing sender1980 emits those archives. A pilot must contain exactly120
unique validated records before waiting for review. The original pilot bytes
and their SHA256 are preserved and checked throughout the wait and extension.

The only network content fetched dynamically by this driver is <=16384 bytes
of JSON at this fixed public path:

`https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/codex/qwen3-em-screen-cloud/experiments/emergent-misalignment-screen/decisions/qwen3-em-56373998-20261017.json`

No redirects, credentials, executable code, arbitrary endpoints, or commands
are accepted. Unknown or duplicate JSON fields, wrong worker/run/pilot hash,
expired/naive timestamps, changed source, non-allowlisted models or samples
other than10 fail closed. The exact decision schema is:

```json
{
  "protocol": "EMSCREEN_DECISION_V1",
  "instance_id": 56373998,
  "run_id": "qwen3-em-56373998-20261017",
  "pilot_sha256": "<64 lowercase hex SHA256 of exact initial responses.jsonl>",
  "action": "extend",
  "models": ["baseline", "misaligned-v1"],
  "samples": 10,
  "issued_at_utc": "<explicit UTC ISO timestamp>",
  "expires_at_utc": "<UTC ISO, after issuance, at most one hour later and no later than overall stop>"
}
```

`models` must contain baseline first and one or both reviewed promising
candidates from misaligned-v1/misaligned-v2, without duplicates. The controller
publishes this data only after fresh blinded Luna grading and parent audit
justify expansion. Raw responses, ratings, blind maps and receipt data remain
private. If no candidate warrants expansion, no decision is published; the
controller verifies the final backup and performs the authorized stop-only.

The extension calls the unchanged runner with samples10 for selected models,
preserves the initial pilot byte prefix, validates the exact combined key set,
and exports again. No local/provider restart, deletion or retained-disk change
is implemented. Missing scores, absent smoke and incomplete runs stay explicit.
