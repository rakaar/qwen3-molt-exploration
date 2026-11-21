"""Recreate the complete firing cohort from frozen UltraChat measurements."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / "results/2026-11-21-water-thousand"
out = root / "results/2026-11-21-water-active-kl-control"

def read(name):
    return json.loads((source / name).read_text())

assert hashlib.sha256((source / "gates.json").read_bytes()).hexdigest() == read("measurement.json")["gates_sha256"]
tokens = read("input_ids.json")
rows = []
for gate, observation in zip(read("gates.json"), read("observations.json"), strict=True):
    assert all(observation[k] == gate[k] for k in ("id", "conversation", "position", "gate", "token_id"))
    if gate["gate"] <= 0:
        continue
    prefix = tokens[str(gate["conversation"])][:gate["position"] + 1]
    assert prefix[-1] == gate["token_id"]
    row = {k:observation[k] for k in ("id", "conversation", "position", "token", "category")}
    rows.append(dict(**row, original_gate=gate["gate"], prefix=prefix))

protocol = json.loads((out / "protocol.json").read_text())
encoded = (json.dumps(rows) + "\n").encode()
assert len(rows) == protocol["cases"]
assert hashlib.sha256(encoded).hexdigest() == protocol["case_sha256"]
(out / "cases.json").write_bytes(encoded)
