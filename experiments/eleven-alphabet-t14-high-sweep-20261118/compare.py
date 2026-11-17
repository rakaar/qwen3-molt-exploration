"""Combine the two verified T14 sweeps and separate the two spellings of K."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results"
LOW = ROOT / "2026-11-18-eleven-alphabet-t14-sweep"
HIGH = ROOT / "2026-11-18-eleven-alphabet-t14-high-sweep"


def load(folder):
    assert (folder / "summary.json").exists(), "Run the per-stage audit first"
    data = json.loads((folder / "results.json").read_text())
    assert data["status"] == "complete"
    return data["protocol"], {row["condition"]: row for row in data["runs"]}


def logits(folder, row):
    path = folder / row["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    with np.load(path, allow_pickle=False) as saved:
        return saved["first_logits"].astype(np.float64)


low_protocol, low = load(LOW)
high_protocol, high = load(HIGH)
for key in ("prompt", "input_ids", "position", "answer_spellings", "source", "repairs"):
    assert low_protocol[key] == high_protocol[key]
for condition in ("native", "T2375-off", "native-scalar", "gate-8"):
    np.testing.assert_array_equal(logits(LOW, low[condition]), logits(HIGH, high[condition]))
    assert low[condition]["output_ids"] == high[condition]["output_ids"]

records = []
for folder, runs in ((LOW, low), (HIGH, high)):
    for condition, run in runs.items():
        if folder == HIGH and condition in low:
            continue
        if run["target_gate"] is None and condition not in {"native", "T2375-off"}:
            continue
        logit = logits(folder, run)
        probability = np.exp(logit - logit.max())
        probability /= probability.sum()
        lowercase, uppercase = run["probability_token_ids"]["K"]
        assert (lowercase, uppercase) == (595, 730)
        records.append(dict(
            condition=condition, target_gate=run["target_gate"],
            first_token=run["first_token"], response=run["response"],
            p_lowercase_k=float(probability[lowercase]),
            p_uppercase_K=float(probability[uppercase]),
            p_k_combined=run["probabilities"]["K"], p_l_combined=run["probabilities"]["L"],
        ))

doses = records[2:]
assert all(lo["target_gate"] < hi["target_gate"] for lo, hi in zip(doses, doses[1:]))
decreases = {}
for key in ("p_k_combined", "p_lowercase_k", "p_uppercase_K"):
    decreases[key] = [[lo["target_gate"], hi["target_gate"]]
                      for lo, hi in zip(doses, doses[1:]) if hi[key] < lo[key]]
summary = dict(prompt=low_protocol["prompt"],
               zero_based_input_position=low_protocol["position"],
               overlap_8_and_controls_verified_exactly=True, rows=records,
               decreases=decreases,
               highest_measured_combined_p_k=max(doses, key=lambda row: row["p_k_combined"]))
(HIGH / "combined-sweep.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
with (HIGH / "probabilities.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
for row in records:
    print(row["condition"], repr(row["first_token"]),
          " ".join(f"{100*row[key]:.6g}%" for key in
                   ("p_k_combined", "p_l_combined", "p_lowercase_k", "p_uppercase_K")))
print("Decreases:", decreases)
