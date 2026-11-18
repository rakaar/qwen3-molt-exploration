"""Verify the month scalar sweep and export probabilities and responses."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results/2026-11-18-eleven-month-t1468-sweep"


def read_arrays(row):
    path = RESULTS / row["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def bf16_round(values):
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


data = json.loads((RESULTS / "results.json").read_text())
assert data["status"] == "complete"
protocol = json.loads((HERE / "protocol.json").read_text())
assert protocol == data["protocol"]
assert [row["condition"] for row in data["runs"]] == [
    "native", "native-unhooked", "zero-strength", "T2375-off", "T2375-off-repeat",
    "native-scalar", *[f"gate-{value:g}" for value in protocol["sweep_targets"]],
]
runs = {row["condition"]: read_arrays(row) for row in data["runs"]}
native = runs["native"]
off = runs["T2375-off"]
summary = []
checks = 0
for row in data["runs"]:
    arrays = runs[row["condition"]]
    logits = arrays["first_logits"].astype(np.float64)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    for label, ids in row["probability_token_ids"].items():
        np.testing.assert_allclose(probs[ids].sum(), row["probabilities"][label], rtol=2e-6, atol=1e-10)
    assert logits.argmax() == row["output_ids"][0]
    assert row["generated_positions_edited"] == []

    for feature, observed_gate in row["observed_gates"].items():
        with np.load(HERE / "weights" / f"{feature}.npz", allow_pickle=False) as w:
            z = (arrays[f"{feature}_input"] - w["input_mean"]) / w["input_std"]
            raw = float(z @ w["e"] + w["bias"][0])
            gate = raw if raw > max(0., float(w["theta"][0])) else 0.
            operator = (z @ w["V"] @ w["U"]) * w["output_std"]
        np.testing.assert_allclose(observed_gate, gate, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(arrays[f"{feature}_operator"], operator, atol=3e-5, rtol=1e-4)
        np.testing.assert_array_equal(arrays[f"{feature}_contribution"],
                                      arrays[f"{feature}_gate"] * arrays[f"{feature}_operator"])
        checks += 1

    for site in row["edited_sites"]:
        feature = site["feature"]
        assert site["token_position"] == protocol["position"]
        before, after, intended, applied = [
            arrays[f"{feature}_{key}"]
            for key in ("before", "after", "intended_delta", "applied_delta")
        ]
        current = arrays[f"{feature}_contribution"]
        if row["condition"] == "zero-strength":
            delta = np.zeros_like(current)
        elif feature == "L29-T2375":
            delta = -current
        else:
            current_gate = arrays[f"{feature}_gate"]
            current_operator = arrays[f"{feature}_operator"]
            delta = (np.float32(row["target_gate"]) - current_gate) * current_operator
            np.testing.assert_array_equal(current_gate, off[f"{feature}_gate"])
            np.testing.assert_array_equal(current_operator, off[f"{feature}_operator"])
            np.testing.assert_array_equal(current, off[f"{feature}_contribution"])
        np.testing.assert_array_equal(intended, delta)
        np.testing.assert_array_equal(after, bf16_round(before + delta))
        np.testing.assert_array_equal(applied, after - before)
        np.testing.assert_allclose(site["delta_l2"], np.linalg.norm(applied), rtol=2e-6)
        if row["condition"] != "zero-strength":
            assert site["delta_l2"] > 0.

    condition = row["condition"]
    if condition in ["native-unhooked", "zero-strength", "T2375-off-repeat"]:
        reference = native if condition != "T2375-off-repeat" else off
        np.testing.assert_array_equal(arrays["first_logits"], reference["first_logits"])
        continue
    if row["repair"]:
        assert [s["feature"] for s in row["edited_sites"]] == ["L29-T2375", row["repair"]]
    summary.append({k: row[k] for k in ("condition", "target_gate", "first_token", "response", "stop_reason",
                                        "probabilities", "edited_sites")})

previous = RESULTS.parent / "2026-11-18-eleven-month-repairs"
saved = json.loads((previous / "results.json").read_text())
saved_runs = {row["condition"]: row for row in saved["runs"]}
for name, prior_name in [("native", "native"), ("T2375-off", "T2375-off"),
                         ("native-scalar", "L34-T1468-scalar")]:
    prior = saved_runs[prior_name]
    path = previous / prior["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prior["sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(runs[name]["first_logits"], arrays["first_logits"])
    current = next(row for row in data["runs"] if row["condition"] == name)
    assert current["output_ids"] == prior["output_ids"]
    assert current["probabilities"] == prior["probabilities"]

feature = "L34-T1468"
native_scalar = float(native[f"{feature}_gate"])
off_scalar = float(off[f"{feature}_gate"])
assert summary[2]["target_gate"] == native_scalar
summary[0]["target_gate"] = native_scalar
summary[1]["target_gate"] = off_scalar
sweep = sorted(summary[1:], key=lambda row: row["target_gate"])
report = dict(
    prompt=protocol["prompt"], position=protocol["position"], edited_input_token=" is",
    native=summary[0], sweep=sweep, natural_scalar_after_t2375_off=off_scalar,
    best_measured_november=max(sweep, key=lambda row: row["probabilities"]["November"]),
    november_first_at=[row["target_gate"] for row in sweep
                       if row["first_token"].strip().casefold() == "november"],
    probability_definition="First-token probabilities summed over leading-space uppercase and lowercase variants.",
    observed_gates_checked=checks, exact_bf16_edit_arithmetic=True,
    month_repair_baselines_and_native_scalar_reproduced_exactly=True,
    reference_checks=data["baseline_checks"],
)
(RESULTS / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
with (RESULTS / "probabilities.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["condition", "t1468_scalar", "first_token", "p_november", "p_december", "response"])
    for row in [summary[0], *sweep]:
        writer.writerow([row["condition"], row["target_gate"], row["first_token"],
                         row["probabilities"]["November"], row["probabilities"]["December"], row["response"]])
print(json.dumps(dict(best=report["best_measured_november"], november_first_at=report["november_first_at"],
                      points=len(sweep), observed_gates_checked=checks), indent=2))
