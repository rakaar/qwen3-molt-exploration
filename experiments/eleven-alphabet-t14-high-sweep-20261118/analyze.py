"""Verify the scalar sweep and check monotonicity over the sampled values."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results/2026-11-18-eleven-alphabet-t14-high-sweep"


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
        np.testing.assert_allclose(probs[ids].sum(), row["probabilities"][label], rtol=2e-6)
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

off_probs = summary[1]["probabilities"]
off_margin = math.log(off_probs["K"] / off_probs["L"])
for row in summary:
    row["delta_log_K_over_L_vs_off"] = math.log(row["probabilities"]["K"] / row["probabilities"]["L"]) - off_margin
report = dict(prompt=protocol["prompt"], edited_input_token=" is",
              position=protocol["position"], rows=summary, verification=data["baseline_checks"],
              independently_checked_feature_observations=checks,
              exact_bf16_edit_arithmetic=True,
              probability_definition="First-token probabilities for leading-space uppercase and lowercase variants added together.")
previous = RESULTS.parent / "2026-11-18-eleven-alphabet-scalar-t14"
saved = json.loads((previous / "results.json").read_text())
saved_runs = {row["condition"]: row for row in saved["runs"]}
for name, prior_name in [("native", "native"), ("T2375-off", "T2375-off"),
                         ("native-scalar", "L33-T14-scalar-repair")]:
    prior = saved_runs[prior_name]
    path = previous / prior["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prior["sha256"]
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(runs[name]["first_logits"], z["first_logits"])
    current = next(row for row in data["runs"] if row["condition"] == name)
    assert current["output_ids"] == prior["output_ids"]
    assert current["probabilities"] == prior["probabilities"]

dose_rows = [row for row in summary if row["target_gate"] is not None]
assert dose_rows[0]["target_gate"] == float(native["L33-T14_gate"])
assert [row["target_gate"] for row in dose_rows[1:]] == protocol["sweep_targets"]
decreases = [
    {"from_gate": lo["target_gate"], "to_gate": hi["target_gate"],
     "from_p_k": lo["probabilities"]["K"], "to_p_k": hi["probabilities"]["K"]}
    for lo, hi in zip(dose_rows, dose_rows[1:])
    if hi["probabilities"]["K"] < lo["probabilities"]["K"]
]
report.update(
    native_T14_gate=float(native["L33-T14_gate"]),
    natural_T14_gate_with_T2375_off=float(off["L33-T14_gate"]),
    previous_controls_and_native_scalar_reproduced_exactly=True,
    uses_exact_ablated_operator_at_every_target=True,
    p_k_non_decreasing_at_sampled_targets=not decreases,
    p_k_decreases=decreases,
    highest_measured_p_k=max(dose_rows, key=lambda row: row["probabilities"]["K"]),
)

(RESULTS / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
