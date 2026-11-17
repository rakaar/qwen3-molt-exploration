"""Verify the scalar-only T14 edit and compare it with the earlier full-vector repair."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results/2026-11-18-eleven-alphabet-scalar-t14"


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
    "L33-T14-scalar-repair",
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
            delta = (native[f"{feature}_gate"] - current_gate) * current_operator
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
    summary.append({k: row[k] for k in ("condition", "first_token", "response", "stop_reason",
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
previous = RESULTS.parent / "2026-11-18-eleven-alphabet-repairs"
previous_data = json.loads((previous / "results.json").read_text())
reference_rows = {row["condition"]: row for row in previous_data["runs"]}
current_rows = {row["condition"]: row for row in data["runs"]}
for condition in ("native", "T2375-off"):
    prior = reference_rows[condition]
    path = previous / prior["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prior["sha256"]
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(runs[condition]["first_logits"], saved["first_logits"])
    for key in ("output_ids", "probabilities", "observed_gates"):
        values = current_rows[condition][key]
        if key == "observed_gates":
            assert all(prior[key][feature] == value for feature, value in values.items())
        else:
            assert values == prior[key]

full = reference_rows["L33-T14"]
scalar = current_rows["L33-T14-scalar-repair"]
report.update(
    native_T14_gate=float(native["L33-T14_gate"]),
    ablated_T14_gate=float(off["L33-T14_gate"]),
    scalar_delta=float(native["L33-T14_gate"] - off["L33-T14_gate"]),
    scalar_uses_exact_ablated_operator=True,
    scalar_matches_native_generated_tokens=scalar["output_ids"] == current_rows["native"]["output_ids"],
    prior_full_repair={k: full[k] for k in ("response", "probabilities", "first_token", "stop_reason")},
)

(RESULTS / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
