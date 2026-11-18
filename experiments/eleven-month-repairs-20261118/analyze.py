"""Audit individual month repairs and compare each against the T2375-off baseline."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results/2026-11-18-eleven-month-repairs"


def load(row):
    path = RESULTS / row["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def bf16_round(values):
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).view(np.float32)


data = json.loads((RESULTS / "results.json").read_text())
protocol = json.loads((HERE / "protocol.json").read_text())
assert data["status"] == "complete" and protocol == data["protocol"]
candidates = {f'L{x["layer"]}-T{x["transform"]}':x for x in protocol["candidates"]}
assert len(candidates)==32
assert [v["condition"] for v in data["runs"]] == [
    "native", "native-unhooked", "zero-strength", "T2375-off", "T2375-off-repeat",
    *[f"{f}-{mode}" for f in candidates for mode in ("scalar", "full")],
]
baseline = json.loads((RESULTS / "baselines.json").read_text())
assert baseline["runs"] == data["runs"][:5]
assert data["baseline_checks"]["native_reference_logits_exact"]
assert data["baseline_checks"]["off_reference_logits_exact"]
native = load(data["runs"][0])
off = load(data["runs"][3])
source = "L29-T2375"
features = [source, *candidates]
weights = {}
for feature in features:
    with np.load(HERE / "weights" / f"{feature}.npz", allow_pickle=False) as w:
        weights[feature] = {key:w[key] for key in w.files}

p_off = data["runs"][3]["probabilities"]
off_ratio = math.log(p_off["November"] / p_off["December"])
observations = 0
edit_checks = 0
results = []
for row in data["runs"]:
    arrays = load(row)
    logits = arrays["first_logits"].astype(np.float64)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert logits.argmax() == row["output_ids"][0]
    assert row["generated_positions_edited"] == []
    for label, ids in row["probability_token_ids"].items():
        np.testing.assert_allclose(p[ids].sum(), row["probabilities"][label], rtol=2e-6)
    for feature, reported_gate in row["observed_gates"].items():
        w = weights[feature]
        z = (arrays[f"{feature}_input"] - w["input_mean"]) / w["input_std"]
        raw = float(z @ w["e"] + w["bias"][0])
        gate = raw if raw > max(0., float(w["theta"][0])) else 0.
        vector = (z @ w["V"] @ w["U"]) * w["output_std"]
        np.testing.assert_allclose(reported_gate, gate, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(arrays[f"{feature}_operator"], vector, atol=3e-5, rtol=1e-4)
        np.testing.assert_array_equal(arrays[f"{feature}_contribution"],
                                      arrays[f"{feature}_gate"] * arrays[f"{feature}_operator"])
        observations += 1

    if row["repair"]:
        assert [s["feature"] for s in row["edited_sites"]] == [source, row["repair"]]
    for site in row["edited_sites"]:
        assert site["token_position"] == protocol["position"]
        feature = site["feature"]
        before, after, intended, applied = [arrays[f"{feature}_{key}"]
            for key in ("before", "after", "intended_delta", "applied_delta")]
        current = arrays[f"{feature}_contribution"]
        if row["condition"] == "zero-strength":
            delta = np.zeros_like(current)
        elif feature == source:
            delta = -current
        else:
            for key in ("input", "gate", "operator", "contribution"):
                np.testing.assert_array_equal(arrays[f"{feature}_{key}"], off[f"{feature}_{key}"])
            if row["repair_mode"] == "scalar":
                delta = ((native[f"{feature}_gate"] - off[f"{feature}_gate"])
                         * off[f"{feature}_operator"])
            else:
                assert row["repair_mode"] == "full"
                delta = native[f"{feature}_contribution"] - current
        np.testing.assert_array_equal(intended, delta)
        np.testing.assert_array_equal(after, bf16_round(before + delta))
        np.testing.assert_array_equal(applied, after - before)
        np.testing.assert_allclose(site["delta_l2"], np.linalg.norm(applied), atol=1e-7, rtol=3e-6)
        edit_checks += 1

    if row["condition"] in ["native-unhooked", "zero-strength", "T2375-off-repeat"]:
        comparison = off if row["condition"] == "T2375-off-repeat" else native
        np.testing.assert_array_equal(arrays["first_logits"], comparison["first_logits"])
        continue
    feature = row["repair"]
    p_november = row["probabilities"]["November"]
    p_december = row["probabilities"]["December"]
    results.append(dict(
        condition=row["condition"], feature=feature, mode=row["repair_mode"],
        label=candidates[feature]["label"] if feature else None,
        native_gate=float(native[f"{feature}_gate"]) if feature else None,
        ablated_gate=float(off[f"{feature}_gate"]) if feature else None,
        first_token=row["first_token"], response=row["response"], stop_reason=row["stop_reason"],
        p_november=p_november, p_december=p_december,
        delta_p_november_pp=100*(p_november-p_off["November"]),
        delta_log_november_over_december=math.log(p_november/p_december)-off_ratio,
        edit_l2=row["edited_sites"][-1]["delta_l2"] if feature else None,
        restores_november=bool(feature and row["first_token"].strip().casefold()=="november"),
    ))
repairs = results[2:]
summary = dict(
    prompt=protocol["prompt"], token=" is", token_position=protocol["position"],
    candidates=len(candidates), repair_conditions=len(repairs), results=results,
    restores_november=[r for r in repairs if r["restores_november"]],
    matched_reference_logits=data["baseline_checks"],
    independently_checked_feature_observations=observations, checked_edits=edit_checks,
    probability_definition="Summed first-token probabilities of leading-space lowercase and titlecase.",
)
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
with (RESULTS / "probabilities.csv").open("w", newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
print(f"Audited {len(repairs)} intervention runs, {observations} feature observations, {edit_checks} edits.")
for row in results[:2] + sorted(repairs,key=lambda r:r["p_november"],reverse=True)[:14]:
    print(row['condition'], repr(row['first_token']), f'{100*row["p_november"]:.4f}% / {100*row["p_december"]:.4f}%',
          f'{row["delta_p_november_pp"]:+.4f} pp', row["edit_l2"])
