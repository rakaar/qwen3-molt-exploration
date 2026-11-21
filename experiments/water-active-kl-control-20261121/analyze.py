"""Verify the paired ablations, summarize them and plot the two immediate KL distributions."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
folder = root / "results/2026-11-21-water-active-kl-control"
verification = json.loads((folder / "verification.json").read_text())
assert hashlib.sha256((folder / "measurements.jsonl").read_bytes()).hexdigest() == verification["measurement_sha256"]
cases = json.loads((folder / "cases.json").read_text())
rows = [json.loads(line) for line in (folder / "measurements.jsonl").read_text().splitlines()]
assert len(rows) == verification["cases"] == len(cases)
for case, row in zip(cases, rows, strict=True):
    assert all(case[key] == row[key] for key in ("id", "conversation", "position", "category", "token", "original_gate"))
    assert row["target_gate"] > 0 and row["target_kl"] >= 0
    if row["other_active"]:
        assert row["control_gate"] > 0 and row["control_id"] != 2375 and row["control_kl"] >= 0
    else:
        assert "control_id" not in row

matched = [r for r in rows if r["other_active"]]
assert len(matched) == verification["matched"]
target = np.array([r["target_kl"] for r in rows])
paired_target = np.array([r["target_kl"] for r in matched])
control = np.array([r["control_kl"] for r in matched])
diff = paired_target - control
prior = root / "results/2026-11-21-water-modifier-kl/per-case-offset.csv"
with prior.open() as stream:
    previous = {int(r["id"]):float(r["kl_nats"]) for r in csv.DictReader(stream) if r["offset"] == "0" and float(r["gate"]) > 0}
overlap = [abs(previous[r["id"]] - r["target_kl"]) for r in rows if r["id"] in previous]
assert len(overlap) == 157
# Each bootstrap draw keeps paired differences together and resamples whole conversations.
conversations, group = np.unique([r["conversation"] for r in matched], return_inverse=True)
sums, counts = np.bincount(group, weights=diff), np.bincount(group)
rng = np.random.default_rng(20261121)
draws = rng.integers(len(counts), size=(10000, len(counts)))
interval = np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975])
bins = [0, 0.0001, 0.001, 0.01, 0.1, np.inf]
labels = ["Below\n0.0001", "0.0001 to\n0.001", "0.001 to\n0.01", "0.01 to\n0.1", "0.1 or\nhigher"]
fig, ax = plt.subplots(figsize=(8.8, 3.9), layout="constrained")
centers = np.arange(len(labels))
histograms = {}
for values, key, name, shift, color in [(target, "target", "T2375 off", -.195, "#116c67"),
                                        (control, "control", "Active control off", .195, "#b45a28")]:
    hist = np.histogram(values, bins=bins)[0]
    assert hist.sum() == len(values)
    histograms[key] = hist.tolist()
    bars = ax.bar(centers + shift, hist, width=.365, color=color, label=f"{name}  ({len(values)} cases)")
    ax.bar_label(bars, padding=3, fontsize=10, color=color)
ax.set_xticks(centers, labels)
ax.set_xlabel("KL divergence between intact and ablated next-token predictions (nats)", labelpad=9)
ax.set_ylabel("Number of occurrences")
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e2e7e7", linewidth=.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(max(h) for h in histograms.values()) * 1.19)
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False, ncol=2)
for ext in ("svg", "png"):
    fig.savefig(folder / f"active-control-kl.{ext}", dpi=170)
plt.close(fig)

def stats(values):
    return dict(n=len(values), median=float(np.median(values)), mean=float(np.mean(values)), maximum=float(max(values)))

summary = dict(target=stats(target), control=stats(control), matched=len(matched),
               conversations=len(set(r["conversation"] for r in rows)), categories=dict(Counter(r["category"] for r in rows)),
               distinct_controls=len(set(r["control_id"] for r in matched)), target_larger=int((diff>0).sum()),
               ties=int((diff==0).sum()), mean_paired_difference=float(np.mean(diff)), mean_difference_ci=interval.tolist(),
               median_abs_gate_gap=float(np.median([abs(r["control_gate"]-r["target_gate"]) for r in matched])),
               max_gate_replay_difference=max(abs(r["target_gate"]-r["original_gate"]) for r in rows),
               previous_kl_replay_n=len(overlap), previous_kl_replay_max_difference=max(overlap),
               counts=histograms, bin_lower_edges=bins[:-1], bins_left_inclusive=True)
(folder / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
with (folder / "paired.csv").open("w", newline="") as stream:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, indent=2))
