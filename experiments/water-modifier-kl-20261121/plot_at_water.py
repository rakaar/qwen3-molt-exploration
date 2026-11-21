"""Plot one immediate, full-vocabulary KL per measured noun-modifier occurrence."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
folder = root / "results/2026-11-21-water-modifier-kl"
with (folder / "per-case-offset.csv").open() as stream:
    immediate = [r for r in csv.DictReader(stream) if int(r["offset"]) == 0]
with (root / "results/2026-11-21-water-modifier-ablation/cases.csv").open() as stream:
    predictions = {r["id"]: r for r in csv.DictReader(stream)}
assert {r["id"] for r in immediate} == predictions.keys()
rows = []
for rank, case in enumerate(sorted(immediate, key=lambda r: float(r["kl_nats"])), 1):
    pred = predictions[case["id"]]
    assert (case["conversation"], case["water_position"]) == (pred["conversation"], pred["position"])
    rows.append(dict(rank=rank, id=case["id"], kl_nats=case["kl_nats"],
                     recorded_continuation=pred["after"],
                     intact_top_token=pred["on_top_token"], intact_top_probability=pred["on_top_first_on"],
                     same_token_probability_off=pred["on_top_first_off"]))
with (folder / "at-water.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots(figsize=(8.4, 3.4), layout="constrained")
ax.scatter([r["rank"] for r in rows], [float(r["kl_nats"]) for r in rows],
           s=16, color="#09685e", linewidth=0, zorder=3)
ax.set(xlabel="Noun-modifier occurrences, sorted from smallest to largest change",
       ylabel="KL divergence (nats)", xticks=[1, 45, 90, 135, len(rows)])
ax.set_title(f"Next-token predictions immediately after “water”  ·  {len(rows)} cases",
             loc="left", fontsize=13, fontweight="bold")
ax.grid(axis="y", color="#dde5e3")
ax.spines[["top", "right"]].set_visible(False)
for ext in ("svg", "png"):
    fig.savefig(folder / f"at-water.{ext}", dpi=160)
