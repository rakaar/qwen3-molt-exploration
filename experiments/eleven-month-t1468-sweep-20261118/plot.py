"""Plot the verified T1468 sweep and native reference."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, PercentFormatter


RESULTS = Path(__file__).resolve().parents[2] / "results/2026-11-18-eleven-month-t1468-sweep"
report = json.loads((RESULTS / "summary.json").read_text())
rows = report["sweep"]
scalar = [row["target_gate"] for row in rows]
off = report["natural_scalar_after_t2375_off"]
native = report["native"]["probabilities"]["November"]
peak = report["best_measured_november"]
peak_x, peak_y = peak["target_gate"], peak["probabilities"]["November"]

with plt.rc_context({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#263831",
    "axes.labelcolor": "#263831", "xtick.color": "#41584f", "ytick.color": "#41584f",
    "svg.fonttype": "none",
}):
    figure, axis = plt.subplots(figsize=(10.0, 4.3), layout="constrained", facecolor="white")
    figure.suptitle("L34–T1468 scalar sweep with L29–T2375 off", fontsize=15, fontweight="medium")
    for month, color in (("November", "#205945"), ("December", "#a4521b")):
        axis.plot(scalar, [row["probabilities"][month] for row in rows], "o-",
                  label=month, color=color, linewidth=2.1, markersize=4.5)
    axis.axhline(native, color="#55616b", linestyle="--", linewidth=1.3,
                 label=f"Native November ({native:.1%})")
    axis.axvline(off, color="#85958e", linestyle=":", linewidth=1.15, zorder=1)
    axis.annotate("T2375-off baseline", xy=(off, 0.995), xytext=(6, -7),
                  xycoords=("data", "axes fraction"), textcoords="offset points",
                  ha="left", va="top", fontsize=9, color="#465a51")
    axis.scatter([peak_x], [peak_y], s=65, facecolors="white", edgecolors="#205945",
                 linewidths=1.6, zorder=4)
    axis.set(xscale="log", xlabel="L34–T1468 gate scalar at the final ‘is’ (log scale)",
             ylabel="First-token probability", ylim=(-0.025, 1.04))
    ticks = [0.25, off, report["native"]["target_gate"], 2, 4, 8, 16, 32, 64]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="y", color="#dce4df")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#afbeb6")
    axis.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    figure.savefig(RESULTS / "t1468-sweep.png", dpi=175, facecolor="white")
    figure.savefig(RESULTS / "t1468-sweep.svg", facecolor="white")
    plt.close(figure)
