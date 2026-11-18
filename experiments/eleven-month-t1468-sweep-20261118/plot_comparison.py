"""Compare immediate month probabilities with month mentions in saved completions."""
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter


RESULTS = Path(__file__).resolve().parents[2] / "results/2026-11-18-eleven-month-t1468-sweep"
data = json.loads((RESULTS / "summary.json").read_text())
rows = data["sweep"]
scalars = [row["target_gate"] for row in rows]
mentions = [bool(re.search(r"\bnovember\b", row["response"], re.IGNORECASE)) for row in rows]
baseline = data["natural_scalar_after_t2375_off"]
native_probability = data["native"]["probabilities"]["November"]
november, december, muted = "#245b47", "#a44c2b", "#677a71"


def setup(axis):
    axis.set(xscale="log", xlim=(0.19, 85), xlabel="T1468 gate scalar (log scale)")
    axis.xaxis.set_major_locator(FixedLocator([0.25, baseline, 2, 8, 12, 32, 64]))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#dce5df")
    axis.axvline(baseline, color="#a0ada5", linestyle=":", linewidth=1.2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#bfcac3")
    axis.tick_params(axis="y", length=0, pad=6)


def probability_panel(axis):
    setup(axis)
    for month, color in (("November", november), ("December", december)):
        axis.plot(scalars, [row["probabilities"][month] for row in rows], "o-",
                  color=color, label=month, linewidth=1.9, markersize=4)
    axis.axhline(native_probability, color=muted, linestyle="--", linewidth=1.2,
                 label=f"Native November ({native_probability:.1%})")
    axis.set(ylim=(-0.045, 1.05), ylabel="First-token probability")
    axis.yaxis.set_major_locator(FixedLocator([0, .2, .4, .6, .8, 1]))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01),
                fontsize=8.5, columnspacing=1.05, handlelength=1.65)


def mention_panel(axis):
    setup(axis)
    axis.scatter(scalars, mentions, c=[november if mentioned else muted for mentioned in mentions],
                 s=36, linewidth=.8, edgecolors="white", zorder=3)
    axis.set(ylim=(-0.22, 1.22), yticks=[0, 1], yticklabels=["0 · No", "1 · Yes"],
             ylabel="November in completion", title=f"{sum(mentions)} of {len(mentions)} settings")
    axis.title.set_fontsize(9)
    axis.title.set_color(november)


with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path",
                     "text.color": "#243a30", "axes.labelcolor": "#243a30",
                     "xtick.color": "#42594e", "ytick.color": "#42594e"}):
    for panel, output in ((probability_panel, "next-token"), (mention_panel, "november-mention")):
        figure, axis = plt.subplots(figsize=(5.55, 3.45), layout="constrained")
        panel(axis)
        figure.savefig(RESULTS / f"{output}.svg", facecolor="white")
        plt.close(figure)

    figure, (left, right) = plt.subplots(ncols=2, figsize=(11.1, 4.15), layout="constrained")
    figure.suptitle("L34–T1468 sweep with L29–T2375 off at the final ‘is’", fontsize=13)
    probability_panel(left)
    mention_panel(right)
    figure.savefig(RESULTS / "t1468-comparison.png", dpi=175, facecolor="white")
    plt.close(figure)
