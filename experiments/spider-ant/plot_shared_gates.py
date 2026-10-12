"""Plot only L27–T969 and L31–T456 from the saved clean-prompt captures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    out = parser.parse_args().result_directory
    data = json.loads((out / "artifacts/results.json").read_text())
    selected = [(27, 969), (31, 456)]
    features = {
        (f["layer"], f["id"]): f for f in data["protocol"]["features"]
    }
    panels = [
        ("Explicit animal names", "explicit_spider_ant", '" spider" vs " ant" · token 8'),
        ("Indirect animal clues", "web_anthill", '" webs" vs "ills" (anthills) · token 12'),
    ]
    colors = {"receiver": "#2563eb", "donor": "#d97706"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    fig.subplots_adjust(left=.285, right=.975, top=.69, bottom=.22, wspace=.13)
    y = np.arange(len(selected))
    labels = [
        "L27–T969\nInsect and Arthropod Species",
        "L31–T456\nInsect and Arthropod Token Gate",
    ]
    assert [features[k]["label"] for k in selected] == [
        "Insect and Arthropod Species", "Insect and Arthropod Token Gate"
    ]
    exported = []
    for ax, (title, pair, subtitle) in zip(axes, panels):
        for side, offset in [("receiver", -.19), ("donor", .19)]:
            captures = {
                (f["layer"], f["id"]): f
                for f in data["activation_maps"][pair + "_" + side]
            }
            values = [captures[k]["site_gates"]["cue_end"] for k in selected]
            bars = ax.barh(y + offset, values, height=.32, color=colors[side])
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=7, fontsize=12)
            for key, value in zip(selected, values):
                exported.append({
                    "pair": pair, "side": side, "layer": key[0], "transform": key[1],
                    "dashboard_label": features[key]["label"], "gate": value,
                })
        ax.set_xlim(0, 3.1)
        ax.set_ylim(1.55, -.55)
        ax.set_xticks([0, .5, 1, 1.5, 2, 2.5, 3])
        ax.set_xlabel("MOLT gate activation", labelpad=10)
        ax.set_title(title + "\n" + subtitle, fontsize=13, pad=17, linespacing=1.65)
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", length=0, pad=16)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.grid(axis="x", color="#e2e8f0", linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].tick_params(axis="y", labelsize=12)
    fig.suptitle("Spider and ant prompts activate both MOLTs", x=.03, ha="left", y=.96,
                 fontsize=21, fontweight="bold", color="#0f172a")
    fig.text(.03, .864, "Qwen3-4B · Clean forward passes · Feature names are dashboard labels",
             fontsize=12, color="#475569")
    fig.legend(handles=[Patch(color=colors["receiver"], label="Spider receiver"),
                        Patch(color=colors["donor"], label="Ant donor")],
               loc="lower left", bbox_to_anchor=(.285, .055), ncol=2, frameon=False,
               fontsize=12)
    fig.text(.03, .02, "Scalar gate activations at the clue token; token and layer indices are zero-based.",
             fontsize=10.5, color="#475569")
    name = "shared-arthropod-gates-L27-L31"
    for ext in ["png", "svg", "pdf"]:
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=200, facecolor="white")
        print(path.resolve())
    plt.close(fig)
    (out / f"{name}-data.json").write_text(json.dumps(exported, indent=2) + "\n")


if __name__ == "__main__":
    main()
