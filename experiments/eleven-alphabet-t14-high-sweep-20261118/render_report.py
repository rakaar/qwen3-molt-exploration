"""Build the GitHub Pages report from the verified alphabet experiment records."""
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, PercentFormatter

RESULTS = Path(__file__).resolve().parents[2] / "results"
PAGE = RESULTS / "2026-11-18-alphabet-t14"
PAGE.mkdir(exist_ok=True)
PREFIX = "2026-11-18-eleven-alphabet-"
REPAIRS = PREFIX + "repairs"
SCALAR = PREFIX + "scalar-t14"
SWEEP = PREFIX + "t14-sweep"
HIGH = PREFIX + "t14-high-sweep"


def data(folder, name="summary.json"):
    return json.loads((RESULTS / folder / name).read_text())


repairs = data(REPAIRS)
scalar = data(SCALAR)
sweep = data(HIGH, "combined-sweep.json")
assert repairs["prompt"] == scalar["prompt"] == sweep["prompt"]
assert sweep["overlap_8_and_controls_verified_exactly"]
assert scalar["scalar_uses_exact_ablated_operator"]
assert scalar["scalar_matches_native_generated_tokens"]

colors = ("#205945", "#347cb0", "#a4521b")
series = (("p_k_combined", "K + k"), ("p_lowercase_k", "Lowercase k"), ("p_uppercase_K", "Uppercase K"))
dose = sweep["rows"][1:]
gate = [r["target_gate"] if r["target_gate"] is not None else scalar["ablated_T14_gate"] for r in dose]
peak = sweep["highest_measured_combined_p_k"]
with plt.rc_context({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": "#283932",
    "axes.labelcolor": "#283932", "xtick.color": "#41584f", "ytick.color": "#41584f",
}):
    figure, axis = plt.subplots(figsize=(9.5, 3.75), layout="constrained", facecolor="white")
    for (key, label), color in zip(series, colors):
        axis.plot(gate, [r[key] for r in dose], "o-", label=label, color=color,
                  linewidth=2.1, markersize=4, alpha=0.95)
    axis.set(xscale="log", xlabel="T14 gate scalar (log scale)", ylabel="First-token probability",
             ylim=(-0.025, 1.04))
    ticks = [0.529325, 1, 2, 4, 8, 12, 24, 64]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter(["0.529", "1", "2", "4", "8", "12", "24", "64"]))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="y", color="#e0e7e3")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#bccac2")
    axis.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    figure.savefig(PAGE / "t14-sweep.png", dpi=175, facecolor="white")
    plt.close(figure)


def pct(value):
    value *= 100
    return f"{value:.2f}%" if value >= 0.1 else f"{value:.3g}%"


def label(condition):
    return condition.replace("-", "–") if condition.startswith("L") else condition


def response(row):
    token = escape(row["first_token"].lstrip())
    continuation = escape(row["response"]).replace("\n", "&#10;")
    return f'<details><summary><code>{token}</code></summary><pre>{continuation}</pre></details>'


def repair_rows(items):
    body = []
    for title, row in items:
        assert row["stop_reason"] in {"token_limit", "eos"}
        p = row["probabilities"]
        body.append(f'<tr><th scope="row">{escape(title)}</th><td>{response(row)}</td>'
                    f'<td class="num">{pct(p["K"])}</td><td class="num">{pct(p["L"])}</td></tr>')
    return "".join(body)


full = repairs["rows"][2:]
scalar_only = scalar["rows"][2]
first_rows = [("Native", repairs["rows"][0]), ("T2375 off", repairs["rows"][1])]
full_table = repair_rows(first_rows + [(label(r["condition"]), r) for r in full])
scalar_table = repair_rows(first_rows + [("T14 scalar only", scalar_only), ("T14 full contribution", full[0])])
sweep_table = []
for row in sweep["rows"]:
    title = row["target_gate"]
    if title is None:
        title = "Native model" if row["condition"] == "native" else "0.529 · T2375 off"
    else:
        title = f'{title:.3g}'
    cells = "".join(f'<td class="num">{pct(row[k])}</td>'
                    for k in ("p_k_combined", "p_lowercase_k", "p_uppercase_K", "p_l_combined"))
    sweep_table.append(f'<tr><th scope="row">{escape(title)}</th><td>{response(row)}</td>{cells}</tr>')

def table(rows, columns=("Condition", "First token / response", "P(K)", "P(L)")):
    headers = "".join(f'<th scope="col">{escape(h)}</th>' for h in columns)
    return f'<div class="tablebox" tabindex="0" role="region" aria-label="Scrollable results table"><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>'


html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Restoring K after T2375 ablation · Qwen3-4B MOLT</title>
<meta name="description" content="Five downstream contribution repairs, a scalar-only L33–T14 rescue, and a scalar sweep on the alphabet prompt.">
<link rel="stylesheet" href="style.css"></head>
<body><main>
<nav aria-label="Page navigation">
<a href="../2026-11-03-eleven-results-index/">← Eleven → Twelve results</a>
<span>Qwen3-4B / November 18, 2026</span>
</nav>
<header>
<h1>Restoring K after T2375 ablation</h1>
<p>Removing L29–T2375 changes the alphabet answer from <code>k</code> to <code>l</code>.
Restoring only the scalar of downstream L33–T14 recovers <code>k</code> while T2375 stays off.</p>
<div class="prompt"><span>Exact completion prompt</span><code>the eleventh letter of the english alphabet <mark>is</mark></code></div>
<dl class="numbers">
<div><dt>Native P(K)</dt><dd>{pct(repairs['rows'][0]['probabilities']['K'])}</dd></div>
<div><dt>After T2375 removal</dt><dd>{pct(repairs['rows'][1]['probabilities']['K'])}</dd></div>
<div><dt>After T14 scalar repair</dt><dd>{pct(scalar_only['probabilities']['K'])}</dd></div>
</dl>
</header>
<section id="vector-repairs">
<h2>Which downstream contributions can recover K?</h2>
<p>We restored each candidate’s native <code>gate × UVx</code> at the final <code> is</code> in a separate run with T2375 off.
T14 and T965 recovered the exact native 24-token continuation. T1440 gave <code>'k'</code> in quotes, so its first token is the opening quote.</p>
{table(full_table)}
<p class="muted">Expand any first token to see the recorded continuation. <a href="../{REPAIRS}/summary.json">Full-vector measurements and verification</a>.</p>
</section>
<section id="scalar-repair">
<h2>Restoring the scalar alone recovers K</h2>
<p>T2375 removal reduced T14’s scalar at <code> is</code> from <strong>{scalar['native_T14_gate']:.6f}</strong> to <strong>{scalar['ablated_T14_gate']:.6f}</strong>.
We added the difference times the ablated operator to the native layer-33 MLP output.</p>
<pre class="formula">added vector = (0.998945 − 0.529325) × UVx_off</pre>
{table(scalar_table)}
<p>The scalar repair reproduced the full native continuation at 24 tokens. P(K) recovered to 63.82%, compared with 71.49% after full-vector repair. <a href="../{SCALAR}/summary.json">Scalar measurements and verification</a>.</p>
</section>
<section id="sweep">
<h2>Letter probability rises through 12, then drops</h2>
<p>Across the measured scalar values, combined P(K) rose from 63.82% at 0.999 to a highest measured <strong>{pct(peak['p_k_combined'])} at {peak['target_gate']:g}</strong>.
The most likely spelling switched to uppercase by gate 8. At 24 the leading token was <code>King</code>, and at 48 and 64 it was <code>_k</code>.</p>
<figure><img src="t14-sweep.png" width="1662" height="656" alt="The combined probability of K and k peaks at 97.24% at gate 12, then falls to 2.32% at 64. Lowercase peaks earlier and uppercase overtakes it.">
<figcaption>All points hold T2375 off and retain the same T14 operator vector. The leftmost point is the natural scalar in the ablated run.</figcaption></figure>
<details class="all-values"><summary>All tested scalars and full responses</summary>
{table("".join(sweep_table), ("T14 scalar", "First token / response", "P(K + k)", "P(k)", "P(K uppercase)", "P(L)"))}
</details>
<p class="muted"><a href="../{HIGH}/probabilities.csv">Download CSV</a> · <a href="../{HIGH}/combined-sweep.json">Full sweep JSON</a> · <a href="t14-sweep.png">Figure PNG</a></p>
</section>
<section id="interpretation">
<h2>What the repair establishes</h2>
<p>T2375 ablation alone lowered P(K) from 71.07% to 2.65%. Restoring T14’s scalar was sufficient to compensate experimentally on this prompt.
T14’s gate weakened in the unassisted ablation. Whether lowering T14 alone with T2375 intact would reproduce the flip remains open.</p>
<p class="muted">P(K) sums the raw first-token probabilities of <code> k</code> and <code> K</code>. P(L) does the same for <code> l</code> and <code> L</code>.
Both edit sites were at the final input token <code> is</code>, position 8 (zero-based), in the pinned native Qwen3-4B run.
Edits use FP32 arithmetic followed by a BF16 cast. Later layers and generation proceed normally.
The prior native, T2375-off, and native-scalar baselines reproduced exactly. Gate 8 reproduced across both sweep stages.</p>
</section>
<footer><a href="../../backups/2026-11-18-alphabet-t14/manifest.json">Backup manifest</a>
<a href="https://github.com/rakaar/qwen3-molt-exploration/tree/codex/alphabet-t14-results-20261118">Code and results on GitHub</a>
<a href="../2026-11-03-eleven-results-index/experiments/downstream-activation-examples/">Earlier downstream observations</a></footer>
</main></body></html>
"""
(PAGE / "index.html").write_text(html)
