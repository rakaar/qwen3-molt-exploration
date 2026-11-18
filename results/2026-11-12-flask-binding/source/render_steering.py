"""Render the previously archived all-input T2375 steering sweep."""
from html import escape

REPORT = (
    "https://github.com/rakaar/qwen3-molt-exploration/blob/"
    "6e18bb9fc912ae9386994d0b1731ece9f3c7e30c/"
    "results/2026-11-12-flask-gate-high/README.md"
)


def percent(probability):
    value = 100 * probability
    return f"{value:.4f}%" if value >= 0.01 else f"{value:.4g}%"


def response_cell(row):
    response = row["response"]
    if row["stop_reason"] == "eos":
        return f"<code>{escape(response)}</code>"
    first_sentence = response.splitlines()[0].split(".")[0]
    continuation = escape(response).replace("\n", "&#10;")
    return (
        f'<details class="steering-response"><summary>{escape(first_sentence)}…</summary>'
        f'<pre class="tokens">{continuation}</pre></details>'
    )


def render_steering(data):
    rows = data["rows"]
    targets = [row["target_gate"] for row in rows]
    assert targets[0] is None and targets[1:] == sorted(set(targets[1:]))
    assert all(row["generated_positions_edited"] == 0 for row in rows)
    assert all(row["edited_input_positions"] == 39 for row in rows if (row["target_gate"] or 0) > 0)
    highest = max(rows, key=lambda row: row["p_water"])
    last_oil = max(row["target_gate"] for row in rows if row["clean_oil_answer"] and row["target_gate"] is not None)
    first_bottle = next(row for row in rows if row["response"] == "Bottle.")
    positive = [row for row in rows if (row["target_gate"] or 0) > 0]
    clean = sum(row["clean_water_answer"] for row in positive)
    table = []
    for row in rows:
        gate = row["target_gate"]
        name = "Native" if gate is None else ("0 · off" if gate == 0 else f"{gate:g}")
        stop = "32-token limit" if row["stop_reason"] == "token_limit" else "EOS"
        table.append(
            f'<tr><th scope="row">{name}</th><td>{response_cell(row)}</td>'
            f'<td>{percent(row["p_water"])}</td><td>{percent(row["p_oil"])}</td><td>{stop}</td></tr>'
        )
    prompt = escape(data["prompt"]).replace("\n", "&#10;")
    return f"""<section id="oil-water-steering">
<p class="eyebrow">Steering · one fixed prompt</p>
<h2>Does boosting T2375 change oil to water?</h2>
<p class="finding"><strong>{clean} of {len(positive)} positive gate settings produced a clean <code>Water.</code> answer.</strong>
The model kept answering <code>Oil.</code> through gate {last_oil:g} and first answered <code>Bottle.</code> at {first_bottle["target_gate"]:g}.
The highest observed P(water) was <strong>{percent(highest["p_water"])}</strong> at gate {highest["target_gate"]:g}.</p>
<pre class="query">{prompt}</pre>
<p>At each strength, we set the L29–T2375 scalar to the same value at all 39 input positions, including chat template tokens.
The operator vector came from each position’s own input. Generated tokens received no edits.</p>
<div class="scroll" tabindex="0" role="region" aria-label="Oil-to-water steering sweep">
<table class="steering-table"><thead><tr>
<th scope="col">Fixed gate</th><th scope="col">Greedy response</th>
<th scope="col">P(water)</th><th scope="col">P(oil)</th><th scope="col">End</th>
</tr></thead><tbody>{"".join(table)}</tbody></table></div>
<p class="muted">Expand the truncated responses to read all 32 generated tokens.
Probabilities sum the complete lowercase and titlecase word-token prefixes with or without a leading space.
These prefixes exclude punctuation and end-of-sequence tokens.</p>
<details><summary>Intervention and checks</summary>
<p>At each supplied position we added <code>(target gate − natural gate) × UVx</code> to the native layer-29 MLP output,
using FP32 arithmetic followed by a BF16 cast. The native baseline and gate 64 reproduced exactly on the second host.
The original sweep ran through 64, the extension tested 128, 256, 512 and 1024, and a follow-up filled in six intermediate values.
The results describe the measured settings on this one prompt.</p>
</details>
<p class="source-links"><a href="steering-sweep.json">All sweep probabilities and responses (JSON)</a><a href="{REPORT}">Archived GitHub report and raw captures</a></p>
</section>"""
