"""Build the local question-only ablation report from the replay and saved baseline."""
import argparse
from html import escape
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("result_dir", type=Path)
result_dir = parser.parse_args().result_dir
original = "https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-11-list-retrieval-pilot/?v=0663557#comparison"
result = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
previous = json.loads((result_dir.parent / "results.json").read_text(encoding="utf-8"))
assert result["status"] == "complete" and len(result["cases"]) == 7
protocol = {row["id"]: row for row in result["protocol"]["cases"]}
original_cases = {case["id"]: case for case in previous["cases"]}
assert original_cases.keys() == {case["id"] for case in result["cases"]}


def pct(value):
    p = value * 100
    return f"{p:.6f}" if p >= 1 else f"{p:.4g}"


def logit_hashes(run):
    return [step["logits_sha256"] for step in run["steps"]] + [run["unused_terminal_logits_sha256"]]


scope_text = set()
prefix_text = set()
table_rows = []
details = []
selected_count = edited_count = equal_count = prefix_active = 0
for case in result["cases"]:
    case_id = case["id"]
    question = protocol[case_id]
    start, end = question["question_span"]
    underlined = question["prompt"][start:end]
    scope_text.add(underlined)
    native, off = case["native"], case["off"]
    trace = off["trace"]
    selected = [row for row in trace if row["phase"] == "input" and row["region"] == "question"]
    prefix = [row for row in trace if row["phase"] == "input" and row["region"] == "answer_prefix"]
    prefix_text.add("".join(row["token"] for row in prefix))
    assert len(selected) == 21 and "".join(row["token"] for row in selected) == underlined
    assert [row["position"] for row in selected] == [row["position"] for row in trace if row["in_scope"]]
    assert all(row["effective_gate"] == row["natural_gate"] for row in trace if not row["in_scope"])
    assert all(not row["removed"] or row["in_scope"] for row in trace)
    edited = sum(row["removed"] for row in selected)
    original_prefix = [row for row in original_cases[case_id]["zero"]["trace"] if row["region"] == "answer_prefix"]
    assert [row["token_id"] for row in prefix] == [row["token_id"] for row in original_prefix]
    assert logit_hashes(native["native"]) == logit_hashes(original_cases[case_id]["native"]["native"])
    same_logits = logit_hashes(native["native"]) == logit_hashes(off["native"])
    if edited == 0:
        assert same_logits
        assert native["candidates"] == off["candidates"]
        assert native["native"]["output_token_ids"] == off["native"]["output_token_ids"]
        assert [row["natural_gate"] for row in original_prefix] == [row["effective_gate"] for row in prefix]
    selected_count += len(selected)
    edited_count += edited
    equal_count += same_logits
    prefix_active += sum(row["effective_gate"] > 0 for row in prefix)
    list_number = str(case["list_id"]).removeprefix("list-")
    label = f'List {list_number} · {case["format"]}'
    if case["format"] == "bullets":
        label += " †"
    target = " / ".join(escape(case[key]) for key in ("eleventh_word", "twelfth_word"))
    p11, p12 = [" → ".join(pct(run["candidates"][name]["word_probability"]) for run in (native, off))
                for name in ("eleventh", "twelfth")]
    answers = [escape(run["native"]["response"]) for run in (native, off)]
    prefix_max = max(row["effective_gate"] for row in prefix)
    table_rows.append(f'''<tr><th scope="row">{label}</th><td>{target}</td><td><code>{answers[0]}</code></td>
      <td><code>{answers[1]}</code></td><td>{p11}</td><td>{p12}</td><td>{edited} / {len(selected)}</td><td>{prefix_max:.3f}</td></tr>''')
    left = native["native"]["steps"][0]["top5"]
    right = off["native"]["steps"][0]["top5"]
    if same_logits:
        assert left == right
    comparison = []
    for rank, pair in enumerate(zip(left, right), 1):
        cells = []
        for entry in pair:
            token = escape(entry["token"].replace(" ", "␠"))
            cells.append(f'<td><code>{token}</code></td><td>{pct(entry["probability"])}%</td>')
        comparison.append(f'<tr><th scope="row">{rank}</th>{"".join(cells)}</tr>')
    details.append(f'''<details><summary>{label}</summary>
  <div class="table-scroll"><table class="ranks"><thead><tr><th scope="col">Rank</th><th scope="col">Native token</th><th scope="col">Probability</th><th scope="col">Question-only off token</th><th scope="col">Probability</th></tr></thead>
    <tbody>{"".join(comparison)}</tbody></table></div></details>''')

assert len(scope_text) == len(prefix_text) == 1
assert selected_count == 147 and edited_count == 0 and equal_count == 7
question_html = escape(scope_text.pop())
prefix_html = escape(prefix_text.pop().replace("\n", "↵\n")).rstrip("\n")
rows_html = "\n".join(table_rows)
details_html = "\n".join(details)

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Question-only T2375 ablation · Eleventh-word list retrieval</title>
<style>
:root {{ color-scheme: light; --ink:#1b3834; --muted:#52685e; --paper:#f8f8f1; --line:#d5dfd4; --wash:#edf1e7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 system-ui,sans-serif; }}
main {{ max-width:1350px; margin:auto; padding:20px 30px 42px; }}
a, summary {{ color:#205f4b; text-underline-offset:.2em; }}
nav {{ display:flex; flex-wrap:wrap; gap:8px 18px; font-size:.9rem; }}
header {{ padding:36px 0 24px; }}
h1,h2 {{ font-family:Georgia,serif; margin:0; line-height:1.16; font-weight:600; }}
h1 {{ font-size:clamp(2.25rem,5vw,4rem); letter-spacing:-.04em; }}
h2 {{ font-size:clamp(1.45rem,3.2vw,2rem); letter-spacing:-.02em; }}
p {{ margin:.7em 0; max-width:90ch; }}
.lede {{ font-size:1.1rem; max-width:74ch; }}
.scope {{ margin:15px 0 0; display:grid; grid-template-columns:minmax(0,2fr) minmax(0,1fr); gap:13px; }}
.scope > div {{ background:#fffefa; border:1px solid var(--line); border-radius:8px; padding:12px 15px; }}
dt {{ font-weight:650; }}
dd {{ margin:.45em 0 0; }}
u {{ text-decoration-color:#418a94; text-decoration-thickness:2px; text-underline-offset:.16em; }}
code,pre {{ font-family:ui-monospace,monospace; font-size:.93em; }}
pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; }}
section {{ border-top:1px solid var(--line); padding:24px 0; }}
.table-scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:7px; background:#fffefa; }}
.table-scroll:focus-visible {{ outline:2px solid #205f4b; outline-offset:3px; }}
table {{ width:100%; border-collapse:collapse; font-size:.9rem; font-variant-numeric:tabular-nums; }}
table.results {{ min-width:1000px; }}
th,td {{ padding:12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
thead {{ background:var(--wash); }}
tbody tr:last-child > * {{ border-bottom:0; }}
.note {{ color:var(--muted); font-size:.88rem; }}
details {{ margin:7px 0; }}
summary {{ cursor:pointer; width:fit-content; padding:3px 0; }}
details .table-scroll {{ max-width:800px; }}
@media(max-width:680px) {{ main {{ padding:16px 14px 30px; }} .scope {{ grid-template-columns:1fr; }} th,td {{ padding:8px; }} }}
</style>
</head>
<body>
<main>
<nav aria-label="Source and data"><a href="{original}">Original all-token ablation</a> <a href="results.json">Raw replay results</a> <a href="probabilities.csv" download>Download probabilities</a></nav>
<header>
  <h1>T2375 off on the question only</h1>
  <p class="lede">All seven answers and decoding logits matched native. T2375 was already zero on the {selected_count} underlined question tokens.</p>
  <dl class="scope">
    <div><dt>Set T2375 to zero only here</dt><dd><u>{question_html}</u></dd></div>
    <div><dt>Chat prefix kept at its native values</dt><dd><pre>{prefix_html}</pre></dd></div>
  </dl>
  <p class="note">The intervention targeted L29–T2375. The list, template, answer prefix and generated positions followed the native forward pass. The prefix retained {prefix_active} positive gates across the seven cases.</p>
</header>
<section aria-labelledby="comparison-title">
  <h2 id="comparison-title">Native vs. question-only gate-off</h2>
  <p>P11 and P12 are full-word probabilities for the exact lowercase targets. Question edits count positive contributions removed out of the 21 eligible tokens in each case.</p>
  <div class="table-scroll" tabindex="0" role="region" aria-label="Seven list ablation comparisons">
  <table class="results"><thead><tr><th scope="col">Case</th><th scope="col">11th / 12th target</th><th scope="col">Native response</th><th scope="col">Gate-off response</th><th scope="col">P11 native → off (%)</th><th scope="col">P12 native → off (%)</th><th scope="col">Question edits</th><th scope="col">Max prefix gate</th></tr></thead>
  <tbody>{rows_html}</tbody></table>
  </div>
  <p class="note">† List 1 and List 5 in bullet format failed the separate native twelfth-word check. All five numbered-list controls passed.</p>
</section>
<section aria-labelledby="tokens-title">
  <h2 id="tokens-title">Top five next tokens by case</h2>
  <p>The ranked tokens and percentages matched exactly for every question-only replay. <code>␠</code> marks a space.</p>
  {details_html}
</section>
</main>
</body>
</html>'''
(result_dir / "index.html").write_text(page + "\n", encoding="utf-8")
print(f"Rendered {len(table_rows)} cases with {selected_count} eligible tokens and {edited_count} edits.")
