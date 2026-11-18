"""Render the saved first-token top five for each native and T2375-off run."""
import csv
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "results/2026-11-11-list-retrieval-pilot/all-token-off"
SITE = SOURCE / "top-five"
ORIGINAL = "https://rakaar.github.io/qwen3-molt-exploration/results/2026-11-11-list-retrieval-pilot/?v=0663557#comparison"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_top_five(run, hashes):
    saved = run["native"]["steps"][0]
    entry = run["arrays"]
    path = SOURCE / entry["file"]
    assert digest(path.read_bytes()) == hashes[path.name] == entry["sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        logits = arrays["first_logits"]
    assert digest(logits.tobytes()) == saved["logits_sha256"]
    top = saved["top5"]
    ids = [row["token_id"] for row in top]
    assert len(ids) == len(set(ids)) == 5 and ids[0] == saved["token_id"]
    ranked_logits = logits[ids]
    assert np.all(ranked_logits[:-1] >= ranked_logits[1:])
    # Exact ties retain the order recorded in the original run.
    assert ranked_logits[-1] == np.partition(logits, -5)[-5]
    weights = np.exp(logits.astype(np.float64) - logits.max())
    np.testing.assert_allclose(weights[ids] / weights.sum(), [row["probability"] for row in top], rtol=2e-6)
    return top


def percentage(probability):
    percent = probability * 100
    return f"{percent:.6f}%" if percent >= 1 else f"{percent:.5g}%"


def cell(row):
    token = row["token"].replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")
    return (f'<td><div class="token-entry"><code title="Token ID {row["token_id"]}">{escape(token)}</code>'
            f'<span class="probability">{percentage(row["probability"])}</span></div></td>')


hashes = {row["path"]: row["sha256"] for row in json.loads((SOURCE / "transfer-manifest.json").read_text())["files"]}
data = (SOURCE / "results.json").read_bytes()
assert digest(data) == hashes["results.json"]
saved = json.loads(data)
cases = saved["cases"]
assert saved["status"] == "complete" and len(cases) == 7
prompts = {case["id"]: case["prompt"] for case in saved["protocol"]["cases"]}
cards = []
csv_rows = []
top_one_same = 0
top_five_membership_changed = 0
twelfth_top_five = 0
for case in cases:
    case_id = case["id"]
    tops = {mode: read_top_five(case[mode], hashes) for mode in ("native", "off")}
    ids = {mode: [row["token_id"] for row in top] for mode, top in tops.items()}
    top_one_same += ids["native"][0] == ids["off"][0]
    top_five_membership_changed += set(ids["native"]) != set(ids["off"])
    for mode, top in tops.items():
        twelfth_top_five += case[mode]["candidates"]["twelfth"]["token_ids"][0] in ids[mode]
        for rank, row in enumerate(top, 1):
            csv_rows.append([case_id, mode, rank, row["token_id"], row["token"], row["probability"]])
    list_number = str(case["list_id"]).removeprefix("list-")
    label = f'List {list_number} · {case["format"]}'
    targets = [escape(case[key]) for key in ("eleventh_word", "twelfth_word")]
    responses = [escape(case[mode]["native"]["response"]) for mode in ("native", "off")]
    table = "\n".join(f'<tr><th scope="row">{rank}</th>{cell(left)}{cell(right)}</tr>'
                      for rank, (left, right) in enumerate(zip(tops["native"], tops["off"]), 1))
    cards.append(f'''<section class="case" id="{case_id}" aria-labelledby="title-{case_id}">
  <h2 id="title-{case_id}">{label}</h2>
  <p class="targets">11th <code>{targets[0]}</code> <span aria-hidden="true">·</span> 12th <code>{targets[1]}</code></p>
  <table aria-label="Top five next tokens for {label}">
    <thead><tr><th scope="col">Rank</th><th scope="col">Native</th><th scope="col">T2375 off</th></tr></thead>
    <tbody>{table}</tbody>
  </table>
  <p class="response">Full answer <code>{responses[0]}</code> → <code>{responses[1]}</code></p>
  <details><summary>View the list and prompt</summary><pre>{escape(prompts[case_id])}</pre></details>
</section>''')

assert len(csv_rows) == 70
SITE.mkdir(exist_ok=True)
with (SITE / "top-five.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["case", "condition", "rank", "token_id", "token", "probability"])
    writer.writerows(csv_rows)

case_markup = "\n".join(cards)
html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Native and T2375-off first-token rankings for the seven eleventh-word list cases.">
  <title>List retrieval · Native and T2375-off top five</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
<main>
  <nav aria-label="Source links"><a href="{escape(ORIGINAL)}">Original list comparison</a><a href="top-five.csv" download>Download token probabilities</a></nav>
  <header>
    <h1>Top five next tokens</h1>
    <p class="intro">Seven eleventh-word list prompts, with native Qwen3-4B beside the run with L29–T2375 off across the input and generation.</p>
    <dl class="overview">
      <div><dt>Same top token</dt><dd>{top_one_same}/{len(cases)}</dd></div>
      <div><dt>Different top-five membership</dt><dd>{top_five_membership_changed}/{len(cases)}</dd></div>
      <div><dt>12th word’s first token in top five</dt><dd>{twelfth_top_five}/{2 * len(cases)}</dd></div>
    </dl>
    <p class="note">Probabilities are for the first token after the prompt. A token can be a word fragment, such as <code>p</code> in <code>pencil</code>. <code>␠</code> marks a space. The 12th-word check uses the exact lowercase target. Equal probabilities keep the saved ranking order.</p>
  </header>
  <div class="cases">{case_markup}</div>
  <footer>
    <p>The native model answered all seven eleventh-word prompts correctly. The two bullet-format cases failed their separate native twelfth-word checks.</p>
    <p>The 14 rankings were checked against the saved logits. <a href="../results.json">Raw results JSON</a></p>
  </footer>
</main>
</body>
</html>'''
(SITE / "index.html").write_text(html + "\n", encoding="utf-8")
print(f"Verified {len(csv_rows)} entries from {len(cases)} pairs of saved logit vectors.")
