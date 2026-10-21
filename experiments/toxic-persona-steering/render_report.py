"""Render local tables; keep raw generations private unless publication is authorized."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, nargs='*', default=[])
parser.add_argument('--review', type=Path)
parser.add_argument('--findings', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
rows, runs = {}, []
for path in args.results:
    run = json.loads(path.read_text())
    runs.append({'file': str(path), 'status': run['status'], 'runtime': run['runtime']})
    for row in run['rows']:
        if row['id'] in rows:
            assert row['answer_sha256'] == rows[row['id']]['answer_sha256'], 'Repeated baselines differ'
        rows[row['id']] = row
reviews = json.loads(args.review.read_text()) if args.review else {}
findings = json.loads(args.findings.read_text()) if args.findings else {}
strengths = sorted(set(P['strengths']) | {row['strength'] for row in rows.values() if row['kind'] == 'additive'})
groups = [('malicious_intent', 'L11–T2234: malicious intent'),
          ('moral_transgression', 'L16–T1519: moral transgression'), ('both', 'Both transforms')]
escape = html.escape


def find(case, group, strength):
    kind = 'native' if strength == 0 else 'additive'
    return next((row for row in rows.values() if row['case_id'] == case and row['kind'] == kind
                 and (strength == 0 or row['group'] == group and row['strength'] == strength)), None)


def cell(row, full=False):
    if row is None:
        return '<td class="pending">Not run</td>'
    grade = reviews.get(row['id'], {})
    coherence = grade.get('coherence', 'awaiting review')
    behavior = grade.get('behavior', '')
    css = {'coherent': 'coherent', 'partly coherent': 'mixed', 'incoherent': 'incoherent'}.get(coherence, 'pending')
    truncation = '<span class="badge">token limit</span>' if row['stop_reason'] == 'token_limit' else ''
    note = escape(grade.get('note', ''))
    badge = f'<span class="behavior {escape(behavior)}">{escape(behavior)}</span>' if behavior else ''
    tag = f'<strong>{escape(coherence)}</strong>{badge}{truncation}'
    if not full:
        return f'<td class="{css}">{tag}<p>{note}</p></td>'
    answer = escape(row['answer'])
    preview = escape(row['answer'].replace('\n', ' ')[:110])
    return f'<td class="{css}">{tag}<details><summary>{preview}…</summary><pre>{answer}</pre></details><small>{len(row["output_token_ids"])} tokens · {escape(row["stop_reason"])}</small></td>'


parts = []
for case in P['cases']:
    case_strengths = sorted(set(P['strengths']) | {row['strength'] for row in rows.values()
                            if row['case_id'] == case['id'] and row['kind'] == 'additive'})
    section = f'<section><h2>{escape(case["id"].replace("_", " ").title())}</h2><blockquote>{escape(case["user"])}</blockquote>'
    for full in [False, True]:
        section += f'<h3>{"Full responses" if full else "Coherence and behavior"}</h3><div class="table-scroll"><table><thead><tr><th scope="col">Steered transforms</th>'
        section += ''.join(f'<th scope="col">{s:g}</th>' for s in [0, *case_strengths]) + '</tr></thead><tbody>'
        for group, label in groups:
            section += f'<tr><th scope="row">{escape(label)}</th>'
            section += ''.join(cell(find(case['id'], group, strength), full) for strength in [0, *case_strengths]) + '</tr>'
        section += '</tbody></table></div>'
    section += '</section>'
    parts.append(section)
complete = bool(runs) and all(run['status'] == 'complete' for run in runs)
status = 'Generations complete' if complete else 'Prepared; results pending' if not runs else 'Run in progress'
evaluated = sum(row['id'] in reviews for row in rows.values() if row['kind'] != 'observe')
body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Two MOLT candidates · steering pilot</title><style>
:root{color-scheme:light;font:16px/1.5 system-ui,sans-serif;color:#202a36;background:#f4f6f9}body{margin:0}main{max-width:1540px;margin:auto;padding:32px 24px 70px}h1{font-size:2.2rem;line-height:1.15;margin-bottom:10px}h2{font-size:1.6rem;margin-bottom:12px}h3{font-size:1.08rem;margin:24px 0 10px}a{color:#175a9b}header p{max-width:1000px}.status{display:inline-block;padding:6px 12px;background:#e3edf8;border:1px solid #bfd1e5;border-radius:5px;font-weight:650}section{margin:36px 0;padding-top:12px;border-top:1px solid #cfd8e4}blockquote{margin:14px 0 22px;padding:16px 20px;background:white;border-left:4px solid #4c729b;font-size:1.1rem}.table-scroll{overflow:auto;border:1px solid #cdd7e2;background:white;border-radius:7px}table{border-collapse:collapse;table-layout:fixed;min-width:1280px;width:100%}th,td{padding:13px 12px;vertical-align:top;text-align:left;border-right:1px solid #dce2e9;border-bottom:1px solid #dce2e9;font-size:.88rem}th{background:#eaf0f7}thead th{font-size:1rem}th:first-child{width:170px}td p{margin:7px 0 0;font-size:.82rem}td strong{font-size:.85rem}.coherent{background:#f0f8f3}.mixed{background:#fff8e7}.incoherent{background:#fcefee}.pending{color:#6a7380;background:#f6f7f9}.badge{display:block;font-size:.72rem;margin:5px 0;color:#765215}details summary{cursor:pointer;margin-top:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;font-size:.84rem;line-height:1.48}small{display:block;color:#687483;margin-top:10px}.method{background:white;padding:18px 22px;border:1px solid #d4deea;border-radius:7px}.method p{max-width:1100px}footer{color:#627082;font-size:.85rem;margin-top:30px}button{padding:7px 12px;cursor:pointer;border:1px solid #becbdd;background:white;border-radius:5px}
</style><main>'''
body += f'<header><p>PRIVATE EXPLORATORY RESULT · 21 OCTOBER 2026</p><h1>Can two MOLT candidates change the model’s persona?</h1><p class="status">{status}</p><p>Columns show additive gate strength. Rows show each transform individually and both together. Zero is the same native baseline repeated for comparison. Each question has a summary table and a full-response table.</p><p>{len([r for r in rows.values() if r["kind"] == "additive"])} steered responses · {evaluated} manually reviewed responses · one greedy generation per condition</p><button onclick="document.querySelectorAll(\'td details\').forEach(x=>x.open=true)">Expand all responses</button></header>'
body += '<style>.behavior{display:table;margin-top:5px;padding:2px 6px;border-radius:4px;font-size:.76rem;background:#e6eaf0;color:#354353}.behavior.harmful{background:#f7d9d7;color:#8c231c}.behavior.ambiguous{background:#f7e7ba;color:#75520c}.behavior.refusal{background:#dce9fa;color:#22578c}.findings{margin-top:26px;padding:18px 22px;background:white;border:1px solid #cbd9e9;border-left:4px solid #4c729b;border-radius:5px}.findings p{max-width:1120px}.findings h2{margin:0 0 10px}</style>'
if findings:
    body += '<aside class="findings"><h2>What this pilot shows</h2>'
    body += ''.join(f'<p>{escape(text)}</p>' for text in findings.get('summary_paragraphs', []))
    body += '</aside>'
body += ''.join(parts)
body += '<section class="method"><h2>How to read this pilot</h2><p>A strength of 2 adds 2 to the estimated gate coefficient; it does not double the natural activation. In the joint condition, each transform receives the displayed strength. The two rows therefore need not have equal perturbation norms.</p><p>The native Qwen3-4B MLP and attention remain in place. We add strength × the transform’s current input-dependent output, from the first user token through generation. The model uses greedy decoding, thinking disabled, and a 512-token response cap. Token-limit truncation is marked separately from coherence.</p><p>Assistant review separates coherence from harmful content, refusal, topic changes and fictional framing. The review packet hides feature identities and strengths. These are selected questions and candidate labels, so this pilot cannot establish a general toxicity rate or a distinct causal persona mechanism. Large edits can cause generic damage; this initial sweep has no matched active-transform controls.</p>'
body += f'<p>Base model: <code>{P["base_repo"]}@{P["base_revision"]}</code><br>MOLT weights: <code>{P["molt_repo"]}@{P["molt_revision"]}</code></p><p><a href="https://huggingface.co/datasets/georglange/qwen3-4b-molt-interpretations">Labels and scores</a> · <a href="{P["source_questions"]}">Exact evaluation-question source</a> · <a href="https://arxiv.org/html/2506.19823v2#S3.SS2">OpenAI’s persona-feature analysis</a></p></section>'
body += f'<footer>Rendered {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %H:%M IST")} · Local report; raw generations have not been published.</footer></main></html>'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(body)
print(json.dumps({'report': str(args.output), 'status': status, 'rows': len(rows), 'reviewed': evaluated}))
