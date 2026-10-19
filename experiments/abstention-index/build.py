"""Build the abstention experiment directory without model inference."""
import html
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
OUT = ROOT / 'results/abstention'
REPO = 'https://github.com/rakaar/qwen3-molt-exploration/blob/main/'
rows = json.loads((SOURCE / 'catalog.json').read_text())
esc = html.escape
OUT.mkdir(parents=True, exist_ok=True)
body = []
for row in rows:
    if 'report' in row:
        assert (ROOT / 'results' / row['report']).is_file(), row['report']
        url = '../' + row['report']
        link = 'Report'
    else:
        assert (ROOT / row['github']).is_file(), row['github']
        url = REPO + row['github']
        link = 'Shortlist'
    links = f'<a href="{esc(url)}">{link} ↗</a>'
    if row.get('code'):
        assert (ROOT / row['code']).exists(), row['code']
        source_url = REPO.replace('/blob/', '/tree/') if row['code'].endswith('/') else REPO
        links += f'<a href="{esc(source_url + row["code"])}">Source ↗</a>'
    body.append(f'<tr><td class="date">{row["date"][5:]}<small>{esc(row["stage"])}</small></td>'
                f'<th scope="row">{esc(row["title"])}</th><td>{esc(row["description"])}</td>'
                f'<td class="links">{links}</td></tr>')
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="An index of Qwen3-4B MOLT abstention, confidence and uncertainty experiments, with descriptions, reports, code and negative results.">
<title>Abstention experiments · Qwen3-4B MOLT</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f8f3;color:#1b363c;font:16px/1.6 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:32px 30px 70px}a{color:#17665b;text-underline-offset:4px}a:focus-visible,input:focus-visible{outline:3px solid #c88b21;outline-offset:4px}.top{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;font-size:14px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:750;color:#4c686c;margin-top:48px}h1{font-size:clamp(34px,5vw,60px);line-height:1.1;letter-spacing:-.035em;margin:12px 0 22px}.lead{font-size:19px;max-width:880px}.start{background:#eaf3ee;border:1px solid #ccddd3;padding:20px 24px;margin:28px 0}.start p{margin:7px 0}.note{font-size:14px;color:#476066;max-width:1000px}.tools{margin:32px 0 15px;display:flex;gap:18px;align-items:end;flex-wrap:wrap}.tools label{flex:1;min-width:240px;font-size:14px;font-weight:650}input{display:block;width:100%;margin-top:7px;padding:11px;border:1px solid #aebfba;border-radius:5px;font:inherit;background:white;color:inherit}.count{font-size:14px;color:#4c686c}.table-wrap{overflow-x:auto;border:1px solid #d6dfd9;border-radius:6px;background:white}table{width:100%;border-collapse:collapse;font-size:14px}caption{text-align:left;padding:17px 20px;font-size:16px;font-weight:700}thead{background:#eaf0eb}th,td{text-align:left;vertical-align:top;padding:18px 16px;border-top:1px solid #e0e6e1}thead th{border:0;font-size:12px;letter-spacing:.04em;text-transform:uppercase}tbody th{width:25%;font-size:15px;font-weight:700}td.date{width:15%;font-variant-numeric:tabular-nums;font-weight:650}small{display:block;color:#617377;font-weight:400;font-size:12px;margin-top:6px}.links{width:95px}.links a{display:block;margin-bottom:9px;white-space:nowrap}tr[hidden]{display:none}footer{margin-top:28px;font-size:13px;color:#5d7477}code{font-size:.94em} @media(max-width:650px){main{padding:24px 16px 48px}.eyebrow{margin-top:30px}.lead{font-size:17px}th,td{padding:13px 11px}table{min-width:780px}.start{padding:16px}.tools{margin-top:22px}}
</style></head><body><main>
<nav class="top"><a href="../../index.html">← All MOLT results</a><a href="https://github.com/rakaar/qwen3-molt-exploration">GitHub repository ↗</a></nav>
<p class="eyebrow">Research directory · Updated 19 October 2026</p><h1>Abstention experiments.</h1>
<p class="lead">What makes Qwen3-4B say “I don’t know”? One place for the steering experiments, native gate heatmaps, confidence searches and negative results.</p>
<aside class="start"><b>Start with the completed comparison.</b><p><a href="../2026-10-19-abstention-complete/index.html">20 questions × input-only, output-only and input + output steering →</a></p><p>For the original narrative and posters: <a href="../2026-10-15-abstention-overview/index.html">the first ten made-up-answer cases</a>.</p></aside>
<p class="note"><b>Reading the results:</b> “Made up an answer” means the passage does not support the answer; it can still be true external knowledge. Clear abstention means a coherent “I don’t know / it’s not in the passage” without giving the substantive answer. Inducing this on an answer-present question is false abstention. The cohorts overlap across reports and must not be counted as independent replications.</p>
<p class="note">Most later L19–T368 experiments add <code>a × low_rank_output(current_input)</code> to the native MLP output; the early fixed-gate experiment instead replaces the estimated gate. Strengths were found with sweeps and follow-ups, not a guaranteed binary search or continuous minimum. The confidence and uncertainty studies below are observational; no useful pre-answer confidence gate or confidence–strength correlation has been established.</p>
<div class="tools"><label for="search">Find an experiment<input id="search" type="search" placeholder="e.g. input, vampire, confidence, fixed gate"></label><span class="count" aria-live="polite">COUNT entries</span></div>
<div class="table-wrap" tabindex="0" aria-label="Experiment table; scroll horizontally on small screens"><table><caption>Descriptions and links</caption><thead><tr><th>Date (2026) / area</th><th>Experiment</th><th>Description</th><th>Links</th></tr></thead><tbody>ROWS</tbody></table></div>
<footer>All layer, token and transform IDs are zero-based. Reports preserve their original scope and dates; later combined analyses may use more complete grids. No new model inference was run to build this directory. <a href="catalog.json">Download the index data</a> · <a href="https://github.com/rakaar/qwen3-molt-exploration/blob/main/experiments/abstention-index/README.md">Index source</a>.</footer>
</main><script>const input=document.querySelector('#search'),rows=[...document.querySelectorAll('tbody tr')],count=document.querySelector('.count');input.addEventListener('input',()=>{const q=input.value.trim().toLowerCase();rows.forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));count.textContent=rows.filter(r=>!r.hidden).length+' / '+rows.length+' entries'});</script></body></html>'''
page = page.replace('COUNT', str(len(rows))).replace('ROWS', ''.join(body))
(OUT / 'index.html').write_text(page)
(OUT / 'catalog.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
print(f'Built {OUT / "index.html"}: {len(rows)} entries')
