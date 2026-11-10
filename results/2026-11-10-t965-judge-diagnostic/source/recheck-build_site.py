"""Publish both highlight scopes explicitly after the paired recheck is verified."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/2026-11-10-t965-highlight-recheck'
SITE = ROOT / 'results/2026-11-10-t965-judge-diagnostic'

report = json.loads((OUT / 'analysis.json').read_text())
assert report['queries'] == 12 and report['unchanged_example_text'] and report['unchanged_descriptions']
asset_version = report['audit_sha256'][:12]
rows = [json.loads(s) for s in (OUT / 'audit.jsonl').read_text().splitlines()]
data = json.loads((SITE / 'study-data.json').read_text())
plan = json.loads((HERE / 'plan.json').read_text())
assert all(data[k] == plan[k] for k in ['baseline_explanation', 'candidate_explanation'])
assert [c['original_fuzz_row'] for c in data['cases']] == [13, 17, 20]
queries = {}
for r in rows:
    q = {k: r[k] for k in ['id', 'condition', 'description_arm', 'original_fuzz_row', 'messages',
                           'response', 'prediction', 'request_sha256', 'finish_reason', 'cached_original_response']}
    q['excerpt'] = r['messages'][-1]['content'].split('Example 0: ', 1)[1].rsplit('\n\nClassify only', 1)[0]
    queries[q['id']] = q
data['highlight_recheck'] = dict(analysis=report, queries=queries, date='10 November 2026',
                                source_audit_sha256=hashlib.sha256((OUT / 'audit.jsonl').read_bytes()).hexdigest())
(SITE / 'study-data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
shutil.copy2(OUT / 'audit.jsonl', SITE / 'highlight-recheck-audit.jsonl')
shutil.copy2(OUT / 'analysis.json', SITE / 'highlight-recheck-analysis.json')
shutil.copy2(HERE / 'plan.json', SITE / 'highlight-recheck-plan.json')
for name in ['run.py', 'analyze.py', 'build_site.py']:
    shutil.copy2(HERE / name, SITE / 'source' / ('recheck-' + name))
html = (SITE / 'index.html').read_text()
html = re.sub(r'<title>.*?</title>', '<title>Same text, different highlights · T965 judge comparison</title>', html, count=1)
html = re.sub(r'<meta name="description" content="[^"]*">', '<meta name="description" content="A fresh paired recheck of old and new feature descriptions, with all original highlights versus the final activation token only.">', html, count=1)
html = html.replace('href="style.css"', f'href="style.css?v={asset_version}"').replace('src="app.js"', f'src="app.js?v={asset_version}"')
header = '''<header><p class="eyebrow">Extended input · same descriptions · two highlight scopes</p>
<h1>Same text.<br>Different highlights.</h1>
<p>The earlier “0, 1, 1” result marked <strong>only the final activation token</strong>. The “0, 0, 0” result kept <strong>all original highlights</strong>. Both include the visible K continuation and the same new description.</p>
<div class="table-scroll"><table><thead><tr><th>Fresh recheck · new description + extended input</th><th>Development</th><th>Tonkatsu</th><th>.wait → key</th></tr></thead><tbody id="scope-summary"></tbody></table></div>
<p class="key"><mark>Highlighted text</mark> = the tokens the judge must classify. A judge answer of <strong>0</strong> does not mean zero gate activation.</p>
<div class="scope-controls" role="group" aria-label="Highlighting condition"><button data-scope="final_only" aria-pressed="true">Only final activation token</button><button data-scope="all_original" aria-pressed="false">All original highlights</button></div>
<p id="scope-banner" class="scope-banner" aria-live="polite"></p>
<button id="rules-button" class="text-button">Read both exact descriptions ↗</button></header>'''
header += '<p class="key">Historical answers reproduced in this fresh run: <strong>' + str(sum(c['matched'] for c in report['controls'])) + '/9</strong>.</p>'
if not report['all_historical_controls_match']:
    header += '<p class="recheck-warning">Some historical answers did not reproduce. The table and cards show the fresh run; the introductory numbers describe the earlier run. See the raw comparison for each difference.</p>'
html = re.sub(r'<header>.*?</header>', header, html, count=1, flags=re.S)
html = re.sub(r'<details class="optional"><summary>What did the later diagnostics show\?</summary>.*?</details>',
              '<p class="key"><a href="diagnostics.html">Earlier diagnostic study: seven conditions and 22 exact queries ↗</a></p>',
              html, count=1, flags=re.S)
html = html.replace('The old description was not tested in this final-highlight-only follow-up.', 'The earlier follow-up had not tested the old description with final-only highlights; the new 12-query recheck does.')
html = html.replace('The unchanged explanation is', 'The unchanged explanation is')
html = html.replace('<details class="optional"><summary>Method, source data and limits</summary>', '<details class="optional"><summary>Method, source data and limits</summary><p id="recheck-verification"></p><p><a href="highlight-recheck-audit.jsonl">Raw 12-query recheck</a> · <a href="highlight-recheck-analysis.json">Recheck verification</a> · <a href="highlight-recheck-plan.json">Frozen recheck prompts</a></p>')
html = html.replace('These are three selected failures.', 'These were three selected failures in the earlier all-highlight comparison.')
html = html.replace('The judge checks whether', 'The results above are the fresh paired recheck, with both highlight scopes and both descriptions. The original seven-window score is historical; the other four examples were not rerun here. The judge checks whether')
(SITE / 'index.html').write_text(html)
css = (SITE / 'style.css').read_text()
css += '\n.scope-controls{display:flex;gap:9px;flex-wrap:wrap;margin:20px 0 10px}.scope-controls [aria-pressed=true]{background:var(--accent);color:white;border-color:var(--accent)}.scope-banner{font-size:14px;border-left:3px solid var(--accent);padding:10px 15px;background:#e9eee2}.active-scope{background:#e9eee2}.score.unrun{font-size:15px}\n'
(SITE / 'style.css').write_text(css)
script = r'''"use strict";
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let study,scope='final_only';const modal=document.getElementById('prompt-modal');
function openModal(title,html){document.getElementById('modal-title').textContent=title;document.getElementById('modal-content').innerHTML=html;modal.showModal();}
document.getElementById('close-modal').addEventListener('click',()=>modal.close());modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
function query(c,arm){return study.highlight_recheck.queries[`row-${c.original_fuzz_row}-${scope}-${arm}`];}
function crop(text,row){const i=row===13?text.indexOf('1. Install Java'):row===17?text.indexOf('4.'):0;return i>0?'…\n'+text.slice(i):text;}
function marked(text,c){let html='',last=0,i=0;const matches=[...text.matchAll(/<<(.*?)>>/gs)];for(const m of matches){html+=esc(text.slice(last,m.index));const next=text.slice(m.index+m[0].length).replaceAll('<<','').replaceAll('>>','').trimStart()[0];const token=c.highlighted_tokens[matches.length===1?c.highlighted_tokens.length-1:i++];html+=`<mark tabindex="0" class="${next==='K'||next==='k'?'':'other'}" title="Recorded original gate: ${token.measured_gate}; next non-whitespace character: ${esc(next)}">${esc(m[1])}</mark>`;last=m.index+m[0].length;}return html+esc(text.slice(last));}
function prompt(c,arm){const q=query(c,arm);openModal(`${c.title} · ${scope==='final_only'?'final highlight only':'all original highlights'} · ${arm==='baseline'?'old':'new'} description`,q.messages.map(m=>`<h3>${esc(m.role)}</h3><pre>${esc(m.content)}</pre>`).join('')+`<h3>Judge answer</h3><pre>${esc(q.response)}</pre>`);}
function render(){document.querySelectorAll('[data-scope]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scope===scope)));document.getElementById('scope-banner').textContent=scope==='final_only'?'Selected: only the final activation token is highlighted. Earlier tokens remain in the text, but are not marked.':'Selected: all original highlighted tokens are retained. Fuzz requires every marked token to fit the description.';document.querySelectorAll('#scope-summary tr').forEach(tr=>tr.classList.toggle('active-scope',tr.dataset.view===scope));document.getElementById('cases').innerHTML=study.cases.map((c,i)=>{const old=query(c,'baseline'),fresh=query(c,'candidate');const notes=scope==='final_only'?{13:'Only the final Development token is marked. The next subject-model token is " Kit", with the space included; its recorded gate is positive.',17:'Only the final ton fragment is marked. Earlier Ton occurrences remain in the passage, unhighlighted.',20:'Only .wait is marked before key(). The comma and plus are still in the text, but no longer part of the classification target.'}:{13:'Both highlighted Development tokens precede Kit.',17:'All three original Ton / ton highlights precede katsu.',20:'The comma and plus precede h; only .wait precedes k. A before-K explanation does not cover every original highlight.'};return `<article class="case" data-case="${c.original_fuzz_row}"><div class="case-head"><h2>${esc(c.title)}</h2><span class="case-number">${String(i+1).padStart(2,'0')} / 03</span></div><pre class="excerpt">${marked(crop(fresh.excerpt,c.original_fuzz_row),c)}</pre><div class="answers"><div class="answer"><span class="score">${Number(old.prediction)}</span><p>Old description<small>${scope==='final_only'?'Final highlight only':'All original highlights'}</small></p></div><div class="answer"><span class="score">${Number(fresh.prediction)}</span><p>New description<small>Same input and highlight scope</small></p></div></div><p class="note">${notes[c.original_fuzz_row]}</p><div class="card-actions"><button data-row="${c.original_fuzz_row}" data-arm="baseline">Full old-description prompt ↗</button><button data-row="${c.original_fuzz_row}" data-arm="candidate">Full new-description prompt ↗</button></div></article>`;}).join('');document.querySelectorAll('[data-arm]').forEach(b=>b.addEventListener('click',()=>prompt(study.cases.find(c=>c.original_fuzz_row===Number(b.dataset.row)),b.dataset.arm)));}
fetch('study-data.json').then(r=>{if(!r.ok)throw Error('Study data unavailable');return r.json();}).then(d=>{study=d;const report=d.highlight_recheck.analysis;document.getElementById('scope-summary').innerHTML=['final_only','all_original'].map(s=>`<tr data-view="${s}"><td>${s==='final_only'?'Only final activation token':'All original highlights'}</td>${d.cases.map(c=>`<td>${Number(d.highlight_recheck.queries[`row-${c.original_fuzz_row}-${s}-candidate`].prediction)}</td>`).join('')}</tr>`).join('');document.querySelectorAll('[data-scope]').forEach(b=>b.addEventListener('click',()=>{scope=b.dataset.scope;render();}));document.getElementById('rules-button').addEventListener('click',()=>openModal('Exact feature descriptions',`<h3>Old</h3><p>${esc(d.baseline_explanation)}</p><h3>New · unchanged</h3><p>${esc(d.candidate_explanation)}</p>`));document.getElementById('recheck-verification').textContent=`Fresh paired recheck: 12 queries, same pinned judge and runtime, unchanged descriptions, identical example text and few-shots. ${report.controls.filter(c=>c.matched).length}/9 historical answers reproduced. Old-description final-only results are new comparisons.`;render();}).catch(e=>{document.getElementById('cases').textContent='Could not load the paired comparisons. Please reload.';console.error(e);});
'''
script = script.replace("fetch('study-data.json')", f"fetch('study-data.json?v={asset_version}')")
(SITE / 'app.js').write_text(script)
files = [dict(path=str(p.relative_to(SITE)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in sorted(SITE.rglob('*')) if p.is_file() and p.name != 'site-manifest.json']
(SITE / 'site-manifest.json').write_text(json.dumps(dict(files=files), indent=2) + '\n')
print('Built explicit all-highlight/final-only views from the verified 12-query recheck.')
