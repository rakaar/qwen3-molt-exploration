"""Build an overview and a complete response explorer for the fixed-gate sweep."""
from pathlib import Path
from html import escape
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot/prefix-boost'
s = json.loads((OUT / 'analysis.json').read_text())
r = json.loads((OUT / 'results.json').read_text())
assert len(s['rows']) == 100 and len(r['cases']) == 20
groups = ''.join(f"<tr><td>{g['target_gate']:g}</td><td>{g['new_first_listed_eleventh']}/18</td><td>{g['new_strict_eleventh']}/18</td><td>{g['new_answer_eleventh_from_native_correct']}/13</td><td>{g['responses_changed']}/20</td><td>{g['literal_eleven']}/20</td></tr>" for g in s['groups'])
hits = [x for x in s['rows'] if x['new_first_listed_eleventh']]
assert len(hits) == 1
hit = hits[0]
hit_case = next(c for c in r['cases'] if c['id'] == hit['id'])
finding = f'''<div class="findings"><p><strong>One new item-11 answer at gate 40, inside a sentence.</strong> Gates 2, 5, 10 and 20 produced no new item-11 answers. None of the 13 native-correct requested-word cases shifted to item 11 at any strength.</p></div>
<p><strong>Observed case:</strong> bullet list 3, requesting item 12. Correct target: <code>{escape(hit['expected'])}</code>; item 11: <code>{escape(hit['eleventh_word'])}</code>. Native: <code>{escape(hit['native_response'])}</code>. At gate 40:</p><blockquote>{escape(hit['response'])}</blockquote>
<p class="muted">This is an item-11 answer with extra text, so it is counted in the answer-content column and not in word-only counts. Both native controls failed here: the twelfth-word request answered bell (item 14), and the saved eleventh-word request answered {escape(hit_case['native_eleventh_question_control']['response'])} (item 10). The result is retained as an exploratory case.</p>'''
summary_table = f'''<div class="scroll"><table id="boost-summary"><thead><tr><th>Fixed gate</th><th>New item-11 answers<br>(including sentences)</th><th>New word-only<br>item-11 answers</th><th>New item-11 answers from<br>correct native requests</th><th>Response texts<br>changed</th><th>Literal “eleven”<br>or 11</th></tr></thead><tbody>{groups}</tbody></table></div>'''
methods = '''<p>Both input-prefix newline tokens receive the same fixed scalar: after <code>&lt;|im_end|&gt;</code> and after <code>&lt;/think&gt;</code>. The local recipient operator is retained. The native MLP output receives <code>(target − natural gate) × operator</code>; this is absolute gate replacement, including when the natural gate was zero.</p><p>List-reading tokens and all generated tokens are unedited. Each of the 100 variants edits exactly two input positions. All 20 native and zero-strength runs exactly match the saved baselines. Independent CPU gate/operator reconstruction and exact BF16 output-rounding checks passed. No run hit the 32-token generation limit.</p>'''
section = f'''<section id="prefix-boost"><p class="eyebrow">Causal follow-up · scalar sufficiency</p><h2>Increase T2375 at both chat-prefix newlines</h2>{finding}<p>All 20 revised fifth/twelfth-word prompts were tested at fixed gates 2, 5, 10, 20 and 40: <strong>100 edited runs</strong>. Two native twelfth-word cases already answered with item 11, so only 18 cases can newly switch. Thirteen native requested-word answers were correct; both requested/eleventh controls were correct for nine cases.</p>{summary_table}<p class="muted">“New item-11 answer” means the first listed word in the completed response is item 11 and it was not already the native answer. The single new case contains only that listed word. Word-only answers apply the additional exact-response format requirement. At gate 40, one different case said the literal word “eleven”; that is not a listed-word retrieval and is excluded from item-11 counts.</p>{methods}<p><a href="prefix-boost/">Inspect all 100 responses and probabilities →</a> · <a href="prefix-boost/probabilities.csv">CSV</a> · <a href="prefix-boost/analysis.json">Verified summary</a></p></section>'''
(OUT / 'section.html').write_text(section + '\n')

cases = []
for c in r['cases']:
    cases.append({k: c[k] for k in ['id','list_id','format','ordinal','expected','eleventh_word','native_requested_correct','native_already_eleventh','native_eleventh_question_control','prefix_positions']})
    cases[-1].update(prompt=next(x['prompt'] for x in r['protocol']['cases'] if x['id']==c['id']), native_response=c['native']['native']['response'])
data = dict(groups=s['groups'], rows=s['rows'], cases=cases, strengths=s['strengths'])
(OUT / 'explorer-data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',',':')) + '\n')
version = hashlib.sha256((OUT / 'explorer-data.json').read_bytes() + (HERE / 'web/explorer.js').read_bytes()).hexdigest()[:12]
html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="only light"><meta name="darkreader-lock"><title>T2375 prefix-gate sweep: all 100 responses</title><link rel="stylesheet" href="../comparison.css?v={version}"><link rel="stylesheet" href="explorer.css?v={version}"><script defer src="explorer.js?v={version}"></script></head><body><main>
<nav><a href="../#prefix-boost">← List-retrieval study</a><span>L29–T2375 · 11 November 2026</span></nav>
<header><p class="eyebrow">Fixed scalar at two input positions · all cases retained</p><h1>Increase the gate.<br>Inspect every answer.</h1>{finding}</header>
<section><h2>All strengths</h2>{summary_table}<p class="muted">Denominators: 20 requests; two already answered item 11 natively; 18 eligible for a new item-11 answer; 13 native-correct requested-word cases. No truncations. “Eleven” as a number word is not the eleventh listed item.</p></section>
<section id="responses"><h2>Native versus increased gate</h2><div class="controls"><label>Fixed scalar <select id="boost-gate" aria-label="Fixed scalar"></select></label><label>Requested word <select id="boost-request" aria-label="Requested word"><option value="all">Both</option><option value="5">5th</option><option value="12">12th</option></select></label><label>List format <select id="boost-format" aria-label="List format"><option value="all">Both</option><option value="numbered">Numbered</option><option value="bullets">Bullets</option></select></label></div><p id="response-count" class="muted"></p>
<p class="muted">Probabilities are <strong>percentages, native → steered</strong>, for the exact lowercase word directly at response start, across all its tokens. These probabilities exclude leading sentences, whitespace variants, punctuation and EOS; a sentence can answer with item 11 while its direct-word probability is low. Full generated-sequence probabilities and word-plus-EOS scores are in the CSV.</p><div class="scroll"><table id="boost-responses"><thead><tr><th>Case / native controls</th><th>Asked / item-11 target</th><th>Native response</th><th>Steered response</th><th>P(direct item-11 word)<br>native → steered (%)</th><th>P(direct requested word)<br>native → steered (%)</th><th>Inspect</th></tr></thead><tbody></tbody></table></div></section>
<details class="methods"><summary>Scope, validation and data</summary>{methods}<p>For the sentence-form ring response, the exact generated token-sequence probability (including its selected EOS) was {100*hit['generated_sequence_probability']:.4f}%. Direct “ring” at response start was {100*hit['steered_p_word_11']:.6f}%; it does not include the sentence form.</p><p>Native requested-word correctness and saved native eleventh-word control correctness are shown separately. None of the nine cases passing both controls newly answered item 11. Natural recipient gates at the selected positions ranged from 0 to about 0.896; targets were 2–40. These are exploratory high-gain edits, not a proof of an array-indexing mechanism.</p><p><a href="analysis.json">Verified summary</a> · <a href="probabilities.csv">All 100 response/probability rows</a> · <a href="results.json">Full captures</a> · <a href="protocol.json">Frozen protocol</a> · <a href="source/run.py">Inference source</a> · <a href="source/analyze.py">Independent checks</a></p></details>
</main><dialog id="boost-modal"><div class="modal-head"><h2 id="boost-modal-title"></h2><button id="boost-close">Close</button></div><div id="boost-modal-body"></div></dialog></body></html>'''
(OUT / 'index.html').write_text(html)
for name in ['explorer.js','explorer.css']:
    (OUT / name).write_text((HERE / 'web' / name).read_text().replace('__VERSION__', version))
(OUT / 'source').mkdir(exist_ok=True)
for name in ['run.py','analyze.py','render.py']:
    shutil.copy2(HERE / name, OUT / 'source' / name)
print('Built five-strength overview and complete 100-response explorer.')
