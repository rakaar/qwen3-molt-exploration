"""Build a standalone local research page from verified token-confound results."""
import hashlib
import html
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DIST = HERE / 'dist'
DIST.mkdir(exist_ok=True)
OUT = ROOT / 'results/2026-11-08-eleven-token-confound'
SOURCE = ROOT / 'comparisons/2026-11-08-eleven-token-confound'
initial_path = SOURCE / 'remote-export/artifacts/results.json'
followup_path = SOURCE / 'followup-clear-slots/remote-export/artifacts/results.json'
initial = json.loads(initial_path.read_text())
followup = json.loads(followup_path.read_text())
summary = json.loads((OUT / 'clear-slots/summary.json').read_text())
initial_summary = json.loads((OUT / 'initial/summary.json').read_text())
reviews = json.loads((ROOT / 'experiments/eleven-token-confound-20261108/native-answer-review.json').read_text())
assert initial['status'] == followup['status'] == 'complete'
assert len(initial['cases']) == 27 and len(followup['cases']) == 9
primary = [next(c for c in followup['cases'] if c['ordinal'] == 11 and c['domain'] == d)
           for d in ['month', 'alphabet', 'roman']]
matrix = next(m for m in summary['final_token_matrices'] if m['ordinal'] == 11)
original = next(m for m in initial_summary['final_token_matrices'] if m['ordinal'] == 11 and m['variant'] == 'original')
assert len({c['final_input']['token_id'] for c in primary}) == 3
assert all(reviews[c['id']]['correct'] is True for c in primary)
esc = html.escape
NAMES = {'month': 'Month', 'alphabet': 'Alphabet', 'roman': 'Roman numeral'}


def token_html(t, final, prefix):
    strength = min(t['gate'] / 2.5, 1)
    background = f'rgba(222, 125, 18, {0.12 + strength * 0.66:.4f})' if t['gate'] else '#f0f1ed'
    text = t['token'].replace(' ', '·').replace('\n', '↵')
    label = f"Position {t['position']}; token {t['token']!r}; token ID {t['token_id']}; gate {t['gate']:.6f}" + ('; compared vector' if final else '')
    return f'''<button class="token {'active' if t['gate'] else 'zero'} {'compared' if final else ''}" style="--token-bg:{background}" title="{esc(label)}" aria-label="{esc(label)}" data-inspector="{prefix}" data-token="{esc(t['token'])}" data-position="{t['position']}" data-id="{t['token_id']}" data-gate="{t['gate']:.6f}" data-raw="{t['raw']:.6f}" data-norm="{t['contribution_norm']:.6f}"><span>{esc(text)}</span><small>{t['position']} · {t['gate']:.3f}</small></button>'''


def card(case, number=None, compact=False):
    prefix = 'inspect-' + case['id'] + ('-all' if compact else '')
    final = case['final_input']
    status = reviews[case['id']]['correct']
    status_label = 'Correct initial answer' if status is True else ('Ambiguous initial answer' if status is None else 'Initial answer control failed')
    tokens = ''.join(token_html(t, t['position'] == final['position'], prefix) for t in case['trace'])
    active = ', '.join(f"{t['token']!r} at {t['position']} (φ {t['gate']:.3f})" for t in case['active_input_tokens']) or 'None — every input gate is zero.'
    serial = f'<span class="case-number">{number:02d}</span>' if number else ''
    return f'''<article class="case {'compact' if compact else ''}" data-case="{case['id']}"><div class="case-heading">{serial}<h3>{NAMES[case['domain']]}</h3><span class="status {'pass' if status is True else 'failed'}">{status_label}</span></div><div class="sentence-response"><div><p class="label">Exact input sentence</p><p class="sentence">{esc(case['prompt'])}</p></div><div class="response"><p class="label">Native response · first {len(case['native']['output_token_ids'])} generated tokens</p><pre>{esc(case['native']['response'])}</pre></div></div><div class="tokens-heading"><span class="label">Every input token · position and φ</span><span>Blue outline = matrix position</span></div><div class="tokens">{tokens}</div><p class="active-list"><b>Active input tokens:</b> {esc(active)}</p><div class="inspector" id="{prefix}" aria-live="polite">Compared token <code>{esc(repr(final['token']))}</code> · position <b>{final['position']}</b> · ID <b>{final['token_id']}</b> · gate <b>{final['gate']:.6f}</b> · contribution L2 <b>{final['contribution_norm']:.6f}</b></div></article>'''


def matrix_table(item, kind='contribution', interactive=False):
    classes = 'cosine-table' + (' primary-matrix' if interactive else '')
    body = '<table class="'+classes+'"><caption class="sr-only">Pairwise cosine similarities between '+kind+' vectors</caption><thead><tr><th scope="col">Compared at<br>final input token</th>'
    for domain, token in zip(item['domains'], item['tokens']):
        body += f'<th scope="col">{NAMES[domain]}<small>{esc(repr(token["token"]))}</small></th>'
    body += '</tr></thead><tbody>'
    for i, domain in enumerate(item['domains']):
        body += f'<tr><th scope="row">{NAMES[domain]}<small>{esc(repr(item["tokens"][i]["token"]))}</small></th>'
        for j in range(3):
            value = item['cosines'][kind][i][j]
            label = 'Undefined' if value is None else f'{value:.5f}'
            alpha = 0 if value is None else max(0.08, min(0.72, (value-.5)*1.2))
            body += f'<td data-row="{i}" data-col="{j}" style="--heat:{alpha:.4f}">{label}</td>'
        body += '</tr>'
    return body + '</tbody></table>'


def control_rows():
    body = ''
    for c in followup['cases']:
        state = reviews[c['id']]['correct']
        label = 'Pass' if state is True else ('Ambiguous' if state is None else 'Failed')
        body += f'<tr><th scope="row">{NAMES[c["domain"]]}</th><td>{c["ordinal"]}</td><td><code>{esc(repr(c["final_input"]["token"]))}</code></td><td>{c["final_input"]["gate"]:.6f}</td><td>{label}</td><td>{esc(c["native"]["response"])}</td></tr>'
    return '<div class="table-scroll"><table class="controls-table"><thead><tr><th>Domain</th><th>Ordinal</th><th>Final token</th><th>Gate φ</th><th>Native initial answer</th><th>Native response · 32-token excerpt</th></tr></thead><tbody>' + body + '</tbody></table></div>'


primary_cards = ''.join(card(c, i+1) for i,c in enumerate(primary))
all_cases = ''.join(f'<details class="case-detail"><summary>{esc(c["id"])} <span>{esc(c["prompt"])}</span></summary>{card(c, compact=True)}</details>' for c in initial['cases'] + followup['cases'])
norm_rows = ''.join(f'<tr><th scope="row">{NAMES[c["domain"]]}</th><td>{c["final_input"]["gate"]:.6f}</td><td>{c["final_input"]["operator_norm"]:.6f}</td><td>{c["final_input"]["contribution_norm"]:.6f}</td></tr>' for c in primary)
css = (HERE / 'style.css').read_text()
js = (HERE / 'app.js').read_text()
matrix_json = json.dumps(matrix, ensure_ascii=False, allow_nan=False).replace('</', r'<\/')
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Different tokens, aligned contributions · T2375</title><meta name="description" content="Three rewritten eleventh prompts, exact native responses, every input-token gate and a 3×3 MOLT contribution cosine matrix."><style>{css}</style></head><body><header><a class="brand" href="#top">QWEN / MOLT NOTES</a><nav aria-label="Sections"><a href="#sentences">Sentences & gates</a><a href="#cosines">3 × 3</a><a href="#controls">Controls</a></nav></header><main id="top"><section class="intro"><p class="eyebrow">8 November 2026 · Qwen3-4B · L29–T2375</p><h1>Different tokens.<br><em>Aligned contributions.</em></h1><p class="lede">Does the high similarity survive when the last input token changes? These three prompts end with <code>called</code>, <code>letter</code> and <code>as</code>. Each gives the correct initial eleventh answer.</p><div class="finding"><strong>0.983–0.988</strong><span>Pairwise cosine between full MOLT contributions<br>at the three different final input tokens.</span></div></section><section id="sentences" class="section"><div class="section-title"><div><p class="eyebrow">01 / Sentences → responses → input gates</p><h2>What the model saw and answered</h2></div><div class="legend"><span><i class="swatch zero"></i>φ = 0</span><span><i class="gradient"></i>0 → 2.5</span><span><i class="swatch outline"></i>Compared token</span></div></div><p class="section-note">Orange shows natural gate strength on input tokens. Click or hover a token for its exact value. Responses are shown verbatim; their tokens have no measured gate coloring. No gate is turned off here.</p>{primary_cards}</section><section id="cosines" class="section"><div class="section-title"><div><p class="eyebrow">02 / One fixed position per prompt</p><h2>The 3 × 3 cosine comparison</h2></div><label class="matrix-select">Vector to compare<select id="vector-kind"><option value="contribution">Full MOLT contribution · φ × UVx</option><option value="operator">Ungated operator · UVx</option><option value="residual">Native residual stream</option></select></label></div><p id="matrix-description" class="section-note">Each cell compares the full natural MOLT contributions at the blue-outlined final input tokens above. No final RMSNorm or unembedding is applied.</p><div class="matrix-shell">{matrix_table(matrix, interactive=True)}</div><p id="pair-readout" class="pair-readout" aria-live="polite">Month “called” ↔ Alphabet “letter”: <b>0.98828</b>. These are different token IDs.</p><details class="extra"><summary>Different gate strengths and vector magnitudes</summary><p>The gate is φ(x); the operator is UVx; the full MOLT contribution is φ(x) × UVx. The checkpoint’s input/output standardizers are retained. Positive scalar gates change length, while leaving operator/contribution cosine unchanged.</p><div class="table-scroll"><table><thead><tr><th>Prompt</th><th>Gate φ</th><th>Operator L2</th><th>Contribution L2</th></tr></thead><tbody>{norm_rows}</tbody></table></div></details><details class="extra"><summary>Compare with the original three “is” prompts</summary><p>Original eleventh-month, alphabet and Roman-numeral prefixes, all measured at final <code> is</code>.</p><div class="table-scroll">{matrix_table(original)}</div></details><aside class="interpretation"><b>What this addresses</b><p>The identical final “is” token is not required for the high contribution alignment. The corresponding residual-stream cosines here are only 0.604–0.677. Shared task context and this operator’s geometry remain possible explanations.</p><p>Some tenth/twelfth controls also activate. Activity and aligned direction are not exclusive to eleven.</p></aside></section><section id="controls" class="section"><p class="eyebrow">03 / Full scope retained</p><h2>Controls and other rewrites</h2><p class="section-note">The first phase froze 27 prompts. After some phrasings produced unclear answer slots, this nine-run follow-up used <code>called / letter / as</code>. Six prompts were new; three alphabet prompts were exact repeats. The follow-up was chosen for answer-slot clarity, before computing its cosines.</p><details class="extra"><summary>Matched 10th / 11th / 12th controls</summary><p>“Pass” means the initial target answer was correct; it does not certify the whole continuation. The tenth-month “called” case starts with a clarification, later gives October and then contradicts itself, so it is marked ambiguous.</p>{control_rows()}</details><details class="extra"><summary>Every sentence, native response and input gate trace · 36 runs / 33 unique prompts</summary><p>This includes unsuccessful “named,” “symbol” and “representation” rewrites, as well as inactive gates. Initial-answer failures remain visible.</p>{all_cases}</details></section><section class="method section"><p class="eyebrow">Method & source data</p><h2>Natural activations, native model</h2><p>All examples are raw completion prefixes. The final input position predicts the first generated token. The observer reads the L29 residual after attention and before post-attention layer normalization; it computes the released MOLT’s standardized gate and operator, and leaves native computation unchanged.</p><p>The three original anchors reproduce exactly. All 36 runs preserve native generated IDs and measured next-token logits with the observer attached. Independent CPU vector audits and both backup hash checks passed.</p><div class="downloads"><a href="data/followup-results.json" download>Follow-up prompts, responses & all token values</a><a href="data/initial-results.json" download>Initial 27-prompt pilot</a><a href="data/clear-slot-summary.json" download>Exact cosine matrices</a><a href="data/all-input-token-traces.csv" download>Follow-up token traces · CSV</a><a href="data/verification.json" download>Verification</a></div><p class="small">The follow-up is adaptive and selected. These are initial-answer observations and vector comparisons; this page adds no causal intervention.</p></section></main><footer>Local research page · 8 November 2026 · Layer, transform and token positions are zero-based.</footer><script type="application/json" id="matrix-data">{matrix_json}</script><script>{js}</script></body></html>'''
residual_section = f'''<section id="residuals" class="section residual-section"><p class="eyebrow">03 / Native residual stream</p><h2>Residual-stream cosine similarities</h2><p class="section-note">Compare the native residual vectors at the same final input positions: <code>called</code>, <code>letter</code> and <code>as</code>. These are the inputs read at L29 after attention and before post-attention layer normalization, before the MOLT’s input standardization. No final RMSNorm or unembedding is applied.</p><figure class="residual-figure"><a href="figures/residual-cosines.png" aria-label="Open the full-size residual-stream similarity plot"><img src="figures/residual-cosines.png" width="2070" height="990" alt="Two 3 by 3 native residual-stream cosine matrices: original prompts all ending in is on the left, and rewritten prompts ending in called, letter and as on the right. Rewritten off-diagonal cosines are 0.676747, 0.604279 and 0.640050."></a><figcaption>Left: the original three “is” prompts. Right: the rewritten prompts above. Both use a fixed 0–1 color scale; these are residual vectors, not MOLT contributions. Click the plot to open it full-size.</figcaption></figure><div class="residual-result"><b>Different final tokens:</b> residual-stream cosines <strong>0.604–0.677</strong>, alongside full MOLT contribution cosines <strong>0.983–0.988</strong>.</div><details class="extra"><summary>Exact 3 × 3 residual values for the rewritten prompts</summary><div class="table-scroll">{matrix_table(matrix, kind='residual')}</div></details></section>'''
residual_section = residual_section.replace('<figure class="residual-figure">',
    '<div class="matrix-shell">' + matrix_table(matrix, kind='residual') + '</div>'
    '<details class="extra"><summary>Original “is” versus rewritten tokens · comparison plot</summary>'
    '<figure class="residual-figure">', 1)
residual_section = residual_section.replace('</figcaption></figure>', '</figcaption></figure></details>', 1)
residual_section = residual_section.replace('<details class="extra"><summary>Exact 3 × 3 residual values for the rewritten prompts</summary><div class="table-scroll">' + matrix_table(matrix, kind='residual') + '</div></details>', '', 1)
assert page.count('<section id="controls"') == 1
page = page.replace('<section id="controls"', residual_section + '<section id="controls"', 1)
page = page.replace('<a href="#controls">Controls</a>', '<a href="#residuals">Residuals</a><a href="#controls">Controls</a>', 1)
page = page.replace('03 / Full scope retained', '04 / Full scope retained', 1)
figures = DIST / 'figures'
figures.mkdir(exist_ok=True)
shutil.copy2(OUT / 'residual-cosines.png', figures / 'residual-cosines.png')
(DIST / 'index.html').write_text(page)
data = DIST / 'data'
data.mkdir(exist_ok=True)
for source, name in [(initial_path, 'initial-results.json'), (followup_path, 'followup-results.json'),
                     (OUT/'clear-slots/summary.json', 'clear-slot-summary.json'),
                     (OUT/'clear-slots/all-input-token-traces.csv', 'all-input-token-traces.csv'),
                     (OUT/'verification.json', 'verification.json')]:
    shutil.copy2(source, data/name)
manifest = {'source_results_sha256': {'initial': hashlib.sha256(initial_path.read_bytes()).hexdigest(), 'followup': hashlib.sha256(followup_path.read_bytes()).hexdigest()},
            'primary_prompts': [c['id'] for c in primary], 'primary_final_token_ids': [c['final_input']['token_id'] for c in primary],
            'all_runs': 36, 'unique_prompts': 33, 'local_only': True}
(DIST / 'data/page-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'index': str(DIST/'index.html'), **manifest}, indent=2))
