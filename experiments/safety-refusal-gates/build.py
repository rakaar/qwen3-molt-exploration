"""Create the HTML report and scientific token-box heatmap from saved measurements."""
import csv
import hashlib
import html
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/2026-10-15-safety-refusal-gates'
CMAP = LinearSegmentedColormap.from_list('gate', ['#f5faf9', '#b9e6dd', '#48b2a3', '#11786e', '#073e3b'])
ESC = html.escape


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visible(row):
    return [t for t in row['answer_tokens'] if not t['special']]


def glyph(token):
    return token.replace(' ', '·').replace('\n', '↵').replace('\t', '⇥') or '∅'


def main():
    data = json.loads((OUT/'measurements.json').read_text())
    reviews = json.loads((OUT/'review.json').read_text())
    assert data['status'] == 'complete' and len(data['rows']) == 3
    by_id = {r['id']: r for r in reviews['rows']}
    for row in data['rows']:
        assert row['all_replay_predictions_match_native'] and row['native_generation_no_hooks']
        assert row['answer_sha256'] == by_id[row['id']]['answer_sha256']
        assert by_id[row['id']]['safe_to_display_full_response'], 'Review before displaying response'
        row['review'] = by_id[row['id']]
        tokens = visible(row)
        assert len(tokens) == row['summary']['visible_tokens']
        assert math.isclose(sum(t['gate'] for t in tokens)/len(tokens), row['summary']['mean_gate'], abs_tol=1e-12)
        assert sum(t['gate'] > 0 for t in tokens) == row['summary']['active_tokens']

    prior_path = ROOT/'results/2026-10-14-passage-answer-gates/artifacts/answer-gates.json'
    prior = json.loads(prior_path.read_text())
    for key in ['base_revision', 'molt_revision', 'asset_sha256', 'layer', 'transform']:
        assert prior['protocol'][key] == data['protocol'][key]
    references = [r for r in prior['rows'] if r['id'] in ['P1-U3', 'P1-U4', 'P2-U4']]
    assert len(references) == 3
    (OUT/'abstention-reference.json').write_text(json.dumps({
        'source': str(prior_path.relative_to(ROOT)), 'source_sha256': sha(prior_path),
        'selection': 'The three clean natural-abstention examples already displayed in the earlier overview.',
        'rows': references}, ensure_ascii=False, indent=2)+'\n')
    vmax = max(t['gate'] for r in data['rows']+references for t in visible(r))
    norm = Normalize(vmin=0, vmax=vmax if vmax else 1)
    gate_counts = sum(r['summary']['active_tokens'] > 0 for r in data['rows'])
    refusal_count = sum(r['review']['safety_refusal'] for r in data['rows'])
    insight = (f'{refusal_count}/3 responses refuse the harmful request; '
               f'L19-T368 activates on {gate_counts}/3 responses.')

    def stats(row):
        s = row['summary']
        return (f"Mean {s['mean_gate']:.4f} · max {s['max_gate']:.4f} · "
                f"active {s['active_tokens']}/{s['visible_tokens']} tokens")

    def token_boxes(row):
        parts = []
        for t in visible(row):
            value = 0.0 if t['gate'] == 0 else t['gate']
            color = to_hex(CMAP(norm(value)))
            ink = '#ffffff' if norm(value) > .60 else '#123b37'
            desc = f"Answer token {t['answer_position']} | token ID {t['token_id']} | gate {value:.8f} | raw {t['raw']:.8f}"
            parts.append(f'<span class="token" tabindex="0" style="--bg:{color};--ink:{ink}" title="{ESC(desc,quote=True)}" data-info="{ESC(desc,quote=True)}"><span class="piece">{ESC(glyph(t["token"]))}</span><span class="value">{value:.3f}</span></span>')
        return ''.join(parts)

    def card(row, previous=False):
        question = row['question'] if previous else row['prompt']
        label = 'Earlier passage-grounded abstention' if previous else row['category']
        grade = '' if previous else f'<span class="badge">{ESC(row["review"]["classification"])}</span>'
        notes = '' if previous else f'<p class="assessment">{ESC(row["review"]["notes"])}</p>'
        return f'''<article class="card"><div class="cardhead"><span class="case">{row['id']} · {ESC(label)}</span>{grade}</div>
<h2>{ESC(question)}</h2><p class="stats">{stats(row)}</p>
<p class="answer">{ESC(row['answer'])}</p><div class="tokens" aria-label="Token activation heatmap for {row['id']}">{token_boxes(row)}</div>{notes}</article>'''

    def table_row(row, group):
        s = row['summary']
        first = s.get('first_active_answer_position')
        if first is None:
            first = next((t['answer_position'] for t in visible(row) if t['gate'] > 0), '—')
        return f'<tr><td>{row["id"]}</td><td>{group}</td><td>{s["mean_gate"]:.4f}</td><td>{s["max_gate"]:.4f}</td><td>{s["active_tokens"]}/{s["visible_tokens"]}</td><td>{first}</td></tr>'

    gradient = ','.join(to_hex(CMAP(x/10)) for x in range(11))
    rows = ''.join(table_row(r, 'Safety refusal' if r['review']['safety_refusal'] else 'Other response') for r in data['rows'])
    rows += ''.join(table_row(r, 'Earlier abstention') for r in references)
    threshold = data['threshold']
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L19-T368 · Safety refusal token gates</title><style>
:root{color-scheme:light;--ink:#153b37;--muted:#566e69;--line:#d9e5e1;--accent:#11786e}*{box-sizing:border-box}body{margin:0;background:#f2f6f4;color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:44px 26px 64px}.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;font-weight:750;color:var(--accent)}h1{font-size:clamp(28px,4.5vw,48px);max-width:950px;line-height:1.14;margin:14px 0 18px;letter-spacing:-.035em}.intro{max-width:960px;color:var(--muted);font-size:18px}.finding{background:#e1f0ea;border-left:4px solid var(--accent);padding:15px 19px;font-size:19px;font-weight:650;margin:24px 0}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:26px 0 15px}.legend{width:340px;max-width:100%}.gradient{height:12px;border-radius:9px;background:linear-gradient(90deg,GRADIENT);border:1px solid #c2d6cd}.ticks{display:flex;justify-content:space-between;font:12px ui-monospace,monospace}.caption{font-size:13px;color:var(--muted)}.card{background:white;border:1px solid var(--line);padding:24px 27px;margin:20px 0;border-radius:15px;box-shadow:0 4px 18px #153b3705}.cardhead{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.case{font-size:13px;text-transform:uppercase;letter-spacing:.06em;font-weight:750;color:var(--accent)}.badge{font-size:12px;font-weight:650;border:1px solid #bbd9ce;padding:2px 10px;border-radius:20px;background:#edf7f1}h2{font-size:22px;line-height:1.35;margin:11px 0}.stats{font:13px/1.5 ui-monospace,monospace;color:var(--muted);margin:9px 0 18px}.answer{white-space:pre-wrap;border-left:3px solid #d9e9e2;padding-left:16px;max-width:1000px;margin:17px 0 22px}.tokens{display:flex;flex-wrap:wrap;gap:5px;align-items:flex-start}.token{display:inline-flex;flex-direction:column;align-items:center;min-width:28px;min-height:46px;padding:4px 7px 3px;background:var(--bg);color:var(--ink);border:1px solid #153b371d;border-radius:5px;cursor:help}.token:hover,.token:focus{outline:2px solid #db8e2c;outline-offset:1px}.piece{font:14px/1.4 ui-monospace,"DejaVu Sans Mono",monospace;white-space:pre}.value{font:9px/1.3 ui-monospace,monospace;opacity:.88}.hidevalues .value{display:none}.hidevalues .token{min-height:30px}.assessment{font-size:13px;color:var(--muted);margin-bottom:0}#inspector{position:sticky;bottom:8px;background:#143d37;color:white;padding:9px 14px;font:12px/1.5 ui-monospace,monospace;border-radius:7px;box-shadow:0 5px 20px #153b3725;z-index:2;min-height:36px;margin-top:20px}.section{margin-top:36px}h3{font-size:21px;margin-bottom:10px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{text-align:left;padding:10px 13px;border-bottom:1px solid var(--line)}th{background:#e7efeb;font-size:12px}.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}details{margin-top:25px}summary{cursor:pointer;font-weight:650}.methods{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin-top:25px}.methods p{max-width:1040px}code{font-size:12px;word-break:break-all}.downloads{display:flex;flex-wrap:wrap;gap:15px;margin:20px 0}a{color:#116f65;text-underline-offset:3px}footer{font-size:12px;color:var(--muted);margin-top:25px}@media(max-width:600px){main{padding:25px 14px 45px}.card{padding:18px 16px}.piece{font-size:12px}.token{padding:4px 5px;min-width:24px}.value{font-size:8px}h2{font-size:19px}.intro{font-size:16px}.finding{font-size:17px}}
</style><main><nav class="caption" aria-label="Related reports"><a href="../../index.html">Research notebook</a> · <a href="../2026-10-15-abstention-overview/index.html">Earlier abstention experiment</a></nav><div class="eyebrow">Qwen3-4B · MOLT layer 19 · transform 368 · rank 128</div>
<h1>Does this transform also activate during safety refusals?</h1>
<p class="intro">Three fixed harmful requests, native greedy responses, and the natural L19-T368 gate on every visible answer token. No steering. The earlier experiment concerned missing passage evidence; these prompts concern safety.</p>
<div class="finding">INSIGHT</div><div class="toolbar"><div class="legend"><div class="gradient"></div><div class="ticks"><span>0 · inactive</span><span>VMAX</span></div><div class="caption">One linear color scale for all six examples, including the earlier references.</div></div><label><input id="showvalues" type="checkbox" checked> Show gate values</label></div>
<p class="caption">Each box is one tokenizer token. · marks a space; ↵ marks a newline. Hover or focus a box for its index, token ID, raw score and gate.</p>
CARDS
<div id="inspector" aria-live="polite">Hover or focus a token to inspect its exact measurement.</div>
<section class="section"><h3>Comparison with the earlier ignorance examples</h3><p class="caption">The three earlier examples are the previously displayed clean natural abstentions, reused from saved measurements with the same model and MOLT asset. Different prompts and response lengths make this a descriptive comparison.</p><div class="tablewrap"><table><thead><tr><th>Case</th><th>Context</th><th>Mean gate</th><th>Max gate</th><th>Active / visible</th><th>First active index</th></tr></thead><tbody>TABLE</tbody></table></div><details><summary>Show the three earlier abstention heatmaps</summary>REFERENCES</details></section>
<section class="methods"><h3>What was measured</h3><p>The native model generated each answer without hooks. An observation-only replay then measured the residual entering layer 19’s post-attention layer normalization. Every next-token argmax was checked against the native generation. The native MLP and attention were unchanged.</p><p>After released input standardization, <code>raw = z dot e_368 + bias_368</code>. The gate equals raw only when <code>raw &gt; THRESHOLD</code> and <code>raw &gt; 0</code>; otherwise it is zero. Model computation is BF16 and gate math is FP32.</p><p>A box records activation while <strong>processing the displayed token</strong>. It does not show the activation that predicted that token. Means include inactive tokens and exclude EOS/special tokens; EOS is preserved in the raw data. Generation uses one user message, no system message, thinking disabled, greedy decoding, seed 0 and a 192-token cap.</p><p><strong>Interpretation:</strong> activity in these examples can show that this gate is shared across refusal contexts. It does not show that the transform causes the refusal, that all safety refusals use it, or that a single mechanism explains the different contexts. No literature-novelty claim is made from this pilot. Response review was by Codex with prompts visible.</p><details><summary>Pinned identity and provenance</summary><p>Base: <code>BASE</code><br>MOLT: <code>MOLT</code><br>Selected asset SHA256: <code>ASSET</code><br>Native model files and selected weights were hash-verified before loading. Full prompts, chat strings, token IDs, raw scores, gates and runtime metadata are included below.</p></details></section>
<div class="downloads"><a href="heatmap.png">PNG heatmap</a><a href="heatmap.svg">Vector heatmap</a><a href="measurements.json">Raw measurements</a><a href="tokens.csv">Token CSV</a><a href="protocol.json">Protocol</a><a href="review.json">Response review</a><a href="https://github.com/rakaar/qwen3-molt-exploration/tree/main/experiments/safety-refusal-gates">Experiment source</a></div><footer>Experiment: 15 October 2026 · Published: 16 October 2026 · Exploratory activation evidence. No new inference was used to render this page.</footer>
</main><script>const toggle=document.getElementById('showvalues');toggle.addEventListener('change',()=>document.body.classList.toggle('hidevalues',!toggle.checked));for(const t of document.querySelectorAll('.token')){for(const event of ['mouseenter','focus'])t.addEventListener(event,()=>document.getElementById('inspector').textContent=t.dataset.info)}</script></html>'''
    substitutions = {'GRADIENT': gradient, 'INSIGHT': ESC(insight), 'VMAX': f'{vmax:.4f}',
        'CARDS': ''.join(card(r) for r in data['rows']), 'TABLE': rows,
        'REFERENCES': ''.join(card(r, True) for r in references), 'THRESHOLD': f'{threshold:.8f}',
        '<code>BASE</code>': '<code>'+data['protocol']['base_revision']+'</code>',
        '<code>MOLT</code>': '<code>'+data['protocol']['molt_revision']+'</code>',
        '<code>ASSET</code>': '<code>'+data['protocol']['asset_sha256']+'</code>'}
    for key, value in substitutions.items():
        page = page.replace(key, value)
    (OUT/'index.html').write_text(page)

    with (OUT/'tokens.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'category', 'answer_position', 'token_id', 'token', 'gate', 'raw', 'threshold'])
        for row in data['rows']:
            for t in visible(row):
                writer.writerow([row['id'], row['category'], t['answer_position'], t['token_id'], t['token'], t['gate'], t['raw'], threshold])

    # Scientific export: all three full native responses, with one rectangle per token.
    font = FontProperties(family='DejaVu Sans Mono', size=10)
    width, left, right, line_height = 16, .55, 15.45, .43
    layout, y = [], 2.00
    for row in data['rows']:
        title = f"{row['id']}  {row['prompt']}"
        title_lines = textwrap.wrap(title, 125)
        top = y
        y += .28*len(title_lines)+.39
        x = left
        positioned = []
        for t in visible(row):
            label = glyph(t['token'])
            tw = TextPath((0, 0), label, prop=font).get_extents().width/72
            tw = max(.27, tw+.15)
            if x+tw > right:
                x = left
                y += line_height
            positioned.append((x, y, tw, label, 0.0 if t['gate'] == 0 else t['gate']))
            x += tw+.035
        y += .71
        layout.append((row, top, title_lines, positioned))
    height = y+.65
    fig = plt.figure(figsize=(width, height), dpi=160, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0,width), ylim=(height,0))
    ax.axis('off')
    ax.text(left, .46, 'Safety refusals · natural L19-T368 activation', fontsize=21, fontweight='bold', color='#153b37', va='center')
    ax.text(left, .85, insight, fontsize=12, color='#315e55', va='center')
    ax.text(left, 1.17, f'No steering · shared linear scale 0–{vmax:.4f} · token boxes show gates while processing the displayed token', fontsize=10, color='#526f67', va='center')
    for i in range(160):
        ax.add_patch(FancyBboxPatch((left+i*6.5/160,1.41),6.5/160+.001,.10,
                     boxstyle='square,pad=0',facecolor=CMAP(i/159),edgecolor='none'))
    ax.text(left,1.72,'0 · inactive',fontsize=8.5,color='#526f67')
    ax.text(left+6.5,1.72,f'{vmax:.4f}',fontsize=8.5,color='#526f67',ha='right')
    for row, top, title_lines, positions in layout:
        for i, line in enumerate(title_lines):
            ax.text(left, top+i*.28, line, fontsize=12.5, fontweight='bold', color='#153b37', va='top')
        ax.text(left, top+.28*len(title_lines)+.07, stats(row), fontsize=10, color='#557269', va='top')
        for x, yy, tw, label, gate in positions:
            color = CMAP(norm(gate)); ink = 'white' if norm(gate) > .60 else '#123b37'
            ax.add_patch(FancyBboxPatch((x,yy),tw,.36,boxstyle='round,pad=0.007,rounding_size=0.035',facecolor=color,edgecolor='#c4d8d0',linewidth=.5))
            ax.text(x+tw/2,yy+.12,label,fontproperties=font,color=ink,ha='center',va='center')
            ax.text(x+tw/2,yy+.285,f'{gate:.2f}',fontsize=6.7,color=ink,ha='center',va='center',fontfamily='DejaVu Sans Mono')
    ax.text(left,height-.40,'· = space; ↵ = newline. Means include zeros; EOS/special tokens excluded. Three fixed prompts: descriptive activation evidence, not a causal test.',fontsize=9,color='#526f67')
    ax.text(left,height-.18,'Qwen3-4B · BF16 native model / FP32 MOLT math · greedy, thinking off · 15 October 2026',fontsize=8.5,color='#526f67')
    fig.savefig(OUT/'heatmap.png', dpi=160)
    fig.savefig(OUT/'heatmap.svg')
    plt.close(fig)
    verification = {'status':'passed', 'cases':3, 'safety_refusals':refusal_count,
        'responses_with_gate_activity':gate_counts, 'shared_color_max':vmax,
        'all_native_replay_predictions_match':True, 'all_response_review_hashes_match':True,
        'all_summaries_recomputed':True, 'earlier_reference_source_sha256':sha(prior_path),
        'output_sha256':{name:sha(OUT/name) for name in ['measurements.json','index.html','heatmap.png','heatmap.svg','tokens.csv','review.json']}}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({**verification,'output_sha256':'saved in verification.json'},indent=2))


if __name__ == '__main__':
    main()
