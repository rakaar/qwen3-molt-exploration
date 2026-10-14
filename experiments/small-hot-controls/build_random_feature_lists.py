"""Render the actual recorded random-control identities; no model execution."""
import csv
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parents[1] / 'results/2026-10-14-random-control-features'
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / 'artifacts/results.json').read_text())
L = json.loads((ROOT / 'labels.json').read_text())
LABELS = {f['id']: f for f in L['features']}
IDS = R['selection']['uniform_ids']
PAIRS = R['selection']['uniform_pairs']
assert len(IDS) == len(set(IDS)) == len(PAIRS) == 100
assert 941 not in IDS and 2339 not in IDS
assert all(a != b for a, b in PAIRS)
assert set(a for a, b in PAIRS) == set(b for a, b in PAIRS) == set(IDS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(group):
    rr = sorted((r for r in R['rows'] if r['group'] == group), key=lambda r: r['sample_index'])
    assert [r['sample_index'] for r in rr] == list(range(100))
    return rr


def feature(f):
    info = LABELS[f]
    return {'layer': 22, 'id': f, 'label': info['label'] or 'Unlabelled in saved export',
            'status': info['status'], 'rank': info['rank']}


def caption(f):
    info = feature(f)
    mark = ' †' if info['status'] == 'limited_evidence' else ''
    return f'L22–M{f} — “{info["label"]}”{mark}'


single_rows = []
pair_rows = []
single_views = [('uniform_single_positive', 'donor', 4.5), ('uniform_single_negative', 'receiver', -4.5)]
pair_views = [('uniform_pair', -4.5, 4.5), ('uniform_pair_reverse', 4.5, -4.5)]
for group, source, coefficient in single_views:
    for i, row in enumerate(rows(group)):
        f = IDS[i]
        assert row['terms'] == [[source, f, coefficient]]
        assert not row['nonzero_edit'] and all(g['gate'] == 0 for g in row['source_gates'])
        single_rows.append({'trial_index': i + 1, 'condition_id': row['id'], 'group': group,
            **feature(f), 'source': source, 'coefficient': coefficient,
            'source_token': 'hot' if source == 'donor' else 'small',
            'source_gate': row['source_gates'][0]['gate'], 'P_cold': row['P_cold']})
for group, a_coefficient, b_coefficient in pair_views:
    for i, row in enumerate(rows(group)):
        a, b = PAIRS[i]
        assert row['terms'] == [['receiver', a, a_coefficient], ['donor', b, b_coefficient]]
        assert not row['nonzero_edit'] and all(g['gate'] == 0 for g in row['source_gates'])
        pair_rows.append({'trial_index': i + 1, 'condition_id': row['id'], 'group': group,
            'A': {**feature(a), 'source': 'receiver', 'source_token': 'small', 'coefficient': a_coefficient},
            'B': {**feature(b), 'source': 'donor', 'source_token': 'hot', 'coefficient': b_coefficient},
            'P_cold': row['P_cold']})

inventory = {'visualization_date': '2026-10-14', 'experiment_date': '2026-10-13',
    'layer': 22, 'token_position': 9, 'indices': 'zero_based', 'random_seed': 20261013,
    'label_source': L['source'], 'single_unique_ids': 100, 'pair_count_per_direction': 100,
    'single_trials': single_rows, 'pair_trials': pair_rows,
    'labels_by_status': dict(Counter(LABELS[f]['status'] for f in IDS)),
    'source_sha256': {'results.json': sha(ROOT / 'artifacts/results.json'),
                      'labels.json': sha(ROOT / 'labels.json'),
                      'build_random_feature_lists.py': sha(__file__)}}
(OUT / 'random-control-feature-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
with (OUT / 'random-single-molt-feature-list.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(single_rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(single_rows)
with (OUT / 'random-pair-molt-feature-list.csv').open('w', newline='') as f:
    fields = ['trial_index', 'condition_id', 'group', 'receiver_id', 'receiver_label', 'receiver_coefficient',
              'donor_id', 'donor_label', 'donor_coefficient', 'P_cold']
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); writer.writeheader()
    for r in pair_rows:
        writer.writerow({'trial_index': r['trial_index'], 'condition_id': r['condition_id'], 'group': r['group'],
            'receiver_id': r['A']['id'], 'receiver_label': r['A']['label'], 'receiver_coefficient': r['A']['coefficient'],
            'donor_id': r['B']['id'], 'donor_label': r['B']['label'], 'donor_coefficient': r['B']['coefficient'], 'P_cold': r['P_cold']})

FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')
INK = '#1d3043'; MUTED = '#53647b'; ORANGE = '#a85720'; BLUE = '#24608a'; PURPLE = '#744286'


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.image = Image.new('RGB', (width, height), '#f5f7fa')
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    f'<rect width="{width}" height="{height}" fill="#f5f7fa"/>']
        self.features = []

    def rect(self, x, y, width, height, fill, radius=0):
        self.draw.rounded_rectangle((x, y, x + width, y + height), radius=radius, fill=fill)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}"/>')

    def text(self, x, y, value, size=24, fill=INK, bold=False, feature_id=None, side=None, trial=None):
        font = ImageFont.truetype(str(FONT_ROOT / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)
        assert x + self.draw.textlength(value, font=font) <= self.width - 40, value
        ascent, descent = font.getmetrics()
        assert y + ascent + descent <= self.height, value
        self.draw.text((x, y + ascent), value, font=font, fill=fill, anchor='ls')
        attrs = ''
        if feature_id is not None:
            attrs = f' data-feature-id="{feature_id}" data-side="{side}" data-trial="{trial}"'
            self.features.append((side, trial, feature_id))
        self.svg.append(f'<text x="{x}" y="{y + ascent}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{fill}"{attrs}>{html.escape(value)}</text>')

    def save(self, stem):
        self.image.save(OUT / (stem + '.png'), optimize=True)
        (OUT / (stem + '.svg')).write_text('\n'.join(self.svg) + '\n</svg>\n')
        # Bounded crops are local QA aids, not substitutes for the complete images.
        self.image.crop((0, 0, self.width, min(1150, self.height))).save(OUT / (stem + '-qa-top.png'))
        self.image.crop((0, self.height - 950, self.width, self.height)).save(OUT / (stem + '-qa-bottom.png'))
        ET.parse(OUT / (stem + '.svg'))


def heading(c, title, subtitle):
    c.text(66, 40, title, size=44, bold=True)
    c.text(66, 104, subtitle, size=25, fill=MUTED)
    c.text(66, 157, 'Receiver: ' + R['protocol']['receiver'], size=23)
    c.text(66, 196, 'Donor: ' + R['protocol']['donor'], size=23)
    c.text(66, 241, 'Recorded run: 13 October 2026 · Qwen3-4B · layer 22 · token 9 (zero-based) · attention unfrozen', size=22, fill=MUTED)


def footer(c, y):
    c.text(66, y, 'Names are the saved 12 October dashboard descriptions. † = limited evidence in that export.', size=22, fill=MUTED)
    c.text(66, y + 39, '“Not observed in 10M tokens” is the saved entry, not an invented feature name. M = MOLT/transform ID.', size=22, fill=MUTED)
    c.text(66, y + 78, 'All 100 sampled MOLTs had zero gates on both source tokens; these random trials applied zero edits.', size=22, fill=MUTED)


single = Canvas(2800, 4935)
heading(single, 'Random control feature inventory — single MOLTs',
        '100 IDs, shown twice: left adds the donor contribution; right subtracts the receiver contribution.')
body_y, row_height = 435, 43
panel_width = 1317
for side, x, title, equation, color in [
    ('left', 66, 'LEFT · 100 donor additions (+4.5)', 'receiver MLP += +4.5 × C_f(x_hot)', ORANGE),
    ('right', 1417, 'RIGHT · 100 receiver subtractions (−4.5)', 'receiver MLP += −4.5 × C_f(x_small)', BLUE)]:
    single.rect(x, 300, panel_width, 135 + 100 * row_height, '#ffffff', radius=12)
    single.text(x + 20, 318, title, size=30, bold=True, fill=color)
    single.text(x + 20, 365, equation, size=25, fill=color)
    for i, f in enumerate(IDS):
        y = body_y + i * row_height
        if i % 2 == 0: single.rect(x + 10, y - 3, panel_width - 20, row_height, '#f1f4f8')
        single.text(x + 20, y + 4, f'{i + 1:03d}', size=22, fill=MUTED)
        single.text(x + 92, y + 3, caption(f), size=24,
                    fill=MUTED if LABELS[f]['status'] == 'not_observed' else INK,
                    feature_id=f, side=side, trial=i + 1)
footer(single, 4770)
assert single.features == [('left', i + 1, f) for i, f in enumerate(IDS)] + [('right', i + 1, f) for i, f in enumerate(IDS)]
single.save('random-single-molt-feature-list')

pair = Canvas(3200, 8050)
heading(pair, 'Random control feature inventory — MOLT pairs',
        '100 pairs, shown twice: A always receives x_small; B always receives x_hot. Only the signs reverse.')
body_y, row_height = 445, 74
panel_width = 1517
for side, x, title, equation, ca, cb in [
    ('left', 66, 'LEFT · 100 pairs: −receiver A, +donor B', 'MLP += −4.5 × C_A(x_small) +4.5 × C_B(x_hot)', -4.5, 4.5),
    ('right', 1617, 'RIGHT · same 100 pairs, reversed signs', 'MLP += +4.5 × C_A(x_small) −4.5 × C_B(x_hot)', 4.5, -4.5)]:
    pair.rect(x, 300, panel_width, 145 + 100 * row_height, '#ffffff', radius=12)
    pair.text(x + 20, 318, title, size=30, bold=True, fill=PURPLE)
    pair.text(x + 20, 365, equation, size=25, fill=PURPLE)
    pair.text(x + 20, 405, 'A = receiver/small contribution; B = donor/hot contribution', size=21, fill=MUTED)
    for i, (a, b) in enumerate(PAIRS):
        y = body_y + i * row_height
        if i % 2 == 0: pair.rect(x + 10, y - 3, panel_width - 20, row_height, '#f1f4f8')
        pair.text(x + 20, y + 17, f'{i + 1:03d}', size=22, fill=MUTED)
        for offset, role, coefficient, f, color in [(0, 'A', ca, a, BLUE), (32, 'B', cb, b, ORANGE)]:
            sign = '+' if coefficient > 0 else '−'
            pair.text(x + 92, y + offset, f'{role} {sign}4.5', size=23, fill=color, bold=True)
            pair.text(x + 205, y + offset, caption(f), size=23,
                      fill=MUTED if LABELS[f]['status'] == 'not_observed' else INK,
                      feature_id=f, side=side, trial=i + 1)
footer(pair, 7880)
assert pair.features == [(side, i + 1, f) for side in ['left', 'right'] for i, ab in enumerate(PAIRS) for f in ab]
pair.save('random-pair-molt-feature-list')

css = 'body{margin:0;background:#f5f7fa;color:#1d3043;font:16px/1.5 system-ui}main{max-width:1300px;margin:auto;padding:28px}section{background:white;border:1px solid #d9e1eb;border-radius:12px;padding:22px;margin:22px 0}a{color:#24608a}img{display:block;width:100%;max-height:550px;object-fit:contain}table{width:100%;border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #e1e7ed;text-align:left;vertical-align:top}td{white-space:pre-wrap}input,select{font:inherit;padding:10px;margin:6px;border:1px solid #a6b3c5;border-radius:6px}.scroll{overflow-x:auto}.meta{color:#53647b;font-size:14px}'
payload = {'singles': [{'index': i + 1, 'left': '+4.5 donor: ' + caption(f), 'right': '−4.5 receiver: ' + caption(f)} for i, f in enumerate(IDS)],
           'pairs': [{'index': i + 1, 'left': '−4.5 receiver A: ' + caption(a) + '\n+4.5 donor B: ' + caption(b),
                       'right': '+4.5 receiver A: ' + caption(a) + '\n−4.5 donor B: ' + caption(b)} for i, (a, b) in enumerate(PAIRS)]}
page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actual random MOLT control features</title><style>{css}</style></head><body><main><p class="meta">Figures generated 14 October 2026 · recorded experiment 13 October · saved labels 12 October</p><h1>Which random MOLTs were used?</h1><p>These lists show the actual 100 sampled IDs and 100 pairings from the recorded layer-22, token-9 control experiment. The same IDs or pairs appear on both sides; the intervention direction changes. No new inference or GPU rental was needed.</p>'
for stem, title, detail in [('random-single-molt-feature-list', 'Figure 1 — 100 individual MOLTs, both signs', 'Left: +4.5 donor contribution. Right: −4.5 receiver contribution. Same 100 IDs; 200 separate trials.'),
                            ('random-pair-molt-feature-list', 'Figure 2 — 100 MOLT pairs, both sign directions', 'Left: −4.5 receiver A +4.5 donor B. Right: +4.5 receiver A −4.5 donor B. Same 100 pairs; 200 separate trials.')]:
    page += f'<section><h2>{title}</h2><p>{detail}</p><a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(title)}"></a><p>Each image contains every row. Open the image to zoom, or use the searchable text list below. <a href="{stem}.png">Full PNG</a> · <a href="{stem}.svg">Zoomable SVG</a> · <a href="{stem}.csv">CSV</a></p></section>'
page += '<section><h2>Search the exact IDs and names</h2><select id="mode"><option value="singles">Single-MOLT controls</option><option value="pairs">Pair controls</option></select><input id="search" placeholder="Search MOLT ID or feature name"><p id="count"></p><div class="scroll"><table><thead><tr><th>Trial</th><th>Left side</th><th>Right side</th></tr></thead><tbody id="rows"></tbody></table></div></section>'
page += '<section><h2>Reading the labels</h2><p>54 of the sampled IDs have the saved entry “Not observed in 10M tokens”; 42 have status complete and four have limited evidence, marked †. Names are descriptions from the saved dashboard, not selection criteria or demonstrated causal meanings.</p><p>All 100 sampled gates are zero at both exact source tokens. These lists identify the uniform random controls, rather than the separate nonzero active-pair controls. The selected T2339/T941 intervention was excluded from random sampling.</p><p><a href="random-control-feature-inventory.json">Complete inventory JSON</a> · <a href="figure-verification.json">Count and provenance verification</a> · <a href="../2026-10-13-small-hot-controls/index.html">Original control results</a> · <a href="README.md">Reproduction and provenance</a></p></section>'
page += '<script id="data" type="application/json">' + json.dumps(payload).replace('</', '<\\/') + '</script><script>const data=JSON.parse(document.getElementById("data").textContent);function render(){const mode=document.getElementById("mode").value,q=document.getElementById("search").value.toLowerCase();const rows=data[mode].filter(r=>(r.left+" "+r.right+" "+r.index).toLowerCase().includes(q));const body=document.getElementById("rows");body.replaceChildren();for(const r of rows){const tr=document.createElement("tr");for(const value of [r.index,r.left,r.right]){const td=document.createElement("td");td.textContent=value;tr.append(td)}body.append(tr)}document.getElementById("count").textContent=rows.length+" of 100 recorded trials in each sign direction";}for(const id of ["mode","search"])document.getElementById(id).addEventListener("input",render);render();</script></main></body></html>'
(OUT / 'index.html').write_text(page)

verification = {'status': 'passed', 'original_results_sha256': inventory['source_sha256']['results.json'],
    'saved_labels_sha256': inventory['source_sha256']['labels.json'], 'single_feature_entries': len(single.features),
    'pair_feature_entries': len(pair.features), 'single_ids_per_side': 100, 'pairs_per_side': 100,
    'single_rows_match_recorded_interventions': True, 'pair_rows_match_recorded_interventions': True,
    'all_labels_copied_by_feature_id': True, 'all_sampled_source_gates_zero': True,
    'single_png_dimensions': list(single.image.size), 'pair_png_dimensions': list(pair.image.size),
    'source_sha256': inventory['source_sha256'], 'notes': '100 single IDs shown twice; 100 pairs shown twice. Complete labels are preserved without truncation. QA crops are not the complete figures.'}
(OUT / 'figure-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps(verification, indent=2))
