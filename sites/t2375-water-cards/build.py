"""Regenerate the local water-feature cards from the measured result files."""
import csv
import json
import re
import shutil
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from math import ceil, isfinite
from pathlib import Path
from statistics import median

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
RESULTS = ROOT / 'results'
PILOT = RESULTS / '2026-11-21-t2375-water-continuations/pilot.csv'
INITIAL = RESULTS / '2026-11-21-water-blind'
EXTENDED = RESULTS / '2026-11-21-water-thousand'
ABLATION = RESULTS / '2026-11-21-water-modifier-ablation'

PILOT_PICKS = {
    'Noun modifier': [(4260, 182, 'water scarcity'), (3390, 307, 'water sports'),
                      (2230, 691, 'water quality'), (8548, 713, 'water slides'), (6413, 913, 'water flow')],
    'Standalone noun': [(2359, 753, 'clean air, water, and food practices'), (2918, 105, 'hot sauce, water, and salt'),
                        (7009, 108, 'With the help of water all the body’s systems'), (2778, 381, 'combine the water and sugar'),
                        (6652, 202, 'water or any kind of stock')],
    'Joined word': [(6138, 547, 'watercolor'), (183, 401, 'waterways'), (2230, 670, 'watersheds')],
    'Hyphenated adjective': [(3322, 1023, 'water-saving showerhead'), (6730, 311, 'water-related activities')],
    'Fixed expression': [(5600, 766, 'high-water mark')],
}
INITIAL_PICKS = {
    'Noun modifier': [(183, 'water cycles'), (224, 'Water sports'), (184, 'water management'),
                      (1, 'water utilities'), (35, 'water molecules')],
    'Other use': [(111, 'Bring water, snacks'), (13, 'rice, water, and koji'), (18, '1 cup of water'),
                  (3, 'clean water and sanitation'), (234, 'Bring the water to a boil')],
}

def read_json(path):
    return json.loads(path.read_text())

@dataclass
class Case:
    id: int
    conversation: int
    position: int
    use: str
    gate: float
    left: str
    token: str
    right: str

    def highlight(self, phrase):
        full = self.left + self.token + self.right
        anchor = len(self.left) + len(self.token) - len(self.token.lstrip())
        token = self.token.strip()
        for match in re.finditer(re.escape(phrase), full):
            i = anchor - match.start()
            if 0 <= i and phrase[i:i + len(token)] == token:
                return escape(phrase[:i]) + '<mark>' + escape(token) + '</mark>' + escape(phrase[i + len(token):])
        raise ValueError(f'Excerpt does not cover measured token at {self.conversation}/{self.position}: {phrase!r}')

    def snippet(self, left_chars=55, right_chars=75):
        left = re.split(r'<\|im_(?:end|start)\|>', self.left)[-1]
        right = re.split(r'<\|im_(?:end|start)\|>', self.right)[0]
        if len(left)>left_chars:
            left = '…' + left[-left_chars:].split(' ',1)[-1]
        if len(right)>right_chars:
            right = right[:right_chars].rsplit(' ',1)[0] + '…'
        return escape(left) + '<mark>' + escape(self.token) + '</mark>' + escape(right)

def stat(value, label):
    return f'<div><strong>{value}</strong><small>{label}</small></div>'

def gate_histogram(rows, title):
    # Bin 0 is the exact zero mass. Positive bins have width 0.5 and include their upper edge.
    counts = [0] * 8
    for row in rows:
        assert isfinite(row.gate) and 0 <= row.gate <= 3.5, row.gate
        counts[ceil(row.gate * 2)] += 1
    total = len(rows)
    descriptions = []
    bars = []
    baseline = 70
    height = 52
    for i, count in enumerate(counts):
        x = 42 if i == 0 else 72 + 20 * (i - 1)
        name = 'Exactly zero' if i == 0 else f'Above {(i-1)/2:g} through {i/2:g}'
        share = count / total
        description = f'{name}: {count} of {total} ({share:.1%})'
        descriptions.append(description)
        bar_height = height * share
        top = baseline - bar_height
        color = 'zero' if i == 0 else 'positive'
        label = f'<text x="{x+8}" y="{top-3:.2f}" class="bin-count">{count}</text>' if count else ''
        bars.append(f'<g data-count="{count}"><title>{escape(description)}</title>'
                    f'<rect class="{color}" x="{x}" y="{top:.2f}" width="16" height="{bar_height:.2f}"/>'
                    f'{label}<rect class="hit" x="{x-2}" y="7" width="20" height="65"/></g>')
    named = f'Gate histogram for {title}, {total} observations. Heights are percentages of the group. ' + '. '.join(descriptions)
    return (f'<figure class="histogram"><figcaption>Gate distribution</figcaption>'
            f'<svg role="img" aria-label="{escape(named)}" viewBox="0 0 219 91" data-total="{total}">'
            f'<title>{escape(title)} gate histogram</title>'
            '<path class="gridline" d="M38 18H211 M38 44H211"/><path class="axis" d="M38 70H61 M70 70H211"/>'
            '<text class="ylabel" x="34" y="21">100%</text><text class="ylabel" x="34" y="47">50%</text>'
            '<text class="ylabel" x="34" y="73">0%</text>'
            + ''.join(bars)
            + '<g class="xlabel"><text x="50" y="86">zero</text><text x="71" y="86">0</text>'
              '<text x="111" y="86">1</text><text x="151" y="86">2</text>'
              '<text x="191" y="86">3</text><text x="209" y="86">3.5</text></g>'
              '</svg></figure>')


def card(title, rows, examples, denominator, active_only=False, sample_size=None):
    n = len(rows)
    fired = sum(r.gate > 0 for r in rows)
    count = n if active_only else fired
    percent = (Decimal(100 * count) / denominator).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    count_label = f'{count:,} / {denominator:,} active' if active_only else f'{count:,} of {denominator:,} fired'
    strength = median(r.gate for r in rows)
    stats = stat(f'{percent}%', count_label) + stat(f'{strength:.2f}', 'median gate')
    assert len(examples) == min(5, n)
    assert len({(r.conversation,r.position) for r, _ in examples}) == len(examples)
    share = f'<span class="share">{n:,} / {sample_size:,} sampled ({100*n/sample_size:.1f}%)</span>' if sample_size else ''
    example_html = ''.join(f'<li><span>{r.highlight(phrase)}</span><b>{r.gate:.2f}</b></li>' for r, phrase in examples)
    return (f'<article><div class="card-top"><div class="card-summary"><h3>{escape(title)}{share}</h3>'
            f'<div class="metrics">{stats}</div></div>{gate_histogram(rows,title)}</div>'
            f'<ul aria-label="Examples and gate values">{example_html}</ul></article>')

def section(title, note, cards, grid, conclusion=''):
    return f'<section><div class="head"><h2>{escape(title)}</h2><p>{note}</p></div><div class="{grid}">{"".join(cards)}</div>{conclusion}</section>'

def table(rows):
    tr = ''.join(f'<tr><td>{escape(r.use)}</td><td>{r.snippet()}</td><td>{r.gate:.2f}</td><td>{r.conversation} / {r.position}</td></tr>' for r in rows)
    return '<div class="scroll" tabindex="0"><table><thead><tr><th>Use</th><th>Text around the measured token</th><th>Gate</th><th>Chat / token</th></tr></thead><tbody>'+tr+'</tbody></table></div>'

def disclosure(title, content):
    return f'<details><summary>{escape(title)}</summary>{content}</details>'

def pilot_section():
    with PILOT.open() as f:
        rows = [Case(i, int(r['conversation']), int(r['position']), r['observed_use'], float(r['gate']),
                     r['prefix'], r['measured_token'], r['continuation'])
                for i,r in enumerate(csv.DictReader(f)) if float(r['gate'])>0]
    assert len(rows)==32
    by_position={(r.conversation, r.position):r for r in rows}
    cards=[]
    for name,picks in PILOT_PICKS.items():
        subset=[r for r in rows if r.use==name]
        examples=[(by_position[(c,p)], phrase) for c,p,phrase in picks]
        assert all(r.use == name for r, _ in examples)
        cards.append(card(name, subset, examples, len(rows), active_only=True))
    return section('What followed “water” in the 32 active pilot cases?',
                   'The pilot began with 100 unique dashboard evaluation windows. Across 29 conversations, 32 distinct active positions contained “water.” Each percentage uses those 32 as its denominator.',
                   cards, 'pilot'), disclosure('All 32 active pilot cases',table(rows))

def initial_section():
    rows = [Case(r['id'],r['conversation'],r['position'],'Noun modifier' if r['modifier'] else 'Other use',
                 r['gate'],r['before'],r['token'].strip(),r['after']) for r in read_json(INITIAL/'observations.json')]
    idx={r.id:r for r in rows}
    assert len(idx)==240
    cards=[]
    for name,picks in INITIAL_PICKS.items():
        subset=[r for r in rows if r.use==name]
        examples=[(idx[i],phrase) for i,phrase in picks]
        assert all(r.use==name for r,_ in examples)
        cards.append(card(name,subset,examples,len(subset),sample_size=len(rows)))
    est=read_json(INITIAL/'summary.json')['differences']['rate']
    conclusion=f'<p class="note">Firing-rate difference, modifier minus other &nbsp;<b>{100*est["estimate"]:.1f} points</b> (95% interval {100*est["lower"]:.1f}–{100*est["upper"]:.1f}).</p>'
    return section('Earlier random sample · 240 whole-word occurrences','199 conversations. Each rate uses all cases in that label group, and each median includes zeros.',cards,'random',conclusion), disclosure('All 240 whole-word observations',table(rows))

CATEGORIES = {
    'M': 'Noun modifier',
    'S': 'Standalone noun',
    'J': 'Joined or inflected word',
    'H': 'Hyphenated adjective',
    'F': 'Fixed expression',
    'O': 'Other use',
}

def extended_section():
    data = read_json(EXTENDED/'observations.json')
    summary = read_json(EXTENDED/'summary.json')
    sampling = read_json(EXTENDED/'sampling.json')
    selections = read_json(SITE/'examples-1000.json')
    rows = [Case(r['id'], r['conversation'], r['position'], CATEGORIES[r['category']],
                 r['gate'], r['before'], r['token'], r['after']) for r in data]
    by_id = {r.id:r for r in rows}
    assert len(by_id) == len(rows) == 1000
    cards = []
    for code,title in CATEGORIES.items():
        subset = [r for r in rows if r.use == title]
        expected = summary['groups'].get(code, {'n':0})
        assert len(subset) == expected['n']
        if not subset:
            cards.append(f'<article><h3>{escape(title)}</h3><div class="empty">0 of 1,000 sampled</div><p class="note">Firing rate unavailable.</p></article>')
            continue
        assert sum(r.gate>0 for r in subset) == expected['fires']
        examples = [(by_id[ident], phrase) for ident,phrase in selections[code]]
        assert all(r.use == title for r,_ in examples)
        cards.append(card(title, subset, examples, len(subset), sample_size=len(rows)))
    joined = read_json(EXTENDED/'joined-position-exploratory.json')
    first, later = joined['word_initial'], joined['after_a_letter']
    assert first['n']+later['n']==summary['groups']['J']['n']
    assert first['fired']+later['fired']==summary['groups']['J']['fires']
    join_note = (f'<p class="joined-note"><b>Inside joined words.</b> In a post-hoc check, {first["fired"]}/{first["n"]} '
                 f'({first["fired"]/first["n"]:.1%}) fired when “water” began the word, such as <i>watercolor</i>. '
                 f'{later["fired"]}/{later["n"]} fired when a letter preceded it, such as <i>underwater</i>. '
                 'Token boundaries differed across spellings.</p>')
    caption = (f"{sampling['sampled_conversations']:,} UltraChat conversations. Overall, {summary["fires"]:,} of 1,000 fired ({summary["fires"]/10:.1f}%). The large percentages are within each use. Medians include zeros.")
    methods = disclosure('How the 1,000 occurrences were drawn', f'<p class="method">We sampled 1,000 occurrences without replacement from {sampling["population"]:,} eligible occurrences across {sampling["eligible_conversations"]:,} conversations. Each occurrence fell within the first 1,024 formatted tokens. We excluded the original dashboard conversations and every conversation in the earlier 240-case study. Two text-only annotators divided the sample into halves and assigned one label per occurrence before gates were measured. The fixed-expression group includes titles and idioms. The reported 95% intervals resample whole conversations 10,000 times. For an embedded spelling such as “underwater,” the highlighted token contains the final “r” of “water” and may include adjacent letters. The model saw the text only through that token.</p>')
    return section('Firing rates across 1,000 random occurrences',caption,cards,'large',extended_contrast(summary)+join_note), disclosure('All 1,000 observations',table(rows)), methods

def extended_contrast(summary):
    main = summary['contrasts']['M_minus_S']
    other = summary['contrasts']['M_minus_other']
    return (f'<p class="note">Noun modifier minus standalone noun &nbsp;<b>{100*main["estimate"]:+.1f} percentage points</b> '
            f'(95% interval {100*main["lower"]:+.1f} to {100*main["upper"]:+.1f}). '
            f'Against all other uses combined, the difference was {100*other["estimate"]:+.1f} points '
            f'({100*other["lower"]:+.1f} to {100*other["upper"]:+.1f}). The three smallest groups contain only 10–20 cases each.</p>')

def probability(value):
    percent = 100 * value
    if 0 < percent < 0.01:
        return f'{percent:.2g}%'
    return f'{percent:.2f}%'

def path_text(sequence):
    # Quoted strings expose leading spaces, punctuation, and escaped newlines.
    return '<code>' + escape(json.dumps(sequence, ensure_ascii=False)) + '</code>'

def ablation_table(rows):
    entries = []
    for row in rows:
        values = [str(row['id']), f"{row['gate']:.2f}", path_text(row['actual_sequence']),
                  probability(row['actual_on']), probability(row['actual_off']),
                  path_text(row['pred_sequence']), probability(row['predicted_on']),
                  probability(row['predicted_off']), path_text(row['off_greedy_sequence'])]
        entries.append('<tr>' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>')
    columns = ['Sample', 'Gate', 'Recorded path', 'Intact', 'Gate off',
               'Intact greedy path', 'Intact', 'Gate off', 'Gate-off greedy path']
    return ('<p class="method">Each probability scores the exact quoted token sequence in its pair. '
            'The last column shows the sequence chosen greedily after ablation.</p>'
            '<div class="scroll case-data" role="region" aria-label="Per-occurrence probabilities" tabindex="0">'
            '<table><thead><tr>' + ''.join(f'<th scope="col">{c}</th>' for c in columns)
            + '</tr></thead><tbody>' + ''.join(entries) + '</tbody></table></div>')

def ablation_section():
    import xml.etree.ElementTree as ET

    rows = read_json(ABLATION/'paired.json')
    expected = {row['id'] for row in read_json(EXTENDED/'observations.json') if row['category']=='M'}
    assert len(rows)==len(expected)==179 and {r['id'] for r in rows}==expected
    assert sum(row['gate']==0 for row in rows)==22
    assert all(row['actual_on']==row['actual_off'] and row['predicted_on']==row['predicted_off'] for row in rows if row['gate']==0)
    assert len([r for r in rows if not r['strict_adjacent']])==4
    assert sum(len(r['target_ids'])==2 for r in rows)==17
    for row in rows:
        for key in ('actual_on','actual_off','predicted_on','predicted_off'):
            assert 0 < row[key] <= 1
    with (ABLATION/'cases.csv').open() as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    assets = SITE/'ablation'
    assets.mkdir(exist_ok=True)
    for name in ('scatter.svg', 'cases.csv', 'summary.json'):
        shutil.copyfile(ABLATION/name,assets/name)
    svg=(assets/'scatter.svg').read_bytes()
    assert ET.fromstring(svg).tag.endswith('svg')
    filename_label = ('Model probabilities with T2375 intact on the horizontal axis and its contribution removed '
                      'at the water token on the vertical axis. Left, the recorded next word. Right, the intact greedy sequence.')
    graphic = svg.decode('utf-8')
    graphic = graphic[graphic.index('<svg '):].replace('<svg ', f'<svg role="img" aria-label="{escape(filename_label)}" ', 1)
    assert graphic.count('id="actual_case_')==graphic.count('id="predicted_case_')==179
    figure = (f'<figure class="ablation-figure"><div class="plot-scroll" tabindex="0" role="region" '
              f'aria-label="Two probability scatter plots, scroll horizontally on small screens">'
              f'{graphic}</div>'
              '<figcaption><span class="mobile-plot-hint">Scroll horizontally to see both panels. </span>One point per occurrence in each panel. A point below the diagonal means ablation lowered the probability '
              'of that sequence. Points on the diagonal retained the same probability. Closely matching cases can overlap.</figcaption></figure>')
    summary = read_json(ABLATION/'summary.json')
    all_results = summary['all']
    assert all_results['actual']['n'] == all_results['predicted']['n'] == len(rows)
    stats_rows = []
    for key, name in [('actual','Recorded sequence'), ('predicted','Intact greedy sequence')]:
        value = all_results[key]
        lo, hi = value['mean_delta_pp_ci95']
        stats_rows.append(f'<tr><th scope="row">{name}</th><td>{probability(value["intact_mean"])}</td>'
                          f'<td>{probability(value["off_mean"])}</td><td>{value["mean_delta_pp"]:+.2f}</td>'
                          f'<td>{lo:+.2f} to {hi:+.2f}</td></tr>')
    changes = [r for r in rows if r['on_top_id'] != r['off_top_id']]
    tied = sum(r['on_max_ties']>1 or r['off_max_ties']>1 for r in changes)
    assert len(changes) == summary['first_top_changed'] and tied == 3
    results = ('<div class="scroll paired-stats"><table><caption>Mean sequence probability across all 179 occurrences</caption>'
               '<thead><tr><th>Target</th><th>Intact</th><th>Gate off</th><th>Change (points)</th>'
               '<th>95% interval (points)</th></tr></thead><tbody>' + ''.join(stats_rows) + '</tbody></table></div>'
               f'<p class="note">The first greedy token changed in <b>{summary["first_top_changed"]} of 179</b> cases. '
               f'The first greedy word changed in {summary["greedy_word_changed"]}. Of the {len(changes)} token changes, {tied} involved '
               f'a tie for the highest logit in one condition at BF16 precision. The other {len(changes)-tied} changed between unique winners. '
               'The recorded sequence moved by at least one percentage '
               f'point in {all_results["actual"]["abs_change_ge_1pp"]} cases. All 22 naturally silent cases retained identical probabilities. '
               'The intervals resample whole conversations.</p>')
    strict = summary['strict_adjacent']
    example = next(r for r in rows if r['id']==885)
    assert example['before'].endswith("Sydney's") and example['actual_sequence'].strip()=='supply'
    example_text = (f'<p class="ablation-example"><b>One prediction that switched.</b> After “Sydney’s <mark>water</mark>,” '
                    f'the recorded word <i>{escape(example["actual_sequence"].strip())}</i> rose from '
                    f'{probability(example["actual_on"])} to {probability(example["actual_off"])}. '
                    f'The intact model’s first choice, <i>{escape(example["pred_sequence"].strip())}</i>, fell '
                    f'from {probability(example["predicted_on"])} to {probability(example["predicted_off"])}. '
                    f'With the gate off, the model chose <i>{escape(example["off_greedy_sequence"].strip())}</i> (sample 885).</p>')
    sensitivity = (f'<p class="note">Four frozen modifier labels contained an intervening conjunction or punctuation. For the '
                   f'175 strictly adjacent cases, the mean changes were {strict["actual"]["mean_delta_pp"]:+.2f} points for the recorded sequence '
                   f'and {strict["predicted"]["mean_delta_pp"]:+.2f} for the intact greedy sequence.</p>')
    links = ('<p class="plot-links"><a href="ablation/scatter.svg">Open full-size plot</a>'
             '<a href="ablation/cases.csv" download>Download the 179 cases (CSV)</a>'
             '<a href="ablation/summary.json" download>Download summary (JSON)</a></p>')
    heading = ('<div class="head"><h2>What changes when T2375 is switched off at “water”?</h2>'
               '<p>We removed T2375’s learned contribution at the measured token in each of the 179 noun-modifier cases. '
               'The left panel scores the word that followed in UltraChat. The right panel scores the word sequence that '
               'the intact model chose greedily. Each point compares the same token sequence before and after the intervention.</p></div>')
    method = ('<p class="method">Subsequent positions receive the effect of the edited prefix without any further intervention. '
              'Both panels score the continuation through the word, including any leading punctuation. Seventeen recorded continuations '
              'required two tokens. Probabilities score the particular token sequence through the word, without a trailing word boundary.</p>')
    return ('<section class="ablation" id="modifier-ablation">'+heading+figure+results+example_text+sensitivity+links+
            disclosure('Inspect all 179 paired predictions',method+ablation_table(rows))+'</section>')


def kl_tail_comparison(folder):
    tails = read_json(folder / 'tails.json')
    with (folder / 'paired.csv').open() as stream:
        measured = {r['id']: r for r in csv.DictReader(stream)}
    with (ABLATION / 'cases.csv').open() as stream:
        predictions = {r['id']: r for r in csv.DictReader(stream)}

    def fraction(k, n):
        return f'{k}/{n} ({k/n:.0%})'

    cards = []
    for band, threshold, ident, prefix in [
        ('low', 'Below 0.0001 nats', '24', 'Turn off the water'),
        ('high', 'At least 0.01 nats', '885', 'Sydney’s water'),
    ]:
        group = tails['target'][band]
        n = group['n']
        pred = predictions[ident]
        case = measured[ident]
        assert case['conversation'] == pred['conversation'] and case['position'] == pred['position']
        tokens = [('supply', 'actual_first_on', 'actual_first_off')]
        assert pred['actual_sequence'].strip() == 'supply'
        if pred['on_top_token'] != pred['actual_sequence']:
            tokens.append((pred['on_top_token'].strip(), 'on_top_first_on', 'on_top_first_off'))
        probs = ''.join(f'<tr><th scope="row">{escape(token)}</th><td>{float(pred[on]):.2%}</td>'
                        f'<td>{float(pred[off]):.2%}</td></tr>' for token, on, off in tokens)
        cards.append(
            f'<article><h4>{band.capitalize()} KL <span>{threshold}</span></h4>'
            f'<p class="tail-size"><b>{n} / {tails["total"]}</b> occurrences ({n/tails["total"]:.1%})</p>'
            '<dl><div><dt>Intact model correctly predicted the next token</dt>'
            f'<dd>{fraction(group["correct"], n)}</dd></div>'
            '<div><dt>The same two tokens occurred consecutively, in this order, earlier in the conversation</dt>'
            f'<dd>{fraction(group["repeated"], n)}</dd></div>'
            f'<div><dt>Median T2375 gate</dt><dd>{group["median_gate"]:.2f}</dd></div></dl>'
            f'<h5>“{prefix}”</h5><p class="tail-context">Recorded next token <b>supply</b>'
            f' · KL {float(case["target_kl"]):.7f} nats</p>'
            f'<table aria-label="Next-token probabilities after {prefix}"><thead><tr>'
            f'<th scope="col">Next token</th><th scope="col">T2375 on</th><th scope="col">T2375 off</th>'
            f'</tr></thead><tbody>{probs}</tbody></table></article>'
        )
    low, high = tails['control']['low'], tails['control']['high']
    middle = tails['total'] - sum(g['n'] for g in tails['target'].values())
    return (
        '<div id="kl-high-low" class="kl-tails"><h3>When did switching off T2375 change the prediction most?</h3>'
        '<p>The intact model predicted the recorded next token more often in low-KL cases. '
        f'The cards compare the two ends of the distribution. The middle range contains {middle} occurrences.</p>'
        '<div class="tail-grid">' + ''.join(cards) + '</div>'
        '<p>The matched controls showed a similar pattern. The intact model predicted the recorded token in '
        f'{fraction(low["correct"], low["n"])} low-control-KL cases and {fraction(high["correct"], high["n"])} '
        'high-control-KL cases.</p></div>'
    )


def kl_section():
    folder = RESULTS / '2026-11-21-water-active-kl-control'
    if not (folder / 'summary.json').is_file():
        return ''
    summary = read_json(folder / 'summary.json')
    target, control = summary['target'], summary['control']
    assets = SITE / 'kl'
    assets.mkdir(exist_ok=True)
    for name in ('active-control-kl.svg', 'paired.csv'):
        shutil.copyfile(folder / name, assets / name)
    n = target['n']
    m = control['n']
    coverage = (f'Every occurrence has one matched control.' if m == n else
                f'{m} occurrences had another active layer-29 MOLT and supplied a control.')
    control_larger = m - summary['target_larger'] - summary['ties']
    rows = ''.join(
        f'<tr><th scope="row">{label}</th><td>{values["n"]}</td>'
        f'<td>{values["median"]:.5f}</td><td>{values["mean"]:.5f}</td></tr>'
        for label, values in [('T2375 off', target), ('Active control off', control)]
    )
    return (
        '<section id="modifier-kl" class="ablation"><div class="head">'
        '<h2>How much does switching off one active MOLT change the prediction after “water”?</h2>'
        f'<p>T2375 fired in {n} of the 1,000 sampled occurrences. Each occurrence contributes one green observation. '
        'We switched off T2375 only at the indexed token and compared the model’s final next-token probabilities with the intact run. '
        'For the orange observations, we switched off a different MOLT that fired at the same layer on the same token, leaving T2375 intact. '
        f'{coverage}</p></div>'
        '<figure class="ablation-figure"><img src="kl/active-control-kl.svg" '
        'style="display:block;margin:auto;width:100%;max-width:880px;height:auto" '
        'alt="Two distributions of immediate next-token KL divergence, grouped into five ranges from below 0.0001 to 0.1 nats or higher.">'
        '<figcaption>Read left to right for larger changes in prediction. Each bar counts occurrences in that KL range. '
        'Zero means the ablated and intact runs predicted identical probabilities for every token. '
        'A value on a bin boundary is counted in the bin to its right.</figcaption></figure>'
        + kl_tail_comparison(folder) +
        '<div class="scroll paired-stats"><table><thead><tr><th>Ablation at “water”</th><th>Occurrences</th>'
        '<th>Median KL (nats)</th><th>Mean KL (nats)</th></tr></thead><tbody>'
        + rows + '</tbody></table></div>'
        f'<p class="note">Switching off the matched control changed the prediction more in <b>{control_larger} of {m} paired occurrences</b> '
        f'({control_larger/m:.1%}).</p>'
        '<p class="method">We chose the other active layer-29 MOLT whose gate was closest to T2375’s gate before comparing predictions. '
        f'This selected {summary["distinct_controls"]} distinct control MOLTs. Both comparisons use KL(intact ‖ ablated) over the whole output vocabulary. '
        'The sample includes joined and inflected words. In those examples the indexed token is the one containing the final “r” of “water”, '
        'and the prediction is for the token immediately after it.</p>'
        '<p class="plot-links"><a href="kl/paired.csv" download>Download occurrence-level KL and control IDs (CSV)</a>'
        '<a href="kl/active-control-kl.svg">Open full-size figure</a></p></section>'
    )


def build():
    initial,initial_table = initial_section()
    pilot,pilot_table = pilot_section()
    methods=disclosure('How to read the samples and gates','<p class="method">T2375 is a learned transform for Qwen3-4B’s layer 29. Each highlighted token carries the displayed gate. A positive value counts as firing. In the 240-case study, occurrences of the whole word “water” were sampled before measurement, labeled using the continuation, and measured using only the model prefix through “water.” The interval resamples whole conversations 10,000 times. Joined and hyphenated words were outside this sample.</p>')
    body=initial+pilot
    notes=initial_table+pilot_table+methods
    if (EXTENDED/'summary.json').exists():
        large, large_table, broad_methods = extended_section()
        followup = ablation_section() if (ABLATION/'scatter.svg').is_file() and (ABLATION/'summary.json').is_file() and (ABLATION/'cases.csv').is_file() else ''
        body = large + followup + kl_section() + initial + pilot
        notes = large_table + initial_table + pilot_table + broad_methods + methods
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>When does T2375 fire on water?</title><style>'''+(SITE/'cards.css').read_text()+'''</style></head><body><main><header><h1>When does T2375 fire on “water”?</h1><span>Qwen3-4B &nbsp;·&nbsp; Layer 29</span></header><p class="legend">Highlighted tokens carry the gate values beside each excerpt. Histogram heights show the percentage of cases in each card. The gray bar counts exact zeros. Green bars cover positive gates in steps of 0.5, and the numbers above the bars are counts.</p>'''+body+'<div class="notes">'+notes+'</div></main></body></html>'
    tmp=SITE/'index.html.tmp'
    tmp.write_text(page)
    tmp.replace(SITE/'index.html')

if __name__=='__main__':
    build()
