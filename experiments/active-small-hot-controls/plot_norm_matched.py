"""CPU-only figures and website section for saved L2-matched MOLT controls.

No model loading or GPU access. Probabilities come from the independently
verified float64 softmax of the saved full-vocabulary BF16 logits.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np

FORWARD = '#067a80'
REVERSE = '#9353a4'
REFERENCE = '#183448'
START = '<!-- norm-matched-controls:start -->'
END = '<!-- norm-matched-controls:end -->'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root):
    rows = []
    evidence = {}
    for directory in [root, root / 'fixed-site']:
        result_path = directory / 'artifacts/results.json'
        verify_path = directory / 'verification.json'
        result = json.loads(result_path.read_text())
        verification = json.loads(verify_path.read_text())
        assert result['status'] == 'complete' and verification['status'] == 'verified'
        for name, expected in result['source_sha256'].items():
            assert digest(directory / name) == expected, name
        for path in [result_path, verify_path]:
            evidence[str(path.relative_to(root))] = digest(path)
        for row in result['rows']:
            row = {**row, **verification['probabilities'][row['id']]}
            row['edit_layer'] = row.get('edit_layer', row['layer'])
            row['effective_terms'] = []
            gates = {(g['source'], g['feature']): g['gate'] for g in row['source_gates']}
            for source, feature_id, coefficient in row['terms']:
                feature = result['selection']['features'][f"{row['layer']}:{feature_id}"]
                row['effective_terms'].append({
                    'source': source, 'layer': row['layer'], 'id': feature_id,
                    'label': feature['label'],
                    'source_gate': gates[source, feature_id],
                    'coefficient': coefficient * row.get('norm_match_scale', 1),
                })
            rows.append(row)
    controls = [r for r in rows if r['stratum'].startswith('norm_matched') and r['edit_layer'] == 22]
    references = {kind: next(r for r in rows if r['id'] == f'selected_{kind}_4.5')
                  for kind in ['positive', 'negative', 'pair']}
    assert len(controls) == 50
    assert all(r['nonzero_actual_edit'] and r['logits_changed_from_clean'] for r in controls)
    assert all(t['source_gate'] > 0 for r in controls for t in r['effective_terms'])
    for row in controls:
        target = references[row['norm_match_to']]['intended_delta_norm']
        assert math.isclose(row['intended_delta_norm'], target, rel_tol=1e-6)
    return controls, references, evidence


def feature_text(row, source):
    term = next(t for t in row['effective_terms'] if t['source'] == source)
    label = term['label']
    if label == 'No cached dashboard label':
        label = 'Label unavailable in saved dashboard export'
    return f"L{term['layer']}–T{term['id']} — {label}"


def pair_label(row):
    return '\n'.join(textwrap.fill(prefix + feature_text(row, source), 80,
                                   subsequent_indent='    ')
                     for prefix, source in [('R: ', 'receiver'), ('D: ', 'donor')])


def axes_format(ax, count, low):
    ax.set_xscale('log')
    ax.set_xlim(low, 1e3)
    ax.set_ylim(count - .4, -.7)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(axis='x', color='#d9e0e6', linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('P(cold), % of full vocabulary · logarithmic scale', fontsize=11)
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)


def pair_figure(root, controls, reference):
    forward = sorted([r for r in controls if r['kind'] == 'pair'],
                     key=lambda r: (r['layer'] != 22, r['layer'], r['sample_index']))
    reverse = {(r['layer'], tuple(t[1] for t in r['terms'])): r
               for r in controls if r['kind'] == 'pair_reverse'}
    assert len(forward) == len(reverse) == 13
    pairs = [(r, reverse[r['layer'], tuple(t[1] for t in r['terms'])]) for r in forward]
    count = 1 + len(pairs)
    fig = plt.figure(figsize=(19.2, 13.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.72, 1.05, .82],
                           left=.025, right=.985, bottom=.12, top=.825, wspace=.035)
    labels, probability, norms = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax in [labels, norms]:
        ax.set_xlim(0, 1)
        ax.set_ylim(count - .4, -.7)
        ax.axis('off')
    low_value = min(r['P_cold'] * 100 for pair in pairs for r in pair)
    low = 10 ** (math.floor(math.log10(low_value)) - 1)
    axes_format(probability, count, low)
    for ax in [labels, probability, norms]:
        ax.axhspan(-.65, .47, color='#e6f2ee', zorder=0)
        for y in range(1, count):
            if y % 2 == 0:
                ax.axhspan(y - .47, y + .47, color='#f5f7fa', zorder=0)
    labels.set_title('Feature names · R = receiver “small”; D = donor “hot”',
                     fontsize=12, loc='left', pad=20)
    probability.set_title('Same intended intervention norm', fontsize=12, pad=20)
    norms.set_title('Actual applied L2 norm', fontsize=12, pad=20)
    for x, text in [(.22, '−R +D'), (.76, '+R −D')]:
        norms.text(x, 1.005, text, transform=norms.transAxes,
                   ha='center', va='bottom', color=FORWARD if x < .5 else REVERSE, fontsize=11)
    labels.text(.01, 0, 'SELECTED PAIR · 4.5×\n' + pair_label(reference),
                fontsize=11.2, va='center', weight='bold', color=REFERENCE)
    probability.scatter(reference['P_cold'] * 100, 0, marker='*', s=240,
                        color=REFERENCE, zorder=5)
    probability.annotate(f"{reference['P_cold'] * 100:.4f}%", (reference['P_cold'] * 100, 0),
                         xytext=(-7, -22), textcoords='offset points', ha='right',
                         fontsize=10.8, weight='bold', color=REFERENCE)
    norms.text(.22, 0, f"{reference['actual_delta_norm']:.4f}",
               ha='center', va='center', fontsize=11.5, weight='bold', fontfamily='monospace')
    norms.text(.76, 0, '—', ha='center', va='center', fontsize=12)
    best = max((r for pair in pairs for r in pair), key=lambda r: r['P_cold'])
    for y, (normal, reversed_row) in enumerate(pairs, 1):
        labels.text(.01, y, f'{y:02d}  ' + pair_label(normal),
                    va='center', fontsize=10.6, color='#223548')
        xs = [normal['P_cold'] * 100, reversed_row['P_cold'] * 100]
        probability.plot(xs, [y, y], color='#b6c1cb', linewidth=1, zorder=1)
        for row, marker, color, x in [(normal, 'o', FORWARD, .22),
                                       (reversed_row, 'v', REVERSE, .76)]:
            probability.scatter(row['P_cold'] * 100, y, color=color,
                                marker=marker, s=53, zorder=4)
            norms.text(x, y, f"{row['actual_delta_norm']:.4f}",
                       ha='center', va='center', fontsize=11.5, fontfamily='monospace')
            if row['id'] == best['id']:
                probability.annotate(f"Best control: {row['P_cold'] * 100:.3g}%",
                                     (row['P_cold'] * 100, y), xytext=(8, -20),
                                     textcoords='offset points', fontsize=9.8, color=FORWARD)
    fig.suptitle('L2-matched active MOLT pairs versus the selected small → hot pair',
                 fontsize=20, weight='bold', x=.025, ha='left', y=.977)
    fig.text(.025, .934,
             f"13 active alternative pairs × 2 sign directions = 26 controls; all injected at layer 22, token 9; attention unfrozen.\n"
             f"Each control’s combined vector is rescaled to intended L2 = {reference['intended_delta_norm']:.4f}; right-hand columns show the actual BF16 write.",
             fontsize=12.5, linespacing=1.6, va='top')
    fig.legend(handles=[Line2D([], [], marker='o', color=FORWARD, ls='', label='−receiver +donor'),
                        Line2D([], [], marker='v', color=REVERSE, ls='', label='+receiver −donor'),
                        Line2D([], [], marker='*', color=REFERENCE, ls='', markersize=13, label='Selected pair, 4.5×')],
               loc='upper left', bbox_to_anchor=(.022, .895), ncol=3, frameon=False, fontsize=11.5)
    actual = [r['actual_delta_norm'] for pair in pairs for r in pair]
    error = max(abs(n / reference['actual_delta_norm'] - 1) for n in actual) * 100
    fig.text(.025, .055,
             f"Actual control norms: {min(actual):.4f}–{max(actual):.4f}; selected: {reference['actual_delta_norm']:.4f}; maximum relative difference: {error:.4f}%.\n"
             'C is the full gate × low-rank output × released output scaling. The net pair vector is norm-matched before BF16 rounding.\n'
             'Other-layer contributions are computed at their source layer, then injected at layer 22. Missing saved labels are marked explicitly.',
             fontsize=11, linespacing=1.55, color='#33485a')
    fig.savefig(root / 'l2-matched-pairs.png', dpi=180)
    fig.savefig(root / 'l2-matched-pairs.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def single_figure(root, controls, references):
    fig = plt.figure(figsize=(20, 10.8))
    outer = fig.add_gridspec(1, 2, left=.025, right=.985, top=.80, bottom=.13, wspace=.06)
    for column, (kind, source, title) in enumerate([
            ('positive', 'donor', 'Donor-only additions'),
            ('negative', 'receiver', 'Receiver-only subtractions')]):
        group = sorted([r for r in controls if r['kind'] == kind],
                       key=lambda r: (r['layer'] != 22, r['layer'], r['sample_index']))
        assert len(group) == 12
        reference = references[kind]
        all_rows = [reference] + group
        inner = outer[column].subgridspec(1, 3, width_ratios=[1.75, .85, .48], wspace=.025)
        labels, probability, norms = [fig.add_subplot(inner[0, i]) for i in range(3)]
        for ax in [labels, norms]:
            ax.set_xlim(0, 1); ax.set_ylim(len(all_rows) - .4, -.7); ax.axis('off')
        low = 10 ** (math.floor(math.log10(min(r['P_cold'] * 100 for r in all_rows))) - 1)
        axes_format(probability, len(all_rows), low)
        probability.set_xlabel('P(cold), % · log scale', fontsize=10)
        probability.tick_params(axis='x', labelsize=9)
        labels.set_title(title + f"\nIntended L2 = {reference['intended_delta_norm']:.4f}",
                         loc='left', fontsize=12, pad=20)
        norms.set_title('Actual L2', fontsize=11, pad=20)
        for y, row in enumerate(all_rows):
            for ax in [labels, probability, norms]:
                if y == 0 or y % 2 == 0:
                    ax.axhspan(y - .47, y + .47, color='#e6f2ee' if y == 0 else '#f5f7fa', zorder=0)
            label = ('SELECTED · ' if y == 0 else '') + feature_text(row, source)
            labels.text(.01, y, textwrap.fill(label, 49), fontsize=10.5,
                        va='center', weight='bold' if y == 0 else 'normal')
            probability.scatter(row['P_cold'] * 100, y,
                                marker='*' if y == 0 else 'o', s=150 if y == 0 else 40,
                                color=REFERENCE if y == 0 else FORWARD if kind == 'positive' else REVERSE, zorder=4)
            norms.text(.55, y, f"{row['actual_delta_norm']:.4f}", ha='center', va='center',
                       fontsize=10.7, fontfamily='monospace', weight='bold' if y == 0 else 'normal')
    fig.suptitle('Single-feature L2 controls · separate norm target for each one-term reference',
                 fontsize=18, weight='bold', x=.025, ha='left', y=.974)
    fig.text(.025, .92,
             '12 active donor controls + 12 active receiver controls. Each is matched to its selected single-feature edit at 4.5×.\n'
             'Every edit is applied at layer 22, token 9, with unfrozen attention; actual BF16 intervention norms are printed per trial.',
             fontsize=12.5, linespacing=1.6, va='top')
    fig.text(.025, .045,
             'These single-feature panels use their own matched norm budgets; they are not comparisons at the selected pair’s L2 = 65.3207.\n'
             'Probabilities are percentages of the full vocabulary. Labels are saved dashboard descriptions; unavailable labels are identified by transform ID.',
             fontsize=11, linespacing=1.6, color='#33485a')
    fig.savefig(root / 'l2-matched-singles.png', dpi=180)
    fig.savefig(root / 'l2-matched-singles.pdf', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    root = parser.parse_args().result_dir
    controls, references, evidence = load(root)
    pair_controls = [r for r in controls if r['kind'] in ['pair', 'pair_reverse']]
    pair_figure(root, pair_controls, references['pair'])
    single_figure(root, controls, references)
    reference = references['pair']
    actual = [r['actual_delta_norm'] for r in pair_controls]
    summary = {
        'source': 'Saved previously executed conditions; CPU-only presentation, no new inference',
        'control_conditions': len(controls), 'paired_control_conditions': len(pair_controls),
        'distinct_control_pairs': 13, 'single_control_conditions': 24,
        'edit_layer_zero_based': 22, 'token_position_zero_based': 9,
        'attention_frozen': False, 'selected_pair_P_cold': reference['P_cold'],
        'selected_pair_intended_L2': reference['intended_delta_norm'],
        'selected_pair_actual_L2': reference['actual_delta_norm'],
        'pair_controls_actual_L2_min': min(actual), 'pair_controls_actual_L2_max': max(actual),
        'max_actual_L2_relative_difference_percent': max(abs(x / reference['actual_delta_norm'] - 1) for x in actual) * 100,
        'paired_controls_max_P_cold': max(r['P_cold'] for r in pair_controls),
        'all_control_source_gates_positive': True, 'all_actual_control_edits_nonzero': True,
        'input_sha256': evidence,
    }
    ordered = [reference] + pair_controls + [references['positive']] + [r for r in controls if r['kind'] == 'positive'] + [references['negative']] + [r for r in controls if r['kind'] == 'negative']
    records = []
    for row in ordered:
        group = ('reference_pair' if row['id'] == reference['id'] else
                 'reference_single' if row['stratum'] == 'selected' else
                 'paired_control' if row['kind'] in ['pair', 'pair_reverse'] else 'single_control')
        records.append({k: row[k] for k in ['id', 'kind', 'layer', 'edit_layer', 'effective_terms',
                       'intended_delta_norm', 'actual_delta_norm', 'native_mlp_norm', 'P_cold', 'P_large']} |
                       {'group': group, 'P_cold_percent': row['P_cold'] * 100,
                        'norm_match_scale': row.get('norm_match_scale')})
    data = {'summary': summary, 'rows': records}
    (root / 'l2-matched-data.json').write_text(json.dumps(data, indent=2) + '\n')
    fields = ['id', 'group', 'kind', 'source_layer', 'edit_layer', 'features_and_effective_coefficients',
              'intended_L2', 'actual_L2', 'native_MLP_L2', 'P_cold_percent', 'P_large_percent', 'norm_match_scale']
    with (root / 'l2-matched-data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in records:
            recipe = ' ; '.join(f"{t['coefficient']:+.8g} × {t['source']} L{t['layer']}–T{t['id']} — {t['label']} [gate={t['source_gate']:.6f}]" for t in row['effective_terms'])
            writer.writerow(dict(zip(fields, [row['id'], row['group'], row['kind'], row['layer'],
                row['edit_layer'], recipe, row['intended_delta_norm'], row['actual_delta_norm'],
                row['native_mlp_norm'], row['P_cold_percent'], row['P_large'] * 100, row['norm_match_scale']])))
    esc = html.escape
    table_rows = []
    for row in records:
        terms = '<br>'.join(f"{t['coefficient']:+.6g} × {t['source']} <b>L{t['layer']}–T{t['id']}</b> — {esc(t['label'])} <small>Source gate {t['source_gate']:.6f}</small>" for t in row['effective_terms'])
        table_rows.append(f"<tr><td>{esc(row['group'])}<small>{esc(row['id'])}</small></td><td>{terms}</td><td>{row['intended_delta_norm']:.6f}</td><td><b>{row['actual_delta_norm']:.6f}</b></td><td>{row['P_cold_percent']:.8g}%</td></tr>")
    fragment = f'''{START}
<section id="norm-matched"><h2>L2-matched controls versus the selected pair</h2>
<p>The paired comparison holds the <b>combined intervention vector's intended L2 norm at {reference['intended_delta_norm']:.4f}</b>. There are <b>13 active alternative pairs, each tested in both sign directions (26 controls)</b>. All edits are applied at layer 22, receiver token 9, with unfrozen attention and the native MLP retained.</p>
<p><code>reference = −4.5 × C_small + 4.5 × C_hot<br>control = −C_receiver,A + C_donor,B<br>matched_control = control × L2(reference) / L2(control)</code></p>
<p>C is the complete gate × low-rank output × released output scaling. Matching the net vector includes cancellation between its two terms. Normalization is performed in FP32 before the original BF16 write. The right-hand columns in the figure show <b>L2(MLP_after − MLP_before)</b>, the actual applied change.</p>
<p>Selected pair: <b>{reference['P_cold'] * 100:.6f}% P(cold)</b>, actual L2 <b>{reference['actual_delta_norm']:.6f}</b>. Best paired control: <b>{summary['paired_controls_max_P_cold'] * 100:.8g}% P(cold)</b>. Actual paired-control norms range from {min(actual):.6f} to {max(actual):.6f}; the maximum relative difference from the selected actual norm is {summary['max_actual_L2_relative_difference_percent']:.5f}% after BF16 rounding.</p>
<a href="l2-matched-pairs.png"><img src="l2-matched-pairs.png" alt="All 13 active alternative MOLT pairs in both directions versus the selected pair, with feature names, next-token probabilities, and actual applied L2 norms"></a>
<p><a href="l2-matched-pairs.png">Full-size PNG</a> · <a href="l2-matched-pairs.pdf">PDF</a> · <a href="l2-matched-data.csv">All 50 controls and three references, CSV</a> · <a href="l2-matched-data.json">Structured data and source hashes</a> · <a href="plot_norm_matched.py">CPU-only figure source</a></p>
<p>The probability axis is logarithmic and displays percentages of the full vocabulary. Feature names are saved dashboard labels; absent labels are marked explicitly. Other-layer source vectors are computed at their original layer, then injected at layer 22. These are the previously executed controls, not new GPU runs.</p>
<details><summary>Single-feature norm-matched controls (24 conditions)</summary><p>The 12 donor additions are matched to selected donor-only L2 {references['positive']['intended_delta_norm']:.4f}; the 12 receiver subtractions are matched to selected receiver-only L2 {references['negative']['intended_delta_norm']:.4f}. These differ from the paired intervention's norm budget.</p><a href="l2-matched-singles.png"><img src="l2-matched-singles.png" alt="Individual active donor and receiver controls with feature names and actual norms, each matched to its single-feature reference"></a><p><a href="l2-matched-singles.png">Full-size PNG</a> · <a href="l2-matched-singles.pdf">PDF</a></p></details>
<details id="norm-matched-table"><summary>Every norm-matched control: features, coefficients and actual norms (53 rows including references)</summary><div class="scroll"><table><thead><tr><th>Condition</th><th>Features and actual contribution multipliers</th><th>Intended L2</th><th>Actual L2</th><th>P(cold)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></details>
</section>
{END}'''
    (root / 'norm-matched-section.html').write_text(fragment + '\n')
    page = (root / 'index.html').read_text()
    if START in page:
        page = re.sub(re.escape(START) + '.*?' + re.escape(END), lambda _: fragment, page, flags=re.S)
    else:
        marker = '<section><h2>Corrected scatterplot</h2>'
        assert page.count(marker) == 1
        page = page.replace(marker, fragment + '\n' + marker)
    if 'href="#norm-matched"' not in page:
        page = page.replace('<p><a href="#features">', '<p><a href="#norm-matched">L2-matched control comparison</a> · <a href="#features">', 1)
    (root / 'index.html').write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
