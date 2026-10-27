"""Build the public CH-ablation report from explicitly selected, nonsecret fields."""
import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ORDER = ['L3-T2334', 'L4-T153', 'L29-T2334', 'L32-T1516', 'L34-T929']
LABELS = {
    'L3-T2334': ('Ch-Initial Word Onset', 32),
    'L4-T153': ('Substring “ch” or “new”', 256),
    'L29-T2334': ('Initial “Ch” Word Onset', 32),
    'L32-T1516': ('Culinary and Proper Noun “Ch” Word Starts', 32),
    'L34-T929': ('Bigram “ch” Prefix Gate', 64),
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def collect():
    l3 = json.loads((WORKSPACE / 'results/2026-10-28-ch-gate-off/summary.json').read_text())
    other = json.loads((WORKSPACE / 'results/2026-10-28-ch-individual-gates/summary.json').read_text())
    cases = json.loads((WORKSPACE / 'experiments/ch-gate-off-20261028/cases.json').read_text())
    by_case = {x['id']: x for x in cases}
    rows = []
    keys = ['case_id', 'word', 'letter_case', 'group', 'native_answer', 'gate_off_answer', 'answer_slot',
            'native_p_answer_slot', 'gate_off_p_answer_slot', 'answer_slot_change_pp',
            'native_p_first_output', 'gate_off_p_first_output', 'target_tokens', 'natural_gates',
            'actual_edit_norms', 'zero_control_exact']
    for original in l3 + other:
        row = {key: original[key] for key in keys}
        row['feature'] = original.get('feature', 'L3-T2334')
        row['active'] = any(g > 0 for g in row['natural_gates'])
        row['nonzero_edit'] = any(norm > 0 for norm in row['actual_edit_norms'])
        row['prompt'] = by_case[row['case_id']]['user']
        row['expected'] = by_case[row['case_id']]['expected']
        rows.append(row)
    ranks = {name: i for i, name in enumerate(ORDER)}
    case_order = {x['id']: i for i, x in enumerate(cases)}
    rows.sort(key=lambda r: (ranks[r['feature']], case_order[r['case_id']]))
    return {'date_local': '2026-10-28', 'rows': rows, 'cases': cases,
            'gates': [{'feature': name, 'label': LABELS[name][0], 'rank': LABELS[name][1]} for name in ORDER],
            'base_repo': 'Qwen/Qwen3-4B', 'base_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
            'molt_repo': 'georglange/qwen3-4b-molt', 'molt_revision': '081ce901b090b64aeecf9bc8709b57c10411e32a',
            'probability_definition': 'Sum of plain and one-leading-space CH/ch single-token variants; prompt-alone first output and fixed-native-prefix answer slot are separate metrics.',
            'intervention': 'One transform at a time; subtract the estimated naturally gated contribution at all input word tokens in its own layer. Native MLP remainder retained; no generated-token edits.',
            'validation': {
                'l3': json.loads((WORKSPACE / 'results/2026-10-28-ch-gate-off/validation.json').read_text()),
                'other_four': json.loads((WORKSPACE / 'results/2026-10-28-ch-individual-gates/validation.json').read_text())}}


def initial_rows(rows):
    result = []
    for r in rows:
        e = html.escape
        delta = 100 * (r['gate_off_p_answer_slot'] - r['native_p_answer_slot'])
        color = 'negative' if delta < 0 else 'positive' if delta > 0 else 'zero'
        status = 'active edit' if r['active'] else 'inactive · no-op'
        gates = ' · '.join(f"{t}: {v:.4f}" for t, v in zip(r['target_tokens'], r['natural_gates']))
        result.append(f'<tr data-feature="{r["feature"]}" data-case="{r["case_id"]}"><td><strong>{r["feature"]}</strong></td>'
                      f'<td><code>{e(r["word"])}</code><span class="prompt-label">{r["letter_case"]}</span></td>'
                      f'<td><span class="badge {"active" if r["active"] else "off"}">{status}</span><span class="gate-values">{e(gates)}</span></td>'
                      f'<td><code>{e(r["native_answer"])}</code></td><td><code>{e(r["gate_off_answer"])}</code></td>'
                      f'<td><code>{e(r["answer_slot"]["prefix"] or "prompt alone")}</code></td>'
                      f'<td>{100*r["native_p_answer_slot"]:.6f}</td><td>{100*r["gate_off_p_answer_slot"]:.6f}</td>'
                      f'<td class="{color}">{delta:+.6f}</td></tr>')
    return ''.join(result)


def plot(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.hashsalt': 'ch-ablations-20261028', 'svg.fonttype': 'none'})
    matrix = np.zeros((12, 5)); mask = np.ones((12, 5), dtype=bool)
    for column, gate in enumerate(data['gates']):
        for row, case in enumerate(data['cases']):
            value = next(x for x in data['rows'] if x['feature'] == gate['feature'] and x['case_id'] == case['id'])
            matrix[row, column] = value['answer_slot_change_pp']
            mask[row, column] = not value['active']
    cmap = LinearSegmentedColormap.from_list('confidence', ['#a4492c', '#fffdf7', '#24654d'])
    cmap.set_bad('#e7e8e0')
    fig, ax = plt.subplots(figsize=(12, 10.2), facecolor='#f8f8f3')
    im = ax.imshow(np.ma.masked_array(matrix, mask), cmap=cmap, norm=TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25), aspect='auto')
    for y in range(12):
        for x in range(5):
            text = 'off' if mask[y, x] else f'{matrix[y,x]:+.3g}'
            color = '#6c746e' if mask[y,x] else 'white' if abs(matrix[y,x]) > 15 else '#202c2b'
            ax.text(x, y, text, ha='center', va='center', fontsize=12, color=color)
    ax.set_xticks(range(5), [x['feature'].replace('-', '\n') for x in data['gates']], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(12), [f"{x['word']}  ·  {'UPPER' if x['letter_case']=='upper' else 'lower'}" for x in data['cases']], fontsize=11)
    ax.tick_params(length=0, pad=12)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax.grid(which='minor', color='#f8f8f3', linewidth=3); ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Change in CH/ch answer-slot probability', x=.07, y=.98, ha='left', fontsize=20, color='#202c2b')
    fig.text(.07, .94, 'Native → one-gate ablation · percentage points · identical native prefixes', fontsize=11, color='#5c6866')
    bar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.045, pad=.07, shrink=.75)
    bar.set_label('Lower probability  ←  change in percentage points  →  higher probability', fontsize=10)
    bar.ax.tick_params(labelsize=10); bar.outline.set_visible(False)
    fig.text(.07, .025, 'Gray “off” cells: gate inactive, so the intervention is an exact no-op.\n12 fixed prompts × 5 separate gates; no joint ablation. Every greedy response is unchanged.', fontsize=10, color='#5c6866')
    fig.subplots_adjust(left=.22, right=.97, top=.86, bottom=.12)
    fig.savefig(output, format='svg', metadata={'Date': None, 'Description': 'All 60 single-gate comparisons; inactive cells distinguished from active ablations.'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'results/2026-10-28-ch-gate-ablations')
    args = parser.parse_args()
    data = json.loads(args.data.read_text()) if args.data else collect()
    rows = data['rows']
    assert len(rows) == 60 and len(data['cases']) == 12 and len(data['gates']) == 5
    assert sum(x['active'] for x in rows) == sum(x['nonzero_edit'] for x in rows) == 15
    assert all(x['native_answer'] == x['gate_off_answer'] for x in rows)
    assert all(x['zero_control_exact'] for x in rows)
    assert all(x['native_answer'] in ('CH', 'ch', x['word'] + ': ch') for x in rows)
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    dump(output / 'data.json', data); dump(output / 'validation.json', data['validation'])
    columns = ['feature', 'case_id', 'word', 'letter_case', 'prompt', 'native_answer', 'gate_off_answer',
               'active', 'nonzero_edit', 'native_p_first_output', 'gate_off_p_first_output',
               'matched_prefix', 'native_p_answer_slot', 'gate_off_p_answer_slot', 'answer_slot_change_pp',
               'target_tokens', 'natural_gates', 'actual_edit_norms']
    with (output / 'comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in columns if key in row}
            record['matched_prefix'] = row['answer_slot']['prefix']
            for key in ['target_tokens', 'natural_gates', 'actual_edit_norms']:record[key] = json.dumps(record[key])
            writer.writerow(record)
    plot(data, output / 'probability-changes.svg')
    gate_rows = []
    for gate in data['gates']:
        values = [x for x in rows if x['feature'] == gate['feature']]
        active = sum(x['active'] for x in values)
        worst = min(values, key=lambda x: x['answer_slot_change_pp'])
        effect = f"{worst['case_id']}: {worst['answer_slot_change_pp']:+.6f} pp" if active else 'No-op on all 12 prompts'
        gate_rows.append(f'<tr><td>{gate["feature"]}<small>rank {gate["rank"]}</small></td><td>{html.escape(gate["label"])}</td><td>{active}</td><td>0</td><td>{effect}</td></tr>')
    prompts = ''.join(f'<article class="prompt-card"><h3>{x["id"]}</h3><pre>{html.escape(x["user"])}</pre></article>' for x in data['cases'])
    template = (HERE / 'template.html').read_text()
    replacements = {'__GATE_ROWS__': ''.join(gate_rows), '__INITIAL_ROWS__': initial_rows(rows),
                    '__GATE_OPTIONS__': ''.join(f'<option value="{x["feature"]}">{x["feature"]}</option>' for x in data['gates']),
                    '__PROMPTS__': prompts, '__DATA_JSON__': json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')}
    for token, value in replacements.items():template = template.replace(token, value)
    assert '__DATA_JSON__' not in template
    (output / 'index.html').write_text(template)
    for file in ['style.css', 'report.js']:shutil.copy2(HERE / file, output / file)
    files = [{'file':str(p.relative_to(output)), 'bytes':p.stat().st_size,
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'publication-manifest.json']
    dump(output / 'publication-manifest.json', {'data_rows':60, 'unique_prompts':12, 'active_ablations':15, 'inactive_noops':45, 'files':files})
    print(json.dumps({'output':str(output), 'comparisons':len(rows), 'active':15, 'files':len(files) + 1}))


if __name__ == '__main__':
    main()
