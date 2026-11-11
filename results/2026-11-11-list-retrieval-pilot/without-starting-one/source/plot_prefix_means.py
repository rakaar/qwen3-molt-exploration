"""Plot per-case mean gate conditional on positive answer-prefix tokens."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot'
RAW = OUT / 'without-starting-one/results.json'
result = json.loads(RAW.read_text())
assert result['status'] == 'complete' and result['interventions'] == 0
assert len(result['cases']) == 30

rows = []
for case in result['cases']:
    assert 'starting from 1' not in case['prompt']
    tokens = [t for t in case['trace'] if t['region'] == 'answer_prefix']
    active = [t for t in tokens if t['gate'] > 0]
    locations = {'after_im_end_gate': 0.0, 'after_think_gate': 0.0}
    for t in active:
        before = ''.join(x['token'] for x in case['trace'][:t['position']])
        if before.endswith('<|im_end|>'):
            assert t['token'] == '\n'
            locations['after_im_end_gate'] = t['gate']
        elif before.endswith('</think>'):
            assert t['token'] == '\n\n'
            locations['after_think_gate'] = t['gate']
        else:
            raise AssertionError('Unexpected active prefix location: ' + case['id'])
    rows.append({
        'case_id': case['id'], 'list_id': case['list_id'], 'format': case['format'], 'ordinal': case['ordinal'],
        'prefix_token_count': len(tokens), 'positive_token_count': len(active),
        'positive_gate_sum': sum(t['gate'] for t in active),
        'conditional_mean_gate': sum(t['gate'] for t in active) / len(active) if active else None,
        **locations, 'expected': case['expected'], 'response': case['native']['response'],
    })
groups = []
for fmt in ['numbered', 'bullets']:
    for n in [5, 11, 12]:
        subset = [r for r in rows if r['format'] == fmt and r['ordinal'] == n]
        groups.append({
            'format': fmt, 'ordinal': n, 'cases': len(subset),
            'active_cases': sum(r['positive_token_count'] > 0 for r in subset),
            'positive_tokens': sum(r['positive_token_count'] for r in subset),
            'after_im_end_active_cases': sum(r['after_im_end_gate'] > 0 for r in subset),
            'after_think_active_cases': sum(r['after_think_gate'] > 0 for r in subset),
        })
summary = {
    'feature': {'layer': 29, 'transform': 2375}, 'condition': 'starting from 1 removed',
    'region': 'answer_prefix', 'definition': 'Within each case: sum(gate for gate > 0) / count(gate > 0), only among chat answer-prefix tokens.',
    'no_positive_tokens': 'Conditional mean is undefined (JSON null); show a cross in a separate inactive strip, never as a numerical zero mean.',
    'exclusions': 'List-reading, user-question, generated-response and leading-wrapper tokens are excluded.',
    'unit': 'One dot per word order, list format and requested ordinal; token averages are per case, not pooled across cases.',
    'gate_threshold': result['runtime']['threshold'],
    'source_results_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
    'groups': groups, 'cases': rows,
}
(OUT / 'prefix-active-means.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
with (OUT / 'prefix-active-means.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 14, 'svg.hashsalt': 't2375-prefix-positive-means-20261111',
    'figure.facecolor': '#f7f6ef', 'axes.facecolor': '#fffef9', 'text.color': '#203d35',
    'axes.labelcolor': '#203d35', 'xtick.color': '#4d6257', 'ytick.color': '#4d6257',
})
colors = ['#2166a6', '#c27616', '#2b8865', '#a8497b', '#7762ac']
offsets = [-0.20, -0.10, 0, 0.10, 0.20]
fig = plt.figure(figsize=(12.8, 7.4))
grid = fig.add_gridspec(2, 2, height_ratios=[5, 0.85], left=0.13, right=0.97, bottom=0.21, top=0.77, hspace=0.09, wspace=0.14)
for col, fmt in enumerate(['numbered', 'bullets']):
    ax = fig.add_subplot(grid[0, col])
    inactive = fig.add_subplot(grid[1, col], sharex=ax)
    ax.set_title('Numbered lists' if fmt == 'numbered' else 'Bullet lists', loc='left', pad=17, fontweight='bold')
    ax.set_xlim(-0.43, 2.43)
    ax.set_ylim(0.45, 1.80)
    ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    ax.grid(axis='y', color='#d8dfd4', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    if col == 0:
        ax.set_ylabel('Mean gate φ(x) over active\nanswer-prefix tokens', labelpad=12)
        inactive.set_ylabel('Inactive\n(mean undefined)', rotation=0, ha='right', va='center', labelpad=12, fontsize=10)
    else:
        ax.tick_params(axis='y', labelleft=False)
    for spine in ax.spines.values():
        spine.set_color('#d8dfd4')
    inactive.set_facecolor('#eceee5')
    inactive.set_ylim(-0.5, 0.5)
    inactive.set_yticks([])
    inactive.set_xticks([0, 1, 2], ['5th word', '11th word', '12th word'])
    inactive.tick_params(axis='x', length=0, pad=10)
    for spine in inactive.spines.values():
        spine.set_color('#d8dfd4')
    for x, n in enumerate([5, 11, 12]):
        group = next(g for g in groups if g['format'] == fmt and g['ordinal'] == n)
        ax.text(x, 1.735, f"{group['active_cases']}/5 active", ha='center', va='center', fontsize=10, color='#5d6d64', bbox={'facecolor':'#fffef9', 'edgecolor':'none', 'pad':2})
        for i in range(5):
            row = next(r for r in rows if r['format'] == fmt and r['ordinal'] == n and r['list_id'] == f'list-{i+1}')
            if row['conditional_mean_gate'] is not None:
                ax.scatter(x + offsets[i], row['conditional_mean_gate'], s=80, c=colors[i], edgecolors='#fffef9', linewidths=0.9, zorder=3)
            else:
                inactive.scatter(x + offsets[i], 0, marker='x', s=46, c=colors[i], linewidths=1.8)
fig.suptitle('T2375 activity in the chat answer prefix', x=0.13, y=0.965, ha='left', fontsize=21, fontweight='bold')
fig.text(0.13, 0.916, 'Revised prompts: “starting from 1” removed · native model · no gate edits', fontsize=11, color='#5d6d64')
legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markeredgecolor='none', markersize=7, label=f'List {i+1}') for i, color in enumerate(colors)]
fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(0.123, 0.887), ncol=5, frameon=False, handletextpad=0.4, columnspacing=2.1)
fig.text(0.13, 0.103, 'One dot per list: average only its positive gate values in the answer prefix. Zeros are excluded.', fontsize=10)
fig.text(0.13, 0.070, '× in the separate strip = no active prefix tokens; no conditional mean exists. Horizontal offsets separate the five lists.', fontsize=10, color='#5d6d64')
fig.text(0.13, 0.035, 'List-reading tokens are excluded, including the numbered-list newline before printed 11.', fontsize=10, color='#5d6d64')
fig.savefig(OUT / 'prefix-active-means.svg', metadata={'Date': None})
fig.savefig(OUT / 'prefix-active-means.png', dpi=180, metadata={'Software': 'Matplotlib'})
plt.close(fig)
print(json.dumps({'cases': len(rows), 'active_cases': sum(r['positive_token_count'] > 0 for r in rows), 'groups': groups}))
