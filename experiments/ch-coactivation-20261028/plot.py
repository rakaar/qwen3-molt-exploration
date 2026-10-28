"""Static scientific figure from independently audited summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/2026-10-28-ch-coactivation'


def main():
    validation = json.loads((OUT / 'validation.json').read_text())
    assert validation['status'] == 'passed'
    data = json.loads((OUT / 'ablation-summary.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False})
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(data))
    groups = [('native', 'Native', '#626b76', -0.24),
              ('L3_off', 'L3 off', '#3375b4', -0.08),
              ('L29_off', 'L29 off', '#d38927', 0.08),
              ('both_off', 'Both off', '#277954', 0.24)]
    for key, label, color, offset in groups:
        values = [100 * (row['native_p_answer_slot'] if key == 'native'
                         else row[key]['p_answer_slot']) for row in data]
        ax.scatter(values, y + offset, s=35, color=color, label=label, zorder=3)
    labels = []
    for row in data:
        name = row['case_id'].replace('_', ' ')
        if row['native_answer'] == 'CE':
            name += ' (native answer CE)'
        labels.append(name)
    ax.set_yticks(y, labels)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.set_ylim(len(data) - 0.5, -0.6)
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('P(CH/ch) at the same native answer prefix (%)')
    ax.grid(axis='x', color='#e5e7eb', linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=4, frameon=False)
    fig.suptitle('CH gates coactivate on 9 of 48 CH prompts', x=.02, ha='left',
                 fontsize=15, fontweight='bold', y=.985)
    fig.text(.02, .925, 'Input-word contribution subtraction: each gate separately and both together.',
             fontsize=10, color='#4b5563')
    fig.text(.02, .018,
             'Qwen3-4B · L3–T2334 and L29–T2334 · 28 October 2026\n'
             'Selected coactive prompts only; conditional next-token mass. Joint L29 contribution is recomputed after the L3 edit.',
             fontsize=8.5, color='#4b5563')
    fig.subplots_adjust(left=.28, right=.975, top=.81, bottom=.13)
    for suffix in ['svg', 'png']:
        fig.savefig(OUT / f'probability-comparison.{suffix}', dpi=170, facecolor='white')
    plt.close(fig)
    print('Saved audited probability-comparison.svg and .png')


if __name__ == '__main__':
    main()
