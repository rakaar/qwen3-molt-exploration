"""Render audited input-token gate observations, with one generated-token comparison."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

audit = json.loads(Path(sys.argv[1]).read_text())
assert audit['status'] == 'pass'
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)
cases = audit['cases']
with_ablation = len(sys.argv) > 3
if with_ablation:
    ablation = json.loads(Path(sys.argv[3]).read_text())
    assert ablation['status'] == 'pass'
    cases.insert(2, {'prompt': ablation['prompt'], 'input_tokens': ablation['input_tokens'],
                     'generated_tokens_observed': ablation['generated_tokens_observed'],
                     'gate_off': True, 'edited_position': ablation['edited_position']})
maximum = max(t['natural_gate'] for c in cases for t in c['input_tokens'])
columns = max(len(c['input_tokens']) for c in cases)
fig, axes = plt.subplots(len(cases), 1, figsize=(11.5, 6.4 + 1.65 * (len(cases) - 3)))
fig.patch.set_facecolor('white')
title = 'L29–T2375: natural activation and gate-off' if with_ablation else 'Where L29–T2375 naturally activates'
subtitle = ('Raw text completion · native Qwen3-4B · red outline marks the only edited token'
            if with_ablation else 'Raw text completion · native Qwen3-4B · no activation changes')
fig.suptitle(title, x=.05, y=.98, ha='left', fontsize=18, weight='bold')
fig.text(.05, .936 if with_ablation else .923, subtitle, fontsize=11, color='#475569')
for ax, case in zip(axes, cases):
    ax.set_xlim(-.1, columns + 1.45)
    ax.set_ylim(-.18, 1.56)
    ax.axis('off')
    prompt_label = case['prompt'] + ('  [gate off at final is]' if case.get('gate_off') else '')
    ax.text(0, 1.34, prompt_label, fontfamily='DejaVu Sans Mono', fontsize=11.5)
    display = [(i, t, False) for i, t in enumerate(case['input_tokens'])]
    if case['generated_tokens_observed']:
        display.append((columns + .35, case['generated_tokens_observed'][0], True))
    for x, token, generated in display:
        edited = case.get('gate_off') and not generated and token['position'] == case['edited_position']
        value = 0. if edited else token['natural_gate']
        fill = '#f3f4f6' if value == 0 else plt.cm.Oranges(.30 + .55 * value / maximum)
        border = '#dc2626' if edited else ('#2563eb' if generated else '#cbd5e1')
        ax.add_patch(Rectangle((x, .30), .94, .77, facecolor=fill, edgecolor=border, linewidth=2 if edited else 1.1))
        label = token['token'].replace(' ', '·').replace('\n', '\\n')
        ax.text(x + .47, .84, label, ha='center', va='center', fontfamily='DejaVu Sans Mono', fontsize=9.6)
        ax.text(x + .47, .50, f'{value:.3f}', ha='center', va='center', fontsize=10,
                weight='bold' if value else 'normal')
        ax.text(x + .47, .13, str(token['position']), ha='center', fontsize=8, color='#64748b')
        if generated:
            ax.text(x + .47, 1.14, 'generated', ha='center', fontsize=8.5, color='#2563eb')
        elif edited:
            ax.text(x + .47, 1.14, f"OFF ({token['natural_gate']:.3f} → 0)", ha='center', fontsize=8.5, color='#dc2626')
fig.text(.05, .042, 'Numbers inside cells: gate activation. Small numbers below: zero-based token position. · denotes a space.', fontsize=9.3, color='#475569')
fig.text(.05, .018, 'The generated-token cell is measured when that token is processed on the next decoding step.', fontsize=9.3, color='#475569')
fig.subplots_adjust(left=.05, right=.985, top=.91 if with_ablation else .88, bottom=.09 if with_ablation else .11, hspace=.23)
filename = 'gate-off-token-comparison' if with_ablation else 'natural-gate-tokens'
for extension in ['png', 'svg']:
    fig.savefig(destination / f'{filename}.{extension}', dpi=180)
print(destination / f'{filename}.png')
