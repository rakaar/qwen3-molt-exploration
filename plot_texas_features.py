import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/molt-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / 'results/texas-features'
r = json.loads((OUT / 'activations.json').read_text())
features = [f for f in r['features'] if f['activations'][26] > 0]
positions = list(range(18, 28))
data = np.array([[f['activations'][p] for p in positions] for f in features])
fig, ax = plt.subplots(figsize=(11.8, 5.5))
im = ax.imshow(data, cmap='Blues', vmin=0, vmax=4.5, aspect='auto')
ax.set_xticks(range(len(positions)), [r['tokens'][p]['text'].strip() for p in positions])
ax.set_yticks(range(len(features)), [f"Layer {f['layer']:>2}  ·  #{f['index']}" for f in features])
ax.tick_params(axis='both', length=0, pad=9)
ax.set_xticks(np.arange(-.5, len(positions), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(features), 1), minor=True)
ax.grid(which='minor', color='#dbe3ed', linewidth=.7)
ax.tick_params(which='minor', length=0)
for y, row in enumerate(data):
    for x, value in enumerate(row):
        if value > 0:
            ax.text(x, y, f'{value:.2f}', ha='center', va='center',
                    color='white' if value > 2.6 else '#17344b', fontsize=11)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.suptitle('Texas-labelled features activate at “Dallas”', x=.03, ha='left', y=.98,
             fontsize=18, fontweight='bold')
fig.text(.03, .915, 'Original Qwen3-4B · same one-word question · before generating “Austin”', fontsize=11)
fig.colorbar(im, ax=ax, fraction=.03, pad=.025, label='Raw feature activation')
fig.text(.03, .04, '8 of 73 tested candidates activate on Dallas. Blank = zero. Layers are zero-based.\n'
                  'One additional candidate activates only on the initial chat marker (not shown). Activation is not causal importance.',
         fontsize=9, color='#435260')
fig.subplots_adjust(left=.20, right=.92, top=.85, bottom=.19)
fig.savefig(OUT / 'texas-on-dallas.png', dpi=180, facecolor='white')
fig.savefig(OUT / 'texas-on-dallas.svg', facecolor='white')

lines = ['# Texas-labelled transcoder features on the Dallas prompt', '',
         'Question: What is the capital of the state containing Dallas?', '',
         'System instruction: Answer with exactly one word. Do not explain.', '',
         'The original Qwen3-4B predicts Austin. No MOLT replacement, transcoder replacement, or intervention was applied.', '',
         '## Result', '',
         '73 unique exact Texas labels were selected from the first 80 semantic-search results across all 36 layers. '
         'Search pagination returned duplicates; this candidate set covers 25 layers and is not exhaustive. '
         'Eight candidates activate at Dallas (full prompt token index 26), and nowhere else in the 37-token prompt. '
         'A ninth candidate activates only at the initial chat marker. The other 64 candidates are zero everywhere.', '',
         '| Layer (zero-based) | Feature | Active token | FP32 activation |', '|---|---|---|---|']
for f in r['features']:
    if f['active_positions']:
        for p in f['active_positions']:
            lines.append(f"| {f['layer']} | [#{f['index']}]({f['url']}) | `{r['tokens'][p]['text']}` (position {p}) | {f['activations'][p]:.6f} |")
lines += ['', '![Question-token heatmap](texas-on-dallas.png)', '', '## Measurement details', '',
          '- Release: mwhanna/qwen3-4b-transcoders at revision ' + r['transcoder_revision'] + '.',
          '- Input hook: each original MLP input, after post_attention_layernorm, matching mlp.hook_in.',
          '- Encoder: ReLU(W_enc x + b_enc), matching circuit-tracer SingleLayerTranscoder.encode. '
          'These ReLU features are independent, so only selected encoder rows and biases are needed.',
          '- Original model and checkpoint tensors: BF16; primary encoder arithmetic: FP32. '
          'The BF16 encoder check produced exactly the same active/inactive pattern for every tested feature and token.',
          '- Captured prompt token IDs exactly match the prior one-word Dallas test; the next predicted token is still Austin.',
          '- All 37 positions were evaluated, including system and assistant-format tokens. The figure shows only question tokens. '
          'The generated answer was not fed into this measurement.',
          '- Approximately 10.2 MB fetched, including file headers and full small bias arrays. No full transcoder layer downloaded.',
          '- Labels are automated Gemini 2.0 Flash descriptions. They are candidates for interpretation, not verified causal functions.',
          '- A positive activation at Dallas does not establish a Dallas → Texas → Austin causal pathway. '
          'No attribution, ablation, intervention, or Jacobian test was performed here.', '',
          'Full activations, including all zero candidates and BF16 checks: [activations.json](activations.json).', '']
(OUT / 'report.md').write_text('\n'.join(lines))
print('Created report and figure')
