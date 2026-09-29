import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/molt-matplotlib')
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).parent / 'results/dallas-molt-gates'
r = json.loads((OUT / 'gates.json').read_text())
rows = [(l['layer'], f) for l in r['layers'] for f in l['active_transforms']]
positions = list(range(18, 28))
data = np.array([[f['gate_by_position'][p] for p in positions] for _, f in rows])
fig, ax = plt.subplots(figsize=(11.8, 6.0))
im = ax.imshow(data, cmap='Purples', vmin=0, vmax=2, aspect='auto')
ax.set_xticks(range(10), [r['tokens'][p]['text'].strip() for p in positions])
ax.set_yticks(range(len(rows)), [f'Layer {layer} · #{f["transform_id"]}' for layer, f in rows])
ax.tick_params(length=0, pad=9)
ax.set_xticks(np.arange(-.5,10,1), minor=True)
ax.set_yticks(np.arange(-.5,len(rows),1), minor=True)
ax.grid(which='minor', color='#e5dfee', linewidth=.7)
ax.tick_params(which='minor', length=0)
for y, values in enumerate(data):
    for x, v in enumerate(values):
        if v > 0:
            ax.text(x,y,f'{v:.2f}',ha='center',va='center',fontsize=9,
                    color='white' if v>1.2 else '#382050')
ax.add_patch(Rectangle((7.5,-.5),1,len(rows),fill=False,edgecolor='#e49100',lw=2))
ax.axhline(4.5,color='white',lw=4)
for s in ax.spines.values():s.set_visible(False)
fig.suptitle('MOLT transforms active at “Dallas”', x=.025,ha='left',y=.98,fontsize=18,fontweight='bold')
fig.text(.025,.92,'The same transforms across the question · original Qwen3-4B inputs · no model replacement',fontsize=11)
fig.colorbar(im,ax=ax,fraction=.03,pad=.025,label='Gate activation')
fig.text(.025,.035,'Rows include only transforms active at Dallas in cached layers 17 and 25. Blank = zero. IDs and layers are zero-based.\n'
         'The outlined column is Dallas. Gate strength is not causal importance; transforms have not been semantically labelled.',fontsize=9,color='#435260')
fig.subplots_adjust(left=.17,right=.92,top=.86,bottom=.17)
fig.savefig(OUT/'dallas-molt-gates.png',dpi=180,facecolor='white')
fig.savefig(OUT/'dallas-molt-gates.svg',facecolor='white')
lines=['# MOLT gates at Dallas', '', 'Question: '+r['question'], '',
       'Original Qwen3-4B inputs. No MOLT or transcoder replacement and no intervention. '
       'Both cached MOLT layers evaluated with saved normalization. Primary results use FP32 MOLT arithmetic on BF16 original-model inputs.', '',
       '| Layer | Transform | Rank | Gate at Dallas | Raw output contribution L2 | Peak gate token (whole prompt) |',
       '|---|---|---|---|---|---|']
for layer,f in rows:
    p=f['peak_gate_position']
    lines.append(f"| {layer} | {f['transform_id']} | {f['rank']} | {f['gate']:.4f} | {f['raw_output_l2']:.4f} | `{r['tokens'][p]['text']}` (position {p}) |")
lines += ['', '![Gate activations](dallas-molt-gates.png)', '',
          'Five transforms activate at Dallas in layer 17, four in layer 25, out of 2,480 per layer. '
          'BF16 and FP32 agree on the active transform IDs at Dallas. '
          'Summing the individual standardized output contributions reproduces Georg’s unchanged MOLT forward result '
          'with maximum absolute errors of 1.61e-6 (layer 17) and 1.55e-6 (layer 25).', '',
          'Output contribution L2 is the vector norm after multiplying by the saved output standard deviation, '
          'excluding the shared output mean. It is not attribution to Austin, and vector norms are not additive because contributions can cancel.', '',
          'The most active gate need not be the most selective for Dallas or the most causally important. '
          'These transform IDs have no semantic labels established by this experiment.', '',
          'Only layers 17 and 25 were checked. No new MOLT weights were downloaded. '
          'The earlier 8.44 GB two-layer download took 508.55 seconds; future transfer times can vary.', '',
          'Full per-token gates for these transforms: [gates.json](gates.json).']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
