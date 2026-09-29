"""Render the saved comparison; no GPU or network calls."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).parent/'results/small-hot-molt-gates-21-22'
d=json.loads((OUT/'comparison.json').read_text())
conditions=['baseline','suppression_only','addition_only','combined','natural_hot']
labels=['Original small prompt','Suppress small only','Add hot only','Suppress small + add hot','Natural hot prompt']

fig,axes=plt.subplots(1,2,figsize=(11,5.5),sharex=True,layout='constrained')
for ax,l in zip(axes,d['layers']):
    ids=sorted({r['id'] for c in ['baseline','combined','natural_hot'] for r in l['conditions'][c]['active_by_token'][9]})
    y=np.arange(len(ids))
    for off,c,label,color in [(-.24,'baseline','Original small','#64748b'),(0,'combined','Intervened small','#2563eb'),(.24,'natural_hot','Natural hot','#ea580c')]:
        values={r['id']:r['gate'] for r in l['conditions'][c]['active_by_token'][9]}
        ax.barh(y+off,[values.get(i,0) for i in ids],height=.23,label=label,color=color)
    ax.set(yticks=y,yticklabels=[f'#{i}' for i in ids],xlabel='Gate activation',title=f"Layer {l['layer']} — operand token, position 9")
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('MOLT transform ID (local to each layer)')
axes[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,-.16),ncol=3,fontsize=10)
fig.suptitle('MOLT gates change after the transcoder intervention (strength 4.5)',fontsize=14)
fig.savefig(OUT/'operand-gates.png',dpi=180);fig.savefig(OUT/'operand-gates.svg');plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
for ax,l in zip(axes,d['layers']):
    for c,label,color in [('baseline','Original small','#64748b'),('combined','Intervened small','#2563eb')]:
        ax.plot(range(29),l['conditions'][c]['counts_by_token'],marker='o',ms=4,label=label,color=color)
    ax.axvline(9,ls=':',color='#ea580c');ax.set(ylabel='Active gates',title=f"Layer {l['layer']}")
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(frameon=False)
axes[-1].set_xticks(range(29))
axes[-1].set_xticklabels([f"{t['position']}: {t['text']!r}" for t in d['tokens']],rotation=70,ha='right',fontsize=8)
fig.suptitle('Gate counts across all prompt tokens; dotted line marks “small”')
fig.savefig(OUT/'counts-by-token.png',dpi=180);fig.savefig(OUT/'counts-by-token.svg');plt.close(fig)

with (OUT/'counts-by-token.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['layer','position','recipient_token','condition','active_gate_count'])
    for l in d['layers']:
        for t in d['tokens']:
            for c in conditions:
                w.writerow([l['layer'],t['position'],t['text'],c,l['conditions'][c]['counts_by_token'][t['position']]])
rows=[]
for c,label in zip(conditions,labels):
    counts=[str(l['conditions'][c]['counts_by_token'][pos]) for pos in [9,28] for l in d['layers']]
    r=d['readouts'][c]
    rows.append('| '+ ' | '.join([label,*counts,r['top_token'],f"{100*r['P_cold']:.6g}%"])+' |')
report=['# MOLT gate comparison at layers 21 and 22','',
    'Completed on the existing RTX 5090. At the `small` token, the combined intervention changes layer 21 from 4 to 5 active gates and layer 22 from 3 to 5. The identities change as well as the counts. All indices are zero-based.','',
    'Recipient prompt: `'+d['recipient_prompt']+'`','',
    'Donor prompt: `'+d['donor_prompt']+'`','',
    'Strength is 4.5, with the same published feature selections as the successful reproduction. The recipient text never changes for the intervention conditions. Natural-hot is a separate untreated donor prompt.','',
    '## Gate counts and model readouts','',
    'Each layer has 2,480 possible gates. Position 9 is `small` in the recipient and `hot` in the donor; position 28 is the final chat-formatted prompt token, where the next answer token is predicted.','',
    '| Condition | L21 at operand | L22 at operand | L21 at final position | L22 at final position | Top next token | P(cold) |',
    '|---|---:|---:|---:|---:|---|---:|',*rows,'',
    '![Gates at operand](operand-gates.png)','',
    '## Gates at the small token: baseline versus combined intervention','']
for l in d['layers']:
    c=l['comparisons']['combined'][9]
    report.extend([f"### Layer {l['layer']}",'',
        f"Turned on: {c['turned_on']}. Turned off: {c['turned_off']}. Shared active gates: {c['shared']}.",'',
        '| Transform ID | Baseline gate | Intervened gate | Change |','|---|---:|---:|---:|'])
    for r in sorted(c['gate_changes'],key=lambda r:r['id']):
        report.append(f"| {r['id']} | {r['before']:.5f} | {r['after']:.5f} | {r['delta']:+.5f} |")
    report.append('')
report.extend(['At the final prompt position, the combined intervention preserves all active gate identities in both layers (7 at layer 21; 5 at layer 22), although their strengths change.','',
    'In layer 22, gates #71 and #941 are off for the original small prompt but on for both the combined intervention and the natural hot prompt. The combined set contains extra gates #40 and #61 beyond the natural-hot set. This is a candidate correspondence, not evidence that these gates mean hot or cause the answer flip.','',
    '![Counts across tokens](counts-by-token.png)','',
    '## What inputs were compared','',
    'We captured residual activations before post_attention_layernorm, matching Georg’s pre_norm MOLT checkpoint configuration. The original MLP itself receives the subsequent normalized activations. We applied the saved dimensionwise mean/std, encoder weights and bias, and Georg’s unchanged JumpReLU implementation. A gate is active exactly when its preactivation exceeds both its learned threshold and zero.','',
    'MOLTs were evaluated as passive observers; they did not replace MLPs in the model run. The existing intervention protocol freezes attention patterns across all layers and pins MLP outputs through layer 22 to original outputs plus decoder edits. Inputs at layer 21 reflect earlier-layer interventions; inputs at layer 22 additionally reflect the layer 21 edit. Neither reflects its own layer’s output edit.','',
    'Only gate behavior was requested and measured here. We did not download U/V transform matrices, reconstruct MLP outputs, or establish MOLT causal mediation or Jacobian fidelity. At strength 4.5, suppression uses signed decoder edits beyond zero; this is not simply turning features off.','',
    '## Verification and provenance','',
    '- The combined intervention reproduces the previous P(cold) exactly: 0.9977401494979858.',
    '- The zero-delta control preserves logits and both layers’ captured inputs exactly.',
    '- Removing interventions restores logits and captured inputs exactly.',
    '- Primary gate computations use FP32 on BF16 model activations. BF16 autocast yields identical active sets at operand position 9 and final position 28 for every measured condition.',
    '- Across all other positions, two marginal activity disagreements occur: one in layer 21 suppression-only and one in layer 22 combined. Full per-token sensitivity counts are in comparison.json.',
    f"- Downloaded {d['download_bytes']:,} bytes ({d['download_bytes']/1e6:.2f} MB), comprising gate tensors, headers, and configs for both layers. Full checkpoints were not downloaded.",
    '- Source range lengths and Content-Range headers were checked; extracted tensors are pinned to the checkpoint revision and hashed individually. These hashes document the downloaded subset; they are not a full-checkpoint hash verification.',
    f"- MOLT repository: georglange/qwen3-4b-molt, revision `{d['molt_revision']}`.",
    '- Qwen model cached revision: `1cfa9a7208912126459214e8b04321603b3df60c`; transcoder revision: `94d176260ac39ce2f882b8b09aba8c118df29bb3`.',
    '- Published intervention implementation: wdk0082/llm-circuits at `af8a37ad836dc401a0ec5dbb71307d8d7ab68028`.',
    f"- Runtime: {d['runtime']}.",'',
    'Artifacts: [full results](comparison.json), [all-token count table](counts-by-token.csv), captured-prenorm-inputs.pt, all-gates.pt, and gate-weights-manifest.json.'])
(OUT/'report.md').write_text('\n'.join(report)+'\n')
print('Saved report, two figures, and all-token CSV.')
