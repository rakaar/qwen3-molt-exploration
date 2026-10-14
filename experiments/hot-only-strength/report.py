"""Render measured P(cold) curves and an auditable static research report."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / 'artifacts/results.json').read_text())
V = json.loads((ROOT / 'verification.json').read_text())
G = json.loads((ROOT / 'artifacts/generations.json').read_text())
assert R['status'] == 'complete' and V['status'] == 'passed'
colors = {0: '#00788a', .5: '#d86d22', 4.5: '#8b49a1'}
labels = {0: 'Hot only: no small subtraction', .5: 'Subtract 0.5× small + add hot', 4.5: 'Subtract 4.5× small + add hot'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
for beta in (0, .5, 4.5):
    rows = sorted((r for r in R['rows'] if r['group'] == 'coarse' and r['beta'] == beta), key=lambda r: r['alpha'])
    for ax in axes:
        ax.plot([r['alpha'] for r in rows], [100*r['P_cold'] for r in rows], '-o', ms=4, color=colors[beta], label=labels[beta])
for ax in axes:
    ax.axhline(99, color='#555', ls='--', lw=1, label='99% threshold')
    ax.set_ylim(-2, 102)
    ax.set_xlabel('Added donor hot contribution strength, alpha')
    ax.set_ylabel('P(cold), % of full vocabulary')
    ax.grid(alpha=.18)
axes[0].set_xlim(0, 32)
axes[0].set_title('Strengths 0–32')
axes[1].set_xscale('symlog', linthresh=4.5)
axes[1].set_xlim(0, 530)
axes[1].set_title('Full tested range 0–512 (symlog x-axis)')
axes[0].legend(loc='best', fontsize=9)
fig.suptitle('Qwen3-4B · layer 22 · prompt token 9 (“small”) · attention unfrozen\nMLP output += −beta × receiver L22–T2339 + alpha × donor L22–T941', fontsize=14)
fig.savefig(ROOT/'hot-strength-sweep.png', dpi=180)
fig.savefig(ROOT/'hot-strength-sweep.pdf')
plt.close(fig)

small = sorted((r for r in R['rows'] if r['group'] == 'small_only'), key=lambda r:r['beta'])
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.plot([r['beta'] for r in small],[100*r['P_cold'] for r in small],'-o',color='#535d83')
ax.axhline(99,ls='--',color='#555')
ax.set_xscale('symlog',linthresh=4.5); ax.set_ylim(-2,102); ax.grid(alpha=.2)
ax.set_xlabel('Subtracted clean small contribution strength, beta');ax.set_ylabel('P(cold), % of full vocabulary')
ax.set_title('Small-only control: no hot donor contribution\nMLP output += −beta × receiver L22–T2339')
fig.savefig(ROOT/'small-only-control.png',dpi=170);plt.close(fig)

fields=['id','group','alpha','beta','P_cold','top_token','actual_delta_norm','actual_delta_over_native_mlp_norm','logit_row','vector_row']
with (ROOT/'conditions.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(R['rows'])
summary={'families':V['families'],'threshold_brackets':R['threshold_brackets'],
         'reference_pair':next(r for r in R['rows'] if r['id']=='coarse_a4.5_b4.5'),
         'best_small_only':max(small,key=lambda r:r['P_cold']),
         'conditions':len(R['rows']),'source_norms':R['source_norms'],'source_gates':R['source_gates'],
         'limitations':'One receiver/donor prompt pair; strength sweep, not broad task generalization. High-gain answer steering does not establish semantic replacement or feature selectivity. Local threshold refinement is not a proof of global minimum.'}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
esc=html.escape
def percent(p):return f'{100*p:.6g}%'
table=''
for beta in (0,.5,4.5):
    f=V['families'][str(beta)]
    threshold='Not reached through 512×' if f['lowest_tested_99_alpha'] is None else f"{f['lowest_tested_99_alpha']:.6g}×"
    table+=f"<tr><td>{labels[beta]}</td><td>{percent(f['maximum_coarse_P_cold'])}</td><td>{f['maximum_coarse_at_alpha']:g}×</td><td>{threshold}</td></tr>"
details=''.join(f"<tr><td>{esc(r['id'])}</td><td>{r['alpha']:g}</td><td>{r['beta']:g}</td><td>{percent(r['P_cold'])}</td><td>{esc(r['top_token'])}</td><td>{r['actual_delta_norm']:.4f}</td></tr>" for r in R['rows'])
generations=''.join(f"<article><b>alpha = {g['alpha']:g}; beta = {g['beta']:g}; P(cold) = {percent(g['first_token']['P_cold'])}</b><pre>{esc(g['text'])}</pre><small>{'EOS reached' if g['ended_with_eos'] else '24-token limit reached'}; initial logits match sweep exactly.</small></article>" for g in G)
brackets=''.join(f'<li>{esc(json.dumps(b))}</li>' for b in R['threshold_brackets'])
tokens=''.join(f"<span class=\"{'target' if t['position']==9 else 'token'}\"><small>{t['position']}</small>{esc(t['text'])}</span>" for t in R['receiver_tokens'])
reference=summary['reference_pair']
hotonly=V['families']['0']
headline='Hot-only steering reaches 99% P(cold)' if hotonly['lowest_tested_99_alpha'] is not None else 'Hot-only steering did not reach 99% in the tested range'
page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hot-only MOLT strength sweep</title><style>
body{{margin:0;background:#f4f6f7;color:#152331;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:32px 20px}}h1{{font-size:clamp(26px,4vw,40px);line-height:1.15}}h2{{margin-top:32px}}article,.card{{background:white;padding:20px;border:1px solid #d9e1e5;border-radius:10px;margin:16px 0}}img{{width:100%;height:auto;background:white}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f4;padding:15px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #d8e1e6}}.scroll{{overflow-x:auto}}.target{{background:#ffe280;border:2px solid #947200}}.token,.target{{display:inline-flex;flex-direction:column;white-space:pre-wrap;padding:4px;margin:2px;border-radius:4px}}small{{color:#556673}}a{{color:#006775}}.badge{{color:#006775;font-weight:700}}</style></head><body><main>
<p class="badge">2026-10-14 · Qwen3-4B · layer 22 · token 9 · unfrozen attention</p><h1>{headline}</h1>
<p>This tests whether the original two-feature result requires both edits at every strength. P(cold) is the full-vocabulary next-token probability of lowercase <code>cold</code> (token 87072).</p>
<div class="card"><b>Receiver:</b><pre>{esc(R['protocol']['receiver'])}</pre><b>Donor:</b><pre>{esc(R['protocol']['donor'])}</pre><p>The intervention writes at receiver “small”, token 9. The donor vector comes from “hot”, also token 9. Indices include the chat template and start at zero; readout is token 28.</p><div>{tokens}</div></div>
<h2>Measured strength curves</h2><a href="hot-strength-sweep.png"><img src="hot-strength-sweep.png" alt="Measured hot strength curves for three small subtraction settings"></a>
<div class="scroll"><table><thead><tr><th>Method</th><th>Best coarse-grid P(cold)</th><th>At hot strength</th><th>Lowest tested 99% strength</th></tr></thead><tbody>{table}</tbody></table></div>
<p>Original pair reference: subtract 4.5× small and add 4.5× hot gives <b>{percent(reference['P_cold'])}</b>. Values at 4.5× alone can be tiny without being exactly zero. The added strength and small subtraction are independently varied here.</p>
<h2>Precisely what changes</h2><pre>MLP_receiver[22, 9, :] += −beta × C_small(clean receiver)
                         +alpha × C_hot(clean donor)

C_f(x) = gate_f(z) × ((z @ V_f) @ U_f) × output_std
z = (prenorm residual x − input_mean) / input_std</pre>
<p><b>L22–T2339:</b> Small-Size Adjectives and Idiomatic Phrases; rank 32. <b>L22–T941:</b> Thermal State and Heat Action Gate; rank 64. These are dashboard gate labels, not proof that their outputs encode only those concepts.</p>
<p>The full gated low-rank output is patched. The source input, gate, and vector are computed once on each clean source prompt. We do not multiply x, replace a gate, or replace the MLP. The native MLP remains, and a fixed vector is added before residual addition. beta = 0.5 subtracts half the estimated clean small contribution; it does not set the gate to −0.5.</p>
<p>The model runs in BF16; MOLT source math is FP32. The FP32 delta is rounded to BF16, then added to native BF16 output. Actual writes, fixed sources, and lossless full-vocabulary BF16 logit bit patterns are saved for every condition.</p>
<h2>Single small-feature control</h2><img src="small-only-control.png" alt="Small-only strength control"><p>Best tested small-only P(cold): {percent(summary['best_small_only']['P_cold'])}, at beta = {summary['best_small_only']['beta']:g}.</p>
<h2>Greedy continuations</h2><p>Only the original prompt token 9 is directly edited. Later attention and all output-token computation remain free. Generation recomputes the full prefix, reapplies the same prompt-position edit, and stops at EOS or 24 generated tokens.</p>{generations}
<h2>Threshold resolution and limits</h2><ul>{brackets}</ul><p>{esc(summary['limitations'])} Probability versus strength may be nonmonotonic; a bounded sweep cannot establish impossibility at all strengths.</p>
<h2>Verification and provenance</h2><p>{len(R['rows'])} recorded conditions. Every batch includes unmodified receiver rows checked bitwise against baseline. Repeat and restoration checks pass. Native attention methods are unchanged; saved attention tensors are identical through layer 22 and differ downstream. An independent CPU audit recomputes source contribution math, full-vocabulary probabilities and BF16 writes.</p>
<p>Base revision: <code>{R['protocol']['base_revision']}</code><br>MOLT revision: <code>{R['protocol']['molt_revision']}</code><br>Selected asset SHA256: <code>{R['protocol']['selected_asset_sha256']}</code></p>
<p><a href="protocol.json">Protocol</a> · <a href="verification.json">CPU audit</a> · <a href="conditions.csv">CSV</a> · <a href="summary.json">Summary</a> · <a href="artifacts/results.json">Raw results</a> · <a href="artifacts/generations.json">Continuations</a> · <a href="hot-strength-sweep.pdf">Figure PDF</a> · <a href="../../experiments/hot-only-strength/run.py">Experiment source</a> · <a href="../../experiments/hot-only-strength/README.md">Reproduction</a></p>
<details><summary>Every recorded condition</summary><div class="scroll"><table><thead><tr><th>Condition</th><th>Hot alpha</th><th>Small beta</th><th>P(cold)</th><th>Top token</th><th>Actual delta norm</th></tr></thead><tbody>{details}</tbody></table></div></details>
</main></body></html>'''
(ROOT/'index.html').write_text(page)
print(json.dumps(summary,indent=2))
