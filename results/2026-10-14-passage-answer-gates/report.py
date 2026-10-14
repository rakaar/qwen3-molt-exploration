"""Raw-gate heatmaps with a shared scale and complete per-token provenance."""
import argparse
import csv
import html
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
import numpy as np

def visible(row):return [t for t in row['answer_tokens'] if not t['special']]
def symbol(token):return token.replace(' ','·').replace('\n','↵').replace('\t','⇥')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    data=json.loads((root/'artifacts/answer-gates.json').read_text())
    assert data['status']=='complete' and len(data['rows'])==10
    rows=data['rows'];groups={g:[r for r in rows if r['group']==g] for g in ['unsupported_answer','abstention']}
    assert len(groups['unsupported_answer'])==4 and len(groups['abstention'])==6
    maximum=max(t['gate'] for r in rows for t in visible(r))
    norm=Normalize(vmin=0,vmax=maximum if maximum>0 else 1)
    cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#dfe4eb')
    summary={'layer':19,'transform':368,'label':data['protocol']['label'],
       'gate_threshold':data['threshold'],'shared_plot_range':[0,maximum],
       'all_saved_greedy_predictions_verified':all(r['saved_greedy_predictions_verified'] for r in rows),
       'all_final_prompt_gates_zero':all(r['final_prompt_token']['gate']==0 for r in rows),
       'groups':{},'cases':[]}
    for group,chosen in groups.items():
        tokens=[t for r in chosen for t in visible(r)]
        summary['groups'][group]={'cases':len(chosen),'cases_with_any_active_answer_token':sum(any(t['gate']>0 for t in visible(r)) for r in chosen),
           'visible_answer_tokens':len(tokens),'active_answer_tokens':sum(t['gate']>0 for t in tokens),
           'peak_gate':max(t['gate'] for t in tokens),'token_weighted_mean_gate':sum(t['gate'] for t in tokens)/len(tokens)}
    for r in rows:summary['cases'].append({'id':r['id'],'group':r['group'],**{k:v for k,v in r['summary'].items() if k!='strongest_tokens'}})
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    csvfields=['case_id','group','answer_position','full_chat_position','token_id','token','special','eos','raw','gate','lowrank_norm','contribution_norm','native_mlp_norm','contribution_over_native_mlp_norm']
    with (root/'token-values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csvfields,extrasaction='ignore');writer.writeheader()
        for r in rows:
            for t in r['answer_tokens']:writer.writerow({'case_id':r['id'],'group':r['group'],**t})
    ordered=groups['unsupported_answer']+groups['abstention']
    length=max(len(visible(r)) for r in rows)
    matrix=np.full((len(ordered),length),np.nan)
    for i,r in enumerate(ordered):matrix[i,:len(visible(r))]=[t['gate'] if t['gate'] else 0 for t in visible(r)]
    fig,ax=plt.subplots(figsize=(14.5,6.4))
    picture=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(10));ax.set_yticklabels([('Unsupported' if r['group']=='unsupported_answer' else 'Abstained')+' · '+r['id']+('*' if r['ancillary_explanation_error'] else '') for r in ordered],fontsize=10)
    ax.set_xticks(np.arange(0,length,5));ax.set_xlabel('Answer token position (zero-based; words differ between rows)',fontsize=11)
    ax.axhline(3.5,color='#27394b',linewidth=1.7)
    for i,r in enumerate(ordered):
        ax.text(length+.25,i,f"{r['summary']['active_tokens']}/{r['summary']['visible_tokens']} active",va='center',fontsize=9)
    ax.text(length+.25,-1,'Active tokens',fontsize=9,fontweight='bold')
    ax.set_xticks(np.arange(-.5,length,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.3);ax.tick_params(which='minor',bottom=False,left=False)
    colorbar=fig.colorbar(picture,ax=ax,pad=.115,fraction=.035);colorbar.set_label('L19–T368 gate activation (FP32)')
    fig.suptitle('L19–T368 · Onset of Refusal or Ignorance Statements',fontsize=17,fontweight='bold',y=.98)
    ax.set_title('Four unsupported answers vs six abstentions · Native Qwen3-4B · No intervention',fontsize=11,pad=22)
    fig.text(.02,.018,'Near-white = zero gate. Grey = no answer token. EOS excluded. *P2-U2 abstains but has an inaccurate explanation. Shared linear scale.',fontsize=9,color='#425466')
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(root/'answer-gate-overview.png',dpi=180);fig.savefig(root/'answer-gate-overview.pdf');plt.close(fig)
    def token_strips(group,filename,title):
        chosen=groups[group];columns=12;counts=[math.ceil(len(visible(r))/columns) for r in chosen]
        fig=plt.figure(figsize=(14.5,sum(.62*n+1.05 for n in counts)+1.1))
        axes=fig.subplots(len(chosen),1,gridspec_kw={'height_ratios':[n+.9 for n in counts]})
        axes=np.atleast_1d(axes)
        for ax,r,n in zip(axes,chosen,counts):
            tokens=visible(r);values=np.full((n,columns),np.nan)
            for i,t in enumerate(tokens):values[i//columns,i%columns]=t['gate'] if t['gate'] else 0
            ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
            for i,t in enumerate(tokens):
                x=i%columns;y=i//columns;color='white' if norm(t['gate'])>.6 else '#17283b'
                ax.text(x,y-.17,symbol(t['token']),ha='center',va='center',fontsize=9,fontfamily='monospace',color=color)
                ax.text(x,y+.19,f"{t['answer_position']} · g={t['gate'] if t['gate'] else 0:.2f}",ha='center',va='center',fontsize=7.6,color=color)
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_xticks(np.arange(-.5,columns,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1.2);ax.tick_params(which='minor',bottom=False,left=False)
            label=r['id']+('*' if r['ancillary_explanation_error'] else '')+' · '+r['question'].strip()
            ax.set_title(textwrap.fill(label,110)+f"\n{r['summary']['active_tokens']}/{len(tokens)} active answer tokens · max gate {r['summary']['max_gate'] if r['summary']['max_gate'] else 0:.3f}",loc='left',fontsize=10,pad=10)
        fig.suptitle(title+'\nL19–T368 · Onset of Refusal or Ignorance Statements',fontsize=16,fontweight='bold',y=.99)
        footnote=f'Shared linear gate scale: 0–{maximum:.3f}. · marks a leading space. Cell label: answer-token index · gate value. EOS excluded.'
        if group=='abstention':footnote+='\n*P2–U2 abstains but gives an inaccurate explanation; its saved abstention grouping is retained.'
        fig.text(.015,.014,footnote,fontsize=9)
        fig.tight_layout(rect=[0,.05,1,.95],h_pad=2.5)
        fig.savefig(root/filename,dpi=160);fig.savefig(root/filename.replace('.png','.pdf'));plt.close(fig)
    token_strips('unsupported_answer','unsupported-answer-token-heatmap.png','FOUR UNSUPPORTED ANSWERS — gate is zero throughout')
    token_strips('abstention','abstention-token-heatmap.png','SIX ABSTENTION DECISIONS — gate activates in every response')
    esc=html.escape
    def card(row):
        tokens=[]
        for t in visible(row):
            background=to_hex(cmap(norm(t['gate'])));foreground='white' if norm(t['gate'])>.6 else '#17283b'
            tooltip=f"Answer token {t['answer_position']}; chat token {t['full_chat_position']}; gate {t['gate']:.6f}; raw {t['raw']:.6f}; contribution norm {t['contribution_norm']:.6f}"
            tokens.append(f'<span class="token" style="background:{background};color:{foreground}" title="{esc(tooltip)}"><b>{esc(symbol(t["token"]))}</b><small>{t["answer_position"]} · {t["gate"] if t["gate"] else 0:.2f}</small></span>')
        note='<p class="note">This response abstains but incorrectly says the passage does not compare the black caiman’s size. Its abstention decision is the saved grouping.</p>' if row['ancillary_explanation_error'] else ''
        return f'<article><h3>{row["id"]} · {esc(row["question"].strip())}</h3><p class="answer">{esc(row["answer"])}</p>{note}<p class="stats">{row["summary"]["active_tokens"]}/{row["summary"]["visible_tokens"]} active tokens · peak gate {row["summary"]["max_gate"] if row["summary"]["max_gate"] else 0:.3f}</p><div class="tokens">'+''.join(tokens)+'</div></article>'
    casesrows=''.join(f'<tr><td>{r["id"]}</td><td>{"Unsupported answer" if r["group"]=="unsupported_answer" else "Abstention"}</td><td>{r["summary"]["active_tokens"]}/{r["summary"]["visible_tokens"]}</td><td>{r["summary"]["max_gate"] if r["summary"]["max_gate"] else 0:.4f}</td><td>{r["summary"]["mean_gate"]:.4f}</td></tr>' for r in ordered)
    groupsections=''.join('<section><h2>'+title+'</h2><p><a href="'+filename+'">Full PNG token heatmap</a></p>'+''.join(card(r) for r in groups[group])+'</section>' for group,title,filename in [('unsupported_answer','Four unsupported answers','unsupported-answer-token-heatmap.png'),('abstention','Six abstention decisions','abstention-token-heatmap.png')])
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>L19–T368 · Passage-answer gate heatmaps</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#17283b;font:16px/1.6 system-ui,sans-serif}main{max-width:1500px;padding:32px 24px;margin:auto}
    h1{font-size:33px;line-height:1.25}h2{font-size:25px}h3{font-size:17px;line-height:1.45}a{color:#145ba6}.eyebrow{font-size:13px;color:#426481;letter-spacing:.07em}
    .summary,.figure,section{background:white;border:1px solid #d3dfea;border-radius:12px;padding:24px;margin:24px 0}.counts{display:flex;gap:28px;flex-wrap:wrap}.counts strong{font-size:32px;display:block}
    .figure img{width:100%;height:auto}.groups{display:grid;grid-template-columns:1fr 1fr;gap:20px}.groups section{margin:0}article{border-top:1px solid #d9e3ed;margin-top:24px;padding-top:15px}
    .answer{white-space:pre-wrap}.tokens{display:flex;gap:4px;flex-wrap:wrap}.token{display:inline-flex;flex-direction:column;align-items:center;border:1px solid #d6dfe7;border-radius:4px;padding:7px 6px;min-width:43px;font:13px/1.5 monospace}
    .token b{font-weight:500}.token small{font-size:10px}.stats{font-size:13px;color:#506475}.note{border-left:3px solid #e5a339;padding-left:12px;font-size:14px}
    table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dae4ed}th{background:#edf3fa}.scroll{overflow:auto}.legend{height:16px;max-width:380px;background:linear-gradient(to right,#ffffcc,#fed976,#fd8d3c,#e31a1c,#800026);border-radius:4px}code{overflow-wrap:anywhere;font-size:13px}.fine{font-size:13px;color:#546577}
    @media(max-width:800px){main{padding:18px 12px}h1{font-size:27px}.groups{grid-template-columns:1fr}.summary,.figure,section{padding:16px}}
    </style></head><body><main><div class="eyebrow">14 OCTOBER 2026 · OBSERVATION ONLY · NATIVE QWEN3-4B</div>
    <h1>L19–T368 on unsupported answers and abstentions</h1><p>Dashboard label: <b>Onset of Refusal or Ignorance Statements</b>. The same 10 saved responses to unanswerable passage questions were replayed without changing the model. Every saved greedy next-token prediction was verified.</p>
    <div class="summary"><div class="counts"><div><strong>0/4</strong>Unsupported answers with any active answer token</div><div><strong>6/6</strong>Abstentions with active answer tokens</div></div>
    <p>The four unsupported answers have zero gate throughout. The six abstentions have peak gates from 2.941 to 3.481, often on <code> not</code>, <code> cannot</code>, and <code> determine</code>.
    P2–U5 contains “do not actually spread the flu” but its gate remains zero, so presence of “not” alone does not explain this sample.</p>
    <p>This is an association in 10 selected answers. It does not establish that the transform causes abstention. One abstention has an inaccurate explanation; its earlier label is retained and flagged below.</p></div>
    <div class="figure"><img src="answer-gate-overview.png" alt="Gate heatmap: zero throughout four unsupported answers, activation in six abstentions"><p><a href="answer-gate-overview.png">Open overview PNG</a> · <a href="answer-gate-overview.pdf">PDF</a></p></div>
    <div class="summary"><h2>How to read the tokens</h2><div class="legend"></div><p>All heatmaps use the same linear colour scale, 0–'''+f'{maximum:.3f}'+'''. The plotted value is the scalar JumpReLU gate, not the low-rank output or contribution norm.
    Near-white means zero gate. Grey cells in PNG figures are padding, not zero activation. EOS is excluded from figures and response summaries but retained in raw data. A dot marks a leading space; token indices are zero-based. Hover over a token for exact gate and contribution values.</p>
    <p>The gate is measured <b>while processing the displayed token</b>; changing it could affect subsequent predictions, not the token already emitted. The final prompt token, which predicts the first answer token, had gate zero in all 10 cases.</p></div>
    <div class="groups">'''+groupsections+'''</div>
    <section><h2>Per-answer measurements</h2><div class="scroll"><table><thead><tr><th>Case</th><th>Saved grouping</th><th>Active / visible tokens</th><th>Peak gate</th><th>Mean gate across visible tokens</th></tr></thead><tbody>'''+casesrows+'''</tbody></table></div>
    <p><a href="token-values.csv">Every token value (CSV)</a> · <a href="artifacts/answer-gates.json">Raw measurement JSON</a> · <a href="summary.json">Summary</a> · <a href="protocol.json">Protocol</a> · <a href="run.py">Measurement code</a> · <a href="report.py">Figure code</a></p></section>
    <section><h2>Setup and what remains to test</h2><p>Native BF16 Qwen3-4B; unfrozen eager attention; thinking disabled. Gate and low-rank contribution math use FP32 at the input to layer-19 post-attention layernorm, with released input standardization and output scaling. No MLP replacement, gate suppression, amplification, or attention editing was performed.</p>
    <p>The next causal test can suppress the transform on the six active abstention responses. The four already-unsupported responses provide zero-gate controls on their clean trajectories. Changing future output tokens could change later gate activity, so each edited trajectory must recompute its own contribution.</p>
    <p>Base revision: <code>'''+esc(data['protocol']['base_revision'])+'''</code><br>MOLT revision: <code>'''+esc(data['protocol']['molt_revision'])+'''</code><br>Selected asset SHA256: <code>'''+esc(data['protocol']['asset_sha256'])+'''</code></p>
    <p><a href="../2026-10-14-passage-grounding-baseline/index.html">Original passages, questions, complete answers, and grading</a> · <a href="baseline-cases.json">Saved passages and questions</a> · <a href="baseline-assessment.json">Fixed earlier judgments</a></p>
    <p class="fine">Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0. <a href="https://rajpurkar.github.io/SQuAD-explorer/">Source</a>. Unsupported means unsupported by the supplied passage, not necessarily false outside it. GPU retained as requested.</p></section></main></body></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps(summary['groups'],indent=2))

if __name__=='__main__':main()
