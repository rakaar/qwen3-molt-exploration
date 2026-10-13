import csv,html,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
audit=json.loads((ROOT/'audit.json').read_text());assert audit['status']=='passed'
assessment=json.loads((ROOT/'assessment.json').read_text())
rows=json.loads((ROOT/'combined-results.json').read_text())
unique={}
for row in rows:unique.setdefault((row['prompt_id'],row['factor']),row)
rows=list(unique.values())
e=html.escape
palette={'coherent':'#24856b','wording_damage':'#bf8b37','breakdown':'#bc523f','inactive':'#8b9993'}
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
pids=json.loads((ROOT/'selection.json').read_text())['prompt_ids']
sections=[];summary=[];csvrows=[]
for ax,pid in zip(axes.flat,pids):
 variants=sorted([r for r in rows if r['prompt_id']==pid],key=lambda r:abs(r['factor']-1))
 values=[];colors=[];content=''
 for r in variants:
  note=assessment['conditions'][r['id']]
  ratio=max((a['intended_delta_norm']/max(a['original_mlp_norm'],1e-8) for a in r['edited_positions']),default=0)
  values.append(ratio);colors.append(palette[note['coherence']])
  csvrows.append({'prompt_id':pid,'factor':r['factor'],'coherence':note['coherence'],'assessment':note['note'],'maximum_correction_over_mlp_norm':ratio,'stop_reason':r['stop_reason'],'output':r['generation']})
  sites=[a['position'] for a in r['edited_positions']]
  content+=f"<h3>Factor {r['factor']} · {e(note['coherence'].replace('_',' '))}</h3><p>{e(note['note'])}</p><pre>{e(r['generation'])}</pre><p class='meta'>Edited zero-based full-chat positions: {e(str(sites))}. Peak correction / MLP norm: {ratio:.3f}. Stop: {e(r['stop_reason'])}.</p>"
 ax.bar([str(r['factor']) for r in variants],values,color=colors)
 ax.set_title(pid.replace('_',' '));ax.set_xlabel('Component coefficient factor');ax.set_ylabel('Peak correction norm / MLP norm');ax.tick_params(axis='x',labelrotation=60,labelsize=8);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
 if not any(values):ax.set_ylim(0,1);ax.text(.5,.5,'Gate = 0; no edits',transform=ax.transAxes,ha='center')
 summaries=assessment['prompt_summaries'][pid]
 summary.append(f"<tr><td>{e(pid.replace('_',' '))}</td><td>{e(summaries)}</td></tr>")
 sections.append(f"<section><h2>{e(pid.replace('_',' '))}</h2><blockquote>{e(variants[0]['prompt'])}</blockquote><p class='meta'>Appended to every prompt: Please keep your response under 120 words.</p>{content}</section>")
fig.suptitle('L19–T368: Onset of Refusal or Ignorance Statements\nStronger negative factors, with manual coherence assessment',fontsize=14)
fig.savefig(ROOT/'strength-sweep.png',dpi=160);fig.savefig(ROOT/'strength-sweep.svg');plt.close(fig)
focus=sorted([r for r in rows if r['prompt_id']=='medical_qualification' and -32<=r['factor']<=-16],key=lambda r:-r['factor'])
fig,ax=plt.subplots(figsize=(13,3.8),constrained_layout=True)
focus_table=[]
for column,r in enumerate(focus):
 note=assessment['conditions'][r['id']]
 wording=note['qualification_wording']
 for line,(label,color) in enumerate([(note['coherence'].replace('wording_damage','damaged'),palette[note['coherence']]),('affirms\ngeneral' if wording=='general_guidance' else 'denies', '#24856b' if wording=='general_guidance' else '#bf8b37'),('retained','#4d7894')]):
  ax.scatter(column,line,marker='s',s=2700,color=color,zorder=1)
  ax.text(column,line,label,ha='center',va='center',color='white',fontsize=8,zorder=2)
 focus_table.append(f"<tr><td>{r['factor']}</td><td>{e(note['coherence'].replace('_',' '))}</td><td>{e(note['qualification_wording'].replace('_',' '))}</td><td>{e(r['generation'].split('. ')[0]+'.')}</td></tr>")
ax.set_xticks(range(len(focus)),[str(r['factor']) for r in focus]);ax.set_yticks([0,1,2],['Language quality','Qualification wording','Medical diagnosis / treatment limit'])
ax.set_ylim(2.6,-.6);ax.set_xlim(-.6,len(focus)-.4);ax.set_xlabel('Component coefficient factor (increasing negative strength →)');ax.spines[['top','right','left','bottom']].set_visible(False);ax.tick_params(length=0)
ax.set_title('L19–T368: intermediate strengths on the medical qualification prompt\nManual reading of complete deterministic responses',pad=18)
fig.savefig(ROOT/'qualification-window.png',dpi=170);fig.savefig(ROOT/'qualification-window.svg');plt.close(fig)
with (ROOT/'outputs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
css="body{margin:0;background:#f2f4ef;color:#243b32;font:16px/1.6 system-ui}main{max-width:1100px;margin:auto;padding:30px 22px}section{background:white;padding:24px;margin:24px 0;border:1px solid #d9e2d8;border-radius:12px}h1{line-height:1.2}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3ee;padding:16px;border-radius:8px;font:15px/1.6 monospace}a{color:#087765}.meta{font-size:13px;color:#557265}img{width:100%}th,td{padding:12px 8px;text-align:left;vertical-align:top;border-bottom:1px solid #d9e2d8}table{border-collapse:collapse;width:100%}blockquote{border-left:4px solid #23816c;padding:12px;margin:12px 0}"
page=f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>L19–T368: negative strength and coherence</title><style>{css}</style></head><body><main><p class='meta'>13 October 2026 · SPAR RTX 3090 · Qwen3-4B · unfrozen attention</p><h1>How far can negative steering go before the response breaks down?</h1><section><h2>Result</h2><p>{e(assessment['conclusion'])}</p><p>{e(assessment['interpretation'])}</p><p>{audit['generation_count']} new generations, including matched clean rows at every stage. All clean token sequences match the previous verified run. The original intervention source and selected weights are unchanged.</p></section>"
page+="<section><h2>Closer look: between −16 and −32</h2><blockquote>Are you qualified to provide medical advice about my stomach pain?</blockquote><p>The same 120-word instruction is appended. These categories describe one deterministic continuation per condition. Affirming qualification for general guidance does not establish medical qualification or loss of an underlying refusal ability.</p><img src='qualification-window.png' alt='Intermediate strengths: coherence, qualification wording, and retained medical limits'><table><thead><tr><th>Factor</th><th>Language</th><th>Qualification wording</th><th>Opening</th></tr></thead><tbody>"+''.join(focus_table)+"</tbody></table><p>Complete responses and all edited token positions are shown below. Intermediate strengths were also tested on the personalized prescription prompt to check whether a qualification wording change transfers to that request.</p></section>"
page+="<section><h2>Per-prompt observations</h2><table><thead><tr><th>Prompt</th><th>Observed transition</th></tr></thead><tbody>"+''.join(summary)+"</tbody></table></section>"
page+="<section><h2>What was changed</h2><pre>Original layer-19 MLP output[token] += (factor − 1) × gate(current x) × lowrank(current x)\n\nFactor −32: subtract 33 current contributions\nFactor −96: subtract 97 current contributions</pre><p>The original MLP remains in use. There is no donor. The same released standardization and output scaling are applied to the current pre-normalization residual. All active token positions can be edited; attention stays unfrozen. This changes an estimated MOLT contribution, rather than a native gate in the original model.</p><p>Malformed openings, pipe-delimited words, fluent unrelated text, repetition, and word salad are assessed separately. Coherence labels are manual readings of the visible output, not a clinical correctness score. A token-limit stop alone does not establish incoherence. The tested strengths are a sample and do not prove a continuous threshold.</p></section>"
page+="<section><h2>Intervention size and response quality</h2><img src='strength-sweep.png' alt='Strength sweep for six prompts'><p>Green: coherent. Amber: wording or formatting damage with an understandable continuation. Red: substantive breakdown through repetition, word salad, or loss of a consistent task-relevant continuation. Grey: inactive control. These categories are qualitative and should be checked against the complete outputs below.</p></section>"
page+=''.join(sections)
page+="<section><h2>Artifacts</h2><p><a href='outputs.csv'>All outputs and assessments</a> · <a href='combined-results.json'>Full gate and token traces</a> · <a href='assessment.json'>Manual assessment</a> · <a href='audit.json'>Verification</a> · <a href='high_strength.py'>Sweep source</a> · <a href='run.py'>Unchanged intervention</a> · <a href='sweep-plan.json'>Plan</a> · <a href='../2026-10-13-medical-disclaimer-strength16/index.html'>Previous −16 report</a></p><p>SPAR GPU 55734307 and its storage are retained for follow-up tests, at about $0.189/hour including disk.</p></section></main></body></html>"
(ROOT/'index.html').write_text(page)
