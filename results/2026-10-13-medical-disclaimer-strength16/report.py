import csv,html,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'artifacts/extension-trials.json').read_text())
audit=json.loads((ROOT/'local-verification.json').read_text());assert audit['status']=='passed'
assessment=json.loads((ROOT/'assessment.json').read_text())
protocol=json.loads((ROOT/'protocol.json').read_text())
selection=json.loads((ROOT/'selection.json').read_text())
e=html.escape
ratios=[]
for r in rows:
 maximum=max((a['intended_delta_norm']/max(a['original_mlp_norm'],1e-8) for a in r['edited_positions']),default=0)
 ratios.append({'id':r['id'],'prompt_id':r['prompt_id'],'factor':r['factor'],'maximum_intended_correction_over_original_mlp_norm':maximum,'edited_positions':len(r['edited_positions']),'stop_reason':r['stop_reason'],'generation':r['generation']})
with (ROOT/'matched-results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(ratios[0]));writer.writeheader();writer.writerows(ratios)
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for ax,pid in zip(axes.flat,selection['prompt_ids']):
 rs=[r for r in ratios if r['prompt_id']==pid]
 ax.bar([str(r['factor']) for r in rs],[r['maximum_intended_correction_over_original_mlp_norm'] for r in rs],color=['#8dada3','#55a48b','#27866f','#ac593a'])
 ax.set_title(pid.replace('_',' '));ax.set_xlabel('Estimated component coefficient factor')
 ax.set_ylabel('Largest correction norm / MLP norm');ax.grid(axis='y',alpha=.2)
 ax.spines[['top','right']].set_visible(False)
 if all(r['maximum_intended_correction_over_original_mlp_norm']==0 for r in rs):
  ax.set_ylim(0,1)
  ax.text(.5,.5,'Gate = 0\nNo edits',transform=ax.transAxes,ha='center',va='center',color='#557265')
fig.suptitle('L19–T368: Onset of Refusal or Ignorance Statements\nMeasured intervention size at each tested factor',fontsize=14)
fig.savefig(ROOT/'intervention-size.png',dpi=160);fig.savefig(ROOT/'intervention-size.svg');plt.close(fig)
sections=[]
timing_rows=[]
for pid in selection['prompt_ids']:
 rs=[r for r in rows if r['prompt_id']==pid]
 clean=next(r for r in rs if r['factor']==1)
 variant=''
 for r in rs:
  peak=r['activation_summary']['maximum_gate'];sites=[a['position'] for a in r['edited_positions']]
  note=assessment['conditions'].get(r['id'],'')
  variant+=f"<h3>Factor {r['factor']}</h3><p>{e(note)}</p><pre>{e(r['generation'])}</pre><p class='meta'>Peak raw gate {peak:.6f}; edited full-chat positions {e(str(sites))}; stop: {e(r['stop_reason'])}.</p>"
 first=next((a for a in clean['activations'] if a['target_gate']>0),None)
 timing='Inactive throughout' if first is None else f"Position {first['position']}, token {first['token']!r}, gate {first['target_gate']:.6f}"
 negative=next(r for r in rs if r['factor']==-16)
 timing_rows.append(f"<tr><td>{e(pid.replace('_',' '))}</td><td>{e(timing)}</td><td>{e(assessment['conditions'][negative['id']])}</td></tr>")
 sections.append(f"<section><h2>{e(pid.replace('_',' '))}</h2><blockquote>{e(clean['prompt'])}</blockquote><p class='meta'>Each prompt includes: {e(protocol['format_suffix'])}</p>{variant}</section>")
resource='GPU retained for follow-up tests; resource verification pending.'
if (ROOT/'resource-state.json').exists():resource=json.loads((ROOT/'resource-state.json').read_text())['summary']
page="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L19–T368 negative factor 16 follow-up</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f2f4ef;color:#243b32;margin:0}main{max-width:1100px;margin:auto;padding:30px 22px}h1{font-size:34px;line-height:1.2}section{background:white;border:1px solid #d9e2d8;border-radius:12px;padding:24px;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3ee;padding:16px;border-radius:8px;font:15px/1.6 ui-monospace,monospace}a{color:#087765}.meta{font-size:13px;color:#557265}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:12px 8px;border-bottom:1px solid #d9e2d8}blockquote{margin:12px 0;border-left:4px solid #23816c;padding:12px}img{width:100%;max-width:1100px}</style></head><body><main><p class="meta">13 October 2026 · SPAR RTX 3090 · Qwen3-4B · unfrozen attention</p><h1>Does stronger negative steering remove the medical limitation?</h1>"""
page+=f"<section><h2>Result</h2><p>{e(assessment['conclusion'])}</p><p>{e(assessment['interpretation'])}</p><p>24 matched intervention generations and 24 clean identity generations across the original six selected prompts. All clean identity checks pass. The copied intervention code has exactly the same SHA-256 as the original pilot.</p></section>"
page+='<section><h2>Where the edit can first act</h2><p>Target: L19–T368, labeled “Onset of Refusal or Ignorance Statements.” The label describes gate activation examples and does not establish a causal refusal mechanism. These positions refer to the clean run. An edit at a generated token affects later tokens; it cannot undo that token.</p><table><thead><tr><th>Prompt</th><th>First active clean position</th><th>Factor −16 outcome</th></tr></thead><tbody>'+''.join(timing_rows)+'</tbody></table></section>'
page+='<section><h2>Strength definitions and comparison</h2><pre>original MLP output[token] += (factor − 1) × gate(current x) × lowrank(current x)\n\nfactor 1: unchanged\nfactor 0: subtract one contribution\nfactor −8: subtract nine contributions\nfactor −16: subtract seventeen contributions</pre><p>The original MLP stays in use. There is no donor. The layer-19 gate and low-rank output are recomputed on the current pre-normalization residual, using the released standardization and output scaling. Negative factors reverse the coefficient of an estimated MOLT component; they do not change a native gate in the original model.</p><p>Each batch contains four identical prompts, with one independent row for each factor. A second four-row clean batch verifies the unchanged reference. Positions are zero-based; full token traces, norms, prompt templates and outputs are preserved.</p><p>The original −8 endpoint was a pilot cutoff, not a demonstrated maximum or a search for the minimum successful strength. This follow-up tests −16 and makes no claim about every larger strength.</p></section>'
page+='<section><h2>Magnitude of the correction</h2><img src="intervention-size.png" alt="Maximum per-token correction norm divided by original MLP norm for six prompts"><p>Values show the largest recorded intended correction norm divided by the original MLP output norm on each evolving run. This is not a clinical-quality or safety score.</p></section>'
page+=''.join(sections)
page+=f"<section><h2>Artifacts and resources</h2><p>{e(resource)}</p><p><a href='matched-results.csv'>Matched results</a> · <a href='artifacts/extension-trials.json'>Full token and gate traces</a> · <a href='assessment.json'>Manual output assessment</a> · <a href='local-verification.json'>Independent checks</a> · <a href='protocol.json'>Protocol</a> · <a href='extension.py'>Extension source</a> · <a href='run.py'>Original intervention source</a> · <a href='../2026-10-13-medical-disclaimer/index.html'>Original pilot</a></p><p>The report assesses visible limitations and fulfillment of the requested task. It does not evaluate clinical correctness. Read the complete generations above.</p></section></main></body></html>"
(ROOT/'index.html').write_text(page)
print(assessment['conclusion'])
