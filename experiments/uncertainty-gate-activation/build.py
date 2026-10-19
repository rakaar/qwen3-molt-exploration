"""Measured query and response heatmaps for seven uncertainty candidates."""
import csv,html,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
S=Path(__file__).resolve().parent;W=S.parents[1];R=W/'results/2026-10-19-uncertainty-gate-activation'
D=json.loads((R/'artifacts/measurements.json').read_text());P=D['protocol'];F=P['features'];rows=D['rows'];h=lambda x:html.escape(str(x),quote=True)
assert D['status']=='complete' and len(rows)==14 and len(F)==7
keys=[f"L{x['layer']}-T{x['transform']}" for x in F];qa=[r for r in rows if r['group']!='positive_control']
scale=max(x['gate'] for r in rows for t in r['prompt_tokens']+r['answer_tokens'] for x in t['features'].values()) or 1.
cmap=colormaps['YlOrRd'];stats=[]
def colour(g):return '#ffffff' if g==0 else '#'+''.join(f'{round(c*255):02x}' for c in cmap(g/scale)[:3])
def tokens(ts,key):
 out=[]
 for t in ts:
  x=t['features'][key];g=x['gate'];title=f"{key} · {t['segment']} token {t['position']} · full position {t['full_chat_position']} · gate {g:.6f} · raw {x['raw']:.6f}"
  out.append(f'<span class="token" data-pos="{t["position"]}" data-gate="{g:.12g}" data-raw="{x["raw"]:.12g}" title="{h(title)}" style="background:{colour(g)};color:{"#fff" if g/scale>.62 else "#18282d"}">{h(t["token"])}</span>')
 return ''.join(out)
def attr(i):return f'data-feature-index="{i}"'+(' hidden' if i>=3 else '')
def heads():return '<div class="grid headings">'+''.join(f'<div {attr(i)}><a href="{h(f["dashboard_url"])}">{keys[i]}</a><span>{h(f["label"])}</span></div>' for i,f in enumerate(F))+'</div>'
groupnames={'correct':'Model answered correctly','made_up':'Model made up an answer','abstention':'Model said “I don’t know; it’s not in the passage”','positive_control':'Separate wording controls'}
def summary_table(segment):
 trs=[]
 for i,(f,key) in enumerate(zip(F,keys)):
  cells=[]
  for group in ['correct','made_up','abstention']:
   rs=[r['summary'][key] for r in qa if r['group']==group]
   meanfield,activefield={'response':('mean_gate','active_tokens'),'query':('query_mean_gate','query_active_tokens'),'last_prompt':('final_prompt_gate','final_prompt_gate')}[segment]
   mean=sum(r[meanfield] for r in rs)/4;count=sum(r[activefield]>0 for r in rs)
   cells.append(f'<td><b>{mean:.4f}</b><small>{count}/4 {"final positions active" if segment=="last_prompt" else "cases have active tokens"}</small></td>')
  trs.append(f'<tr><th>{key}<small>{h(f["label"])}</small></th>{"".join(cells)}</tr>')
 return '<div class="scroll"><table class="summary-table" data-segment="'+segment+'"><thead><tr><th>Candidate</th><th>Correct · 4</th><th>Made up · 4</th><th>Natural abstention · 4</th></tr></thead><tbody>'+''.join(trs)+'</tbody></table></div>'
sections=[]
for group in groupnames:
 xs=[r for r in rows if r['group']==group];cases=[]
 for r in xs:
  panels=[];inputpanels=[]
  for i,(f,key) in enumerate(zip(F,keys)):
   sm=r['summary'][key];ans=[t for t in r['answer_tokens'] if not t['special']]
   panels.append(f'<div class="panel answer-panel" data-feature="{key}" {attr(i)}><strong class="feature-id">{key}</strong><div class="query-map"><p class="block-label">Question tokens</p><p class="metrics">Mean <b>{sm["query_mean_gate"]:.4f}</b> · Active <b>{sm["query_active_tokens"]}/{sm["query_tokens"]}</b></p><div class="tokens query-tokens">{tokens(r["query_tokens"],key)}</div></div><p class="block-label">Native response tokens</p><p class="metrics">Mean <b>{sm["mean_gate"]:.4f}</b> · Peak <b>{sm["max_gate"]:.4f}</b> · Active <b>{sm["active_tokens"]}/{sm["visible_tokens"]}</b></p><div class="tokens response-tokens">{tokens(ans,key)}</div><p class="before">Before first answer prediction: gate <b>{sm["final_prompt_gate"]:.4f}</b><br><small>Last encoded prompt token; raw score {sm["final_prompt_raw"]:.4f}.</small></p></div>')
   inputpanels.append(f'<div class="panel prompt-panel" data-feature="{key}" {attr(i)}><strong class="feature-id">{key}</strong><div class="tokens">{tokens(r["prompt_tokens"],key)}</div></div>')
   stats.append({'id':r['id'],'group':r['group'],'feature':key,**sm})
  context=h(r.get('context','The wording control has no passage.'))
  cases.append(f'<article class="case" data-id="{r["id"]}" data-group="{group}" id="case-{r["id"]}"><h3><span>{r["id"]}</span> {h(r["question"])}</h3><div class="grid">{"".join(panels)}</div><details class="inputs"><summary>All input tokens: instructions, passage, question and chat markers</summary><p class="fine">The question-only maps above exclude the shared instructions and passage. Full positions are retained here.</p><div class="grid">{"".join(inputpanels)}</div></details><details class="passage"><summary>Passage and exact model input</summary><p>{context}</p><pre>{h(r["chat"])}</pre></details></article>')
 links='' if group=='positive_control' else f'<a href="{group}-priority-token-heatmaps.png">Priority-three PNG</a> · <a href="{group}-secondary-token-heatmaps.png">Other-four PNG</a>.'
 text='Two unsteered repetition tasks probe explicit information-gap and hedging wording. Excluded from the twelve QA cases and every group mean.' if group=='positive_control' else 'Four fixed unsteered examples; two from each original passage. '+links
 sections.append(f'<section class="cohort" id="{group}"><h2>{groupnames[group]}</h2><p class="fine">{text}</p>{heads()}{"".join(cases)}</section>')
response_active={k:sum(r['summary'][k]['active_tokens']>0 for r in qa) for k in keys};query_active={k:sum(r['summary'][k]['query_active_tokens']>0 for r in qa) for k in keys};last_active={k:sum(r['summary'][k]['final_prompt_gate']>0 for r in qa) for k in keys}
firstnotes=[]
for key in keys[:3]:
 counts=[sum(r['summary'][key]['active_tokens']>0 for r in qa if r['group']==g) for g in ['correct','made_up','abstention']]
 firstnotes.append(f'<li><b>{key}</b>: active in {counts[0]}/4 correct, {counts[1]}/4 made-up and {counts[2]}/4 natural-abstention responses.</li>')
findings='<ul>'+''.join(firstnotes)+'</ul>'
if all(query_active[k]==last_active[k]==0 for k in keys[:3]):findings+='<p><b>Before answering:</b> all three priority gates are zero on the question tokens and at the final input position in all twelve QA cases.</p>'
V=json.loads((R/'verification.json').read_text());failed_controls=[k for k,v in V['positive_response_control_per_gate'].items() if not v]
controlnote='All seven candidates activate in at least one control response.' if not failed_controls else 'No response activation in either wording control for '+', '.join(failed_controls)+'. These negative control results are preserved; a wording probe is not guaranteed to activate every candidate.'
nav=''.join(f'<a href="#{g}">{label}</a>' for g,label in [('correct','Correct answers'),('made_up','Made-up answers'),('abstention','Natural abstentions'),('positive_control','Controls')])
js="""document.querySelectorAll('[data-view-button]').forEach(b=>b.addEventListener('click',()=>{const mode=b.dataset.viewButton;document.body.dataset.view=mode;document.querySelectorAll('[data-feature-index]').forEach(x=>{const i=Number(x.dataset.featureIndex);x.hidden=mode==='first'?i>=3:mode==='secondary'?i<3:false});document.querySelectorAll('[data-view-button]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)))}));"""
page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Uncertainty gates · native 4/4/4 QA comparison</title><link rel="stylesheet" href="style.css"></head><body data-view="first"><main><p class="eyebrow">Qwen3–4B · no steering · 19 October 2026</p><h1>Uncertainty MOLTs on queries and unsteered answers</h1><p class="lead">The same four correct answers, four made-up answers and four natural abstentions. Seven label-selected candidates; the three priority gates are shown first. Question tokens and response tokens have separate maps and means.</p><nav>{nav}<a href="#method">Method</a></nav><div class="note"><b>Response activity in the priority candidates</b>{findings}<p class="fine">Activity while processing an emitted answer can reflect its wording. It is not by itself evidence of uncertainty before answering.</p></div><h2>Response-token means</h2>{summary_table('response')}<p class="fine">Each cell averages the four response means, including zero gates at every non-special response token. These summary tables always include all seven candidates.</p><details><summary>Question-token means</summary>{summary_table('query')}</details><details><summary>Gate values before the first answer prediction</summary>{summary_table('last_prompt')}</details><div class="legend"><b>Shared activation scale</b><div class="ramp"></div><div class="ends"><span>0 · white = inactive</span><span>{scale:.4f} · maximum across every input and response</span></div><p class="fine">The same scale applies to every gate, question and response, including controls. Hover a token for its exact gate, raw score and position.</p></div><div class="feature-views" role="group" aria-label="Visible candidate gates"><button data-view-button="first" aria-pressed="true">Priority three</button><button data-view-button="secondary" aria-pressed="false">Other four</button><button data-view-button="all" aria-pressed="false">All seven</button></div>{''.join(sections)}<section id="method"><h2>What was measured?</h2><p>The twelve QA case IDs, prompts, original grades and saved native responses are identical to the earlier confidence-candidate screen. Two wording controls are separate. Fresh generation has no monitoring hooks and exactly reproduces all twelve saved native token sequences. Observation-only replay preserves all {V['verified_next_token_predictions']} native next-token predictions across fourteen cases.</p><pre>z = (residual after attention at the candidate layer − released input mean) / released input std
raw = z dot encoder row + bias
gate = raw when raw &gt; threshold and raw &gt; 0; otherwise 0</pre><p>Hooks only read inputs to post-attention layernorm and return None. The native MLP and attention remain unchanged. There is no gate clamp, added contribution, donor patch or other steering. Query-only positions come from tokenizer character offsets overlapping the exact final question; shared instructions, passage and chat markers are kept separately.</p><p>Colour on a response token measures its gate after that token has already been emitted. The final encoded input position is measured before predicting the first answer token. Those are different observation times. {h(controlnote)}</p><p class="fine">Pinned Qwen3–4B/MOLT revisions, BF16 model and FP32 gate math, greedy batch one, unfrozen eager attention, KV cache, thinking disabled and a 96-new-token cap. Means exclude special tokens. All indices are zero-based.</p><h2>Interpretation limits</h2><p>Twelve selected examples do not establish a general confidence detector. Missing-information wording and latent uncertainty may differ. Gates in later layers can reflect already formed answers; a between-group response contrast does not establish a causal reason for different steering strengths. No steering or correlation analysis is performed here. Failed wording probes remain visible.</p><div class="downloads"><a href="artifacts/measurements.json">All measured tokens</a><a href="gate-summary.csv">Means and active-token counts</a><a href="verification.json">Verification</a><a href="protocol.json">Frozen protocol</a><a href="cases.json">Questions and original grades</a><a href="source/run.py">Executed runner</a></div><p class="fine">SQuAD2.0 selected text: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0. “Made up” means unsupported by the passage. SPAR GPU retained for follow-ups; no lifecycle changes.</p></section></main><script>{js}</script></body></html>'''
(R/'index.html').write_text(page);shutil.copy2(S/'style.css',R/'style.css')
(R/'summary.json').write_text(json.dumps({'shared_colour_max':scale,'qa_response_active_cases':response_active,'qa_question_active_cases':query_active,'qa_final_prompt_active_cases':last_active,'response_summaries':stats},indent=2)+'\n')
with (R/'gate-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
(R/'source').mkdir(exist_ok=True)
for p in S.iterdir():
 if p.is_file():shutil.copy2(p,R/'source'/p.name)
print(json.dumps({'qa_response_active_cases':response_active,'qa_question_active_cases':query_active,'qa_final_prompt_active_cases':last_active,'shared_colour_max':scale}))
