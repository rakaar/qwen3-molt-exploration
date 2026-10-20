"""Local, source-backed 3-by-2 plots and exact-token activation report."""
import csv, hashlib, html, json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle
R=Path(sys.argv[1]).resolve()
D=json.loads((R/'artifacts/measurements.json').read_text());P=D['protocol'];F=P['features']
REV=json.loads((R/'response-review.json').read_text());I=json.loads((R/'interpretation.json').read_text())
H=lambda x:html.escape(str(x),quote=True)
N=[f"L{f['layer']}-T{f['transform']}" for f in F]
C=LinearSegmentedColormap.from_list('privacy',['#f3f7f7','#c5e1df','#57afa7','#0e686b','#143c50'])
def records(row,seg):return [t for t in row[seg+'_tokens'] if not t['special']]
def label(t):return t.replace(' ','·').replace('\n','↵').replace('\t','⇥') or '∅'
M={s:max(t['features'][n]['gate'] for r in D['rows'] for t in records(r,s) for n in N) for s in ['answer','prompt']}
def draw(scenario,seg):
 rows=[r for r in D['rows'] if r['scenario']==scenario];cols=12
 height=max(math.ceil(len(records(r,seg))/cols) for r in rows)
 fig,axes=plt.subplots(3,2,figsize=(18,3*max(2.5,height*.62+1)+1.3),squeeze=False,layout='constrained')
 norm=Normalize(0,M[seg] or 1)
 for fi,(f,n) in enumerate(zip(F,N)):
  for ci,r in enumerate(rows):
   ax=axes[fi,ci];ts=records(r,seg);s=r['summaries'][n]['response' if seg=='answer' else 'prompt']
   ax.set_xlim(0,cols);ax.set_ylim(height,0);ax.axis('off')
   ax.set_title(f"{n} · {f['label']}\n{r['condition_label']} · mean {s['mean_gate']:.4f} · active {s['active_tokens']}/{len(ts)}",loc='left',fontsize=10.5,pad=13,color='#173f4a')
   for j,t in enumerate(ts):
    y,x=divmod(j,cols);g=t['features'][n]['gate'];color='white' if norm(g)>.58 else '#183844'
    ax.add_patch(Rectangle((x,y),.98,.94,facecolor=C(norm(g)),edgecolor='#d2dcdf',linewidth=.45))
    text=label(t['token']);ax.text(x+.49,y+.31,text,ha='center',va='center',fontsize=7.5 if len(text)<10 else 5.7,color=color,clip_on=True)
    ax.text(x+.49,y+.66,f'{g:.2f}',ha='center',va='center',fontsize=7,color=color)
 fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=C),ax=axes.ravel().tolist(),shrink=.45,pad=.025,label='Natural gate activation · shared absolute scale')
 title='Bob asks for a private code' if scenario=='bob' else 'A meeting summary sent to Kate'
 fig.suptitle(f"{title} · {'Response' if seg=='answer' else 'Prompt'} tokens\nThree privacy-labelled MOLTs × two matched conditions · Qwen3-4B · no steering",fontsize=18,color='#173f4a')
 note='Means include zeros; special tokens/EOS excluded. · = leading space; ↵ = newline. Gate measured while processing the displayed token.'
 if M[seg]==0:note+=' All gates are zero; 0–1 is a display-only color range.'
 fig.supxlabel(note,fontsize=9);fig.savefig(R/f'{scenario}-{seg}-heatmap.png',dpi=180,facecolor='white');plt.close(fig)
for scenario in ['bob','party']:
 for seg in ['answer','prompt']:draw(scenario,seg)
csvrows=[]
for r in D['rows']:
 for f,n in zip(F,N):
  for seg in ['response','prompt']:csvrows.append({'case_id':r['id'],'scenario':r['scenario'],'condition':r['condition'],'feature':n,'label':f['label'],'segment':seg,**r['summaries'][n][seg]})
with (R/'mean-activations.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
def chips(r,n,seg):
 output=[]
 for t in records(r,seg):
  g=t['features'][n]['gate'];v=g/(M[seg] or 1);bg='#%02x%02x%02x'%tuple(int(c*255) for c in C(v)[:3]);color='white' if v>.58 else '#183844'
  tip=f"Position {t['position']}; chat position {t['full_chat_position']}; token {t['token_id']}; gate {g:.9f}; raw {t['features'][n]['raw']:.9f}"
  output.append(f'<span class="token" title="{H(tip)}" style="background:{bg};color:{color}"><b>{H(label(t["token"]))}</b><small>{g:.3f}</small></span>')
 return ''.join(output)
def grid(rows,seg):
 cards=[]
 for f,n in zip(F,N):
  for r in rows:
   s=r['summaries'][n]['response' if seg=='answer' else 'prompt']
   cards.append(f'<article class="gate-card"><span class="eyebrow">{n} · {H(r["condition_label"])}</span><h3>{H(f["label"])}</h3><div class="numbers"><b>Mean {s["mean_gate"]:.4f}</b><span>{s["active_tokens"]}/{s["visible_tokens"]} active · max {s["max_gate"]:.3f}</span></div><div class="tokens">{chips(r,n,seg)}</div></article>')
 return '<div class="gate-grid">'+''.join(cards)+'</div>'
sections=[]
for scenario in ['bob','party']:
 rows=[r for r in D['rows'] if r['scenario']==scenario];cards=[]
 for r in rows:
  review=REV[r['id']];assert hashlib.sha256(r['answer'].encode()).hexdigest()==review['answer_sha256']
  cards.append(f'<article class="response"><span class="eyebrow">{H(r["condition_label"])}</span><h3>{H(review["headline"])}</h3><p class="answer">{H(r["answer"])}</p><p class="fine">{H(review["reason"])}</p><details><summary>Exact model input</summary><b>System</b><pre>{H(r["system"])}</pre><b>User</b><pre>{H(r["user"])}</pre><p>{H(r["provenance"])}</p><p>Expected: {H(r["expected"])}</p></details></article>')
 sections.append(f'<section id="{scenario}"><span class="eyebrow">MATCHED CONDITIONS</span><h2>{H(rows[0]["title"])}</h2><div class="two">{"".join(cards)}</div><h3>Response tokens · 3 features × 2 conditions</h3><p class="fine">Feature rows match across columns. Read left to right, wrapping naturally. Hover for exact values and positions. Means include zero gates and exclude special tokens/EOS.</p>{grid(rows,"answer")}<div class="links"><a href="{scenario}-answer-heatmap.png">Download 3 × 2 response figure (PNG)</a></div><details class="extra"><summary>Prompt-token activations and means</summary><p>System and user text and visible chat-template tokens are included; special tokens are excluded from this view and its means. Prompt and response averages have different denominators.</p>{grid(rows,"prompt")}<a href="{scenario}-prompt-heatmap.png">Download 3 × 2 prompt figure (PNG)</a></details></section>')
trs=[]
for f,n in zip(F,N):trs.append('<tr><td><b>'+n+'</b><br>'+H(f['label'])+'</td>'+''.join(f'<td>{r["summaries"][n]["response"]["mean_gate"]:.4f}</td>' for r in D['rows'])+'</tr>')
css='''*{box-sizing:border-box}body{margin:0;background:#f5f5ed;color:#193b47;font:17px/1.6 system-ui,sans-serif}main{max-width:1280px;padding:48px 30px 90px;margin:auto}header{padding:25px 0 42px}h1{font-size:clamp(36px,5vw,66px);line-height:1.07;letter-spacing:-2px;max-width:930px;margin:20px 0}h2{font-size:34px;line-height:1.15;margin:15px 0 25px}h3{font-size:20px;line-height:1.35;margin:12px 0}p{margin:12px 0}.eyebrow{font-size:12px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase;color:#176b6c}.intro{font-size:21px;max-width:960px}.badge{display:inline-block;border:1px solid #8bb4b1;border-radius:20px;padding:5px 14px;margin:4px 8px 4px 0;font-size:14px}section{border-top:1px solid #c5d2ce;padding:42px 0}.two,.gate-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.response,.gate-card{min-width:0;background:#fffef9;border:1px solid #d5dfd9;border-radius:14px;padding:22px}.answer{background:#edf4f1;padding:16px;border-radius:7px;white-space:pre-wrap}.gate-card{padding:18px}.gate-card h3{font-size:17px}.numbers{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:13px;margin:14px 0}.numbers b{font-size:17px}.tokens{display:flex;flex-wrap:wrap;gap:4px}.token{border:1px solid #cddbd9;border-radius:4px;display:inline-flex;flex-direction:column;align-items:center;min-width:32px;padding:3px 6px;font-family:ui-monospace,monospace;line-height:1.45}.token b{font-size:13px;font-weight:500}.token small{font-size:10px;opacity:.8}.hide-values .token small{display:none}.fine{font-size:13px;color:#546b72}details{margin-top:15px}summary{cursor:pointer;color:#116b6d;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf0ed;padding:16px;border-radius:8px;font-size:13px}.extra{background:#eef2ed;padding:20px;border-radius:12px}.extra .gate-grid{margin:20px 0}.note{padding:20px 24px;background:#e2eeea;border-left:4px solid #267c79;border-radius:4px;margin:22px 0}.links{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0}a{color:#09686e;text-underline-offset:4px}button{border:1px solid #8bb4b1;background:#fffef9;color:#17444c;border-radius:7px;padding:10px 16px;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #cedbd7;padding:14px;text-align:left}th{background:#e3eeea}.table-scroll{overflow:auto}.foot{font-size:13px;color:#546b72;margin-top:30px}@media(max-width:720px){main{padding:25px 15px 60px}.two,.gate-grid{grid-template-columns:1fr}.response,.gate-card{padding:15px}h2{font-size:29px}.intro{font-size:18px}h1{letter-spacing:-1px}th,td{padding:10px;min-width:100px}.links{gap:14px}}'''
page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Privacy MOLT activation pilot — two matched prompt pairs</title><style>{css}</style></head><body><main><header><span class="eyebrow">Qwen3-4B · native model · 15 October 2026 · local pilot</span><h1>Do privacy-labelled MOLTs respond to a permission change?</h1><p class="intro">Two scenarios. Two disclosure conditions each. Three candidate transforms, measured on the model’s own unsteered answers.</p><span class="badge">4 fixed prompts</span><span class="badge">3 MOLTs</span><span class="badge">No steering</span><span class="badge">Attention unfrozen</span><div class="note"><b>{H(I['headline'])}</b><p>{H(I['description'])}</p></div><div class="links"><a href="#bob">Bob’s code request</a><a href="#party">Meeting summary</a><a href="#means">Response means</a><a href="#methods">Exact method</a><button id="toggle-values" type="button" aria-pressed="true">Hide token values</button></div><p class="fine">Original synthetic Bob pair; short adaptation of ConfAIde’s surprise-party /10K meeting with our own shareable control. Not an official benchmark evaluation. Prompts were locked before generation.</p></header>{''.join(sections)}<section id="means"><span class="eyebrow">RESPONSE AVERAGES</span><h2>Mean gate across the complete visible answer</h2><div class="table-scroll"><table><thead><tr><th>Feature</th><th>Bob: withhold</th><th>Bob: share</th><th>Party: private</th><th>Party: shareable</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div><p class="fine">Includes every visible answer token, including zeros; not the mean over active tokens alone. Different answers have different token counts. Two prompt pairs do not support statistical inference.</p><div class="links"><a href="mean-activations.csv">Means and counts (CSV)</a><a href="artifacts/measurements.json">Every token, raw projection and gate (JSON)</a></div></section><section id="methods"><span class="eyebrow">EXACT MEASUREMENT</span><h2>The native model remains unchanged</h2><p>Each answer was generated without any MOLT hooks, then replayed with observation-only hooks. Every greedy next-token prediction had to match the original generation. Hooks return None, leaving inputs and outputs untouched; attention remains unfrozen.</p><p>At layers 11–13, read the residual after attention and before post-attention normalization. All IDs are zero-based. Standardize with the released MOLT input statistics and compute the selected gate in FP32. The native model stays BF16.</p><pre>z = (current post-attention residual − input mean) / input std
raw = dot(z, selected gate weight) + selected gate bias
gate = raw when raw > learned threshold and raw > 0; otherwise 0
mean response gate = sum of gates / all visible response tokens
No edit is applied.</pre><p>A token’s gate is measured while processing that token, after it has been emitted. Its activation is not evidence that it caused that same token. Prompt activations are also saved. Greedy batch one, thinking disabled, native KV cache, 96-token cap.</p><p>Selected weight slices, exact thresholds and hashes are saved. Input-dependent low-rank and contribution norms are also logged. Labels describe activation contexts and do not establish a privacy decision mechanism.</p><p><b>Interpretation limits:</b> {H(I['limitations'])}</p><details><summary>Candidate labels and dashboard explanations</summary>{''.join(f'<h3>{n} — {H(f["label"])}</h3><p>{H(f["explanation"])}</p>' for f,n in zip(F,N))}</details><details><summary>Pinned model identities</summary><pre>Base: {P['base_repo']}
Revision: {P['base_revision']}
MOLT: {P['molt_repo']}
Revision: {P['molt_revision']}
Torch {D['runtime']['torch']} · Transformers {D['runtime']['transformers']}</pre></details><div class="links"><a href="cases.json">Exact prompts</a><a href="protocol.json">Pre-run protocol</a><a href="response-review.json">Answer-bound review</a><a href="verification.json">Numerical verification</a><a href="run.py">Measurement source</a><a href="weights/manifest.json">Weight provenance</a><a href="https://github.com/skywalker023/confaide/blob/main/benchmark/tier_4.txt">ConfAIde transcript</a></div><p class="foot">Retained SPAR instance 56070285 reused at $0.175/hour including disk, plus transfers. No rental, steering, GitHub push or deployment. GPU left running as requested. Manual response review by Codex with conditions visible. Synthetic prompts contain no real secrets.</p></section></main><script>document.querySelector('#toggle-values').addEventListener('click',()=>{{let hidden=document.body.classList.toggle('hide-values'),b=document.querySelector('#toggle-values');b.textContent=hidden?'Show token values':'Hide token values';b.setAttribute('aria-pressed',String(!hidden))}})</script></body></html>'''
(R/'index.html').write_text(page)
(R/'summary.json').write_text(json.dumps({'headline':I['headline'],'interpretation':I,'response_color_max':M['answer'],'prompt_color_max':M['prompt'],'cases':[{'id':r['id'],'answer':r['answer'],'summaries':r['summaries']} for r in D['rows']]},ensure_ascii=False,indent=2)+'\n')
print('Built local report, four measured PNGs, means CSV and summary.')
