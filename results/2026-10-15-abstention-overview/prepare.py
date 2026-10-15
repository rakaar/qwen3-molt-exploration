"""Combine saved experiments and render measured native-gate evidence, no inference."""
from pathlib import Path
import hashlib,json,csv,math,textwrap
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/2026-10-15-abstention-overview';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads((ROOT/p).read_text())
old='results/2026-10-15-vampire-intermediate';new='results/2026-10-15-six-new-abstentions';obs='results/2026-10-14-passage-answer-gates'
oldrows=read(old+'/artifacts/all-results.json')['rows'];newrows=read(new+'/artifacts/all-steering.json')['rows'];baselines=read(old+'/baseline-generations.json')['rows'];oldp=read(old+'/baseline-cases.json');newcases=read(new+'/cohort-lock.json')['cases'];newsum=read(new+'/summary.json')
contexts={p['id']:p['text'] for p in oldp['passages']};base={r['id']:r for r in baselines};oldmap={(r['case_id'],r['added_gate']):r for r in oldrows if not r['trial_id'].startswith('fractional_')}
cases=[]
for cid,st in [('P1-U2',24),('P1-U5',16),('P2-U1',4),('P2-U5',24)]:
 b=base[cid];row=oldmap[cid,st];ok=cid!='P2-U5'
 cases.append({'id':cid,'cohort':'Original four baseline failures','question':b['question'],'context':contexts[b['passage_id']],'baseline':b['answer'],'steered':row['answer'],'strength':st,'success':ok,'strength_description':'Successful tested addition' if ok else 'Representative coherent failed addition','source_id':b['source_id'],'trial_id':row['trial_id'],'source_report':'../2026-10-15-vampire-intermediate/index.html','positive_strength_count':128 if not ok else None})
N={r['case_id']:r for r in newcases}
for s in newsum['cases']:
 cid=s['case_id'];b=N[cid];v=s['selected_strength'] if s['success'] else {'R13':26,'S07':20}[cid];row=next(r for r in newrows if r['case_id']==cid and r['added_gate']==v)
 cases.append({'id':cid,'cohort':'Six new baseline failures, fixed before steering','question':b['question'],'context':b['context'],'baseline':b['answer'],'steered':row['answer'],'strength':v,'success':s['success'],'strength_description':'Successful tested addition' if s['success'] else 'Representative coherent failed addition','source_id':b['source_id'],'trial_id':row['trial_id'],'source_report':'../2026-10-15-six-new-abstentions/index.html','positive_strength_count':s['positive_trials']})
assert len(cases)==10 and sum(c['success'] for c in cases)==7
measure=read(obs+'/artifacts/answer-gates.json');chosen=['P1-U2','P1-U5','P2-U1','P1-U3','P1-U4','P2-U4'];by={r['id']:r for r in measure['rows']};examples=[]
for cid in chosen:
 r=by[cid];tokens=[t for t in r['answer_tokens'] if not t['special']]
 assert not r['ancillary_explanation_error']
 examples.append({'id':cid,'group':'made_up' if r['group']=='unsupported_answer' else 'natural_abstention','question':r['question'],'answer':r['answer'],'tokens':tokens,'mean_gate':sum(t['gate'] for t in tokens)/len(tokens),'peak_gate':max(t['gate'] for t in tokens),'active_tokens':sum(t['gate']>0 for t in tokens),'visible_tokens':len(tokens),'saved_predictions_verified':r['saved_greedy_predictions_verified']})
assert all(x['mean_gate']==0 for x in examples[:3]) and all(x['mean_gate']>0 for x in examples[3:])
maximum=max(t['gate'] for r in measure['rows'] for t in r['answer_tokens'] if not t['special']);norm=Normalize(0,maximum);cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#edf0f1')
cols=14;heights=[math.ceil(len(r['tokens'])/cols)+1.2 for r in examples]
fig,axes=plt.subplots(6,1,figsize=(16,sum(heights)*.61+1.2),gridspec_kw={'height_ratios':heights})
for ax,r in zip(axes,examples):
 n=math.ceil(len(r['tokens'])/cols);values=np.full((n,cols),np.nan)
 for i,t in enumerate(r['tokens']):values[i//cols,i%cols]=max(0,t['gate'])
 ax.imshow(np.ma.masked_invalid(values),aspect='auto',norm=norm,cmap=cmap,interpolation='nearest')
 for i,t in enumerate(r['tokens']):
  token=t['token'].replace(' ','·').replace('\n','↵');color='white' if norm(t['gate'])>.65 else '#142b33'
  ax.text(i%cols,i//cols-.16,token,ha='center',va='center',fontsize=min(9.5,125/max(1,len(token))),fontfamily='monospace',color=color)
  ax.text(i%cols,i//cols+.22,f"{t['answer_position']} · {max(0,t['gate']):.2f}",ha='center',va='center',fontsize=8,color=color)
 title=('MADE-UP ANSWER' if r['group']=='made_up' else 'NATURAL I DON’T KNOW')+f" · {r['id']} · mean gate {r['mean_gate']:.3f} · {r['active_tokens']}/{r['visible_tokens']} active"
 ax.set_title(title+'\n'+textwrap.fill(r['question'],130),loc='left',fontsize=11,pad=12);ax.set_xticks([]);ax.set_yticks([]);ax.set_xticks(np.arange(-.5,cols,1),minor=True);ax.set_yticks(np.arange(-.5,n,1),minor=True);ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',left=False,bottom=False)
fig.suptitle('Natural L19–T368 gate activations — three made-up answers and three abstentions',fontsize=18,fontweight='bold',y=.995)
fig.text(.02,.012,f'No steering. Shared scale 0–{maximum:.3f}. Mean includes zero gates across visible response tokens; EOS excluded. · = leading space. Grey = padding.\nGate measured while processing each displayed token; observational association, not causal proof.',fontsize=10)
fig.tight_layout(rect=[0,.055,1,.965],h_pad=2.2);fig.savefig(OUT/'native-token-heatmap.png',dpi=180);plt.close(fig)
# Compact measured overview for poster and screen.
length=max(len(r['tokens']) for r in examples);matrix=np.full((6,length),np.nan)
for i,r in enumerate(examples):matrix[i,:len(r['tokens'])]=[max(0,t['gate']) for t in r['tokens']]
fig,ax=plt.subplots(figsize=(11.5,3.25));im=ax.imshow(np.ma.masked_invalid(matrix),norm=norm,cmap=cmap,interpolation='nearest',aspect='auto');ax.set_yticks(range(6),[f"{r['id']}  {'Made up' if r['group']=='made_up' else 'Abstained'}  · mean {r['mean_gate']:.3f}" for r in examples]);ax.set_xticks(range(0,length,5));ax.set_xlabel('Response token position; words differ by row');ax.axhline(2.5,c='#294851',lw=1.5);fig.colorbar(im,ax=ax,pad=.02,label='Gate');ax.set_title('Native model: no steering · L19–T368');fig.tight_layout();fig.savefig(OUT/'native-gate-overview.png',dpi=220);plt.close(fig)
data={'date':'2026-10-15','model':'Qwen3-4B','layer':19,'transform':368,'rank':128,'label':measure['protocol']['label'],'base_revision':measure['protocol']['base_revision'],'molt_revision':measure['protocol']['molt_revision'],'asset_sha256':measure['protocol']['asset_sha256'],'counts':{'original_successes':3,'original_cases':4,'new_successes':4,'new_cases':6,'combined_successes':7,'combined_cases':10,'new_questions_screened':80},'cases':cases,'observational_examples':examples,'observational_scale':[0,maximum],'observation_source':obs+'/artifacts/answer-gates.json','success_definition':'Coherent, explicit passage-grounded withholding with no unsupported substantive answer or false explanation. Readable but ambiguous self-corrections are not clear successes.','terminology':'Made up means unsupported by the supplied passage; some such answers are true general knowledge.','reviewer':'Codex, condition labels visible; no independent blinded human review.','source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [old+'/artifacts/all-results.json',new+'/artifacts/all-steering.json',obs+'/artifacts/answer-gates.json',new+'/cohort-lock.json']}}
(OUT/'evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with (OUT/'correlation-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['id','group','question','answer','mean_gate','peak_gate','active_tokens','visible_tokens']);w.writeheader()
 for r in examples:w.writerow({k:r[k] for k in w.fieldnames})
print(json.dumps({'combined_cases':len(cases),'successes':sum(c['success'] for c in cases),'natural_means':{x['id']:x['mean_gate'] for x in examples}},indent=2))
