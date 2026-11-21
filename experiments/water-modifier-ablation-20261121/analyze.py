"""Summarize paired effects and render two equally scaled probability plots."""
from pathlib import Path
import csv,json,hashlib
from xml.etree import ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

out=Path(__file__).resolve().parents[2]/'results/2026-11-21-water-modifier-ablation'
rows=json.loads((out/'paired.json').read_text())
verification=json.loads((out/'verification.json').read_text())
assert verification['paired_sha256']==hashlib.sha256((out/'paired.json').read_bytes()).hexdigest()
assert len(rows)==len({r['id'] for r in rows})==179
assert sum(r['gate']==0 for r in rows)==22
conv,ci=np.unique([r['conversation'] for r in rows],return_inverse=True)
gate=np.array([r['gate'] for r in rows])
strict=np.array([r['strict_adjacent'] for r in rows])
def summary(rs,stem,seed):
    on=np.array([r[stem+'_on'] for r in rs]);off=np.array([r[stem+'_off'] for r in rs]);delta=100*(off-on)
    conv,cl=np.unique([r['conversation'] for r in rs],return_inverse=True)
    totals=np.bincount(cl,weights=delta);den=np.bincount(cl);rng=np.random.default_rng(seed)
    means=[]
    for _ in range(100):
        i=rng.integers(0,len(conv),size=(100,len(conv)))
        means.extend(totals[i].sum(axis=1)/den[i].sum(axis=1))
    example=min(range(len(rs)),key=lambda i:delta[i]);plus=max(range(len(rs)),key=lambda i:delta[i])
    return dict(n=len(rs),conversations=len(conv),intact_mean=float(on.mean()),off_mean=float(off.mean()),mean_delta_pp=float(delta.mean()),median_delta_pp=float(np.median(delta)),mean_delta_pp_ci95=np.quantile(means,[.025,.975]).tolist(),off_lower=int((delta<0).sum()),off_higher=int((delta>0).sum()),exact_tie=int((delta==0).sum()),abs_change_ge_1pp=int((abs(delta)>=1).sum()),largest_decrease=dict(id=rs[example]['id'],delta_pp=float(delta[example])),largest_increase=dict(id=rs[plus]['id'],delta_pp=float(delta[plus])))
stats={}
for name,subset in [('all',rows),('active_only',[r for r in rows if r['gate']>0]),('strict_adjacent',[r for r in rows if r['strict_adjacent']])]:
    stats[name]={stem:summary(subset,stem,20261121) for stem in ['actual','predicted']}
stats.update(n=len(rows),conversations=len(conv),zero_controls=sum(r['gate']==0 for r in rows),first_top_changed=sum(r['on_top_id']!=r['off_top_id'] for r in rows),greedy_word_changed=sum(r['pred_word']!=r['off_greedy_word'] for r in rows),actual_matches_intact_greedy=sum(r['target_ids']==r['pred_ids'] for r in rows),pred_nonwords=sum(r['pred_kind']!='word' for r in rows),pred_multitoken=sum(len(r['pred_ids'])>1 for r in rows),actual_multitoken=sum(len(r['target_ids'])>1 for r in rows),bootstrap='10,000 conversation-cluster bootstrap resamples. Delta is off minus intact in percentage points.')
(out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
fields=['id','conversation','position','before','water_token','after','strict_adjacent','gate','word','actual_sequence','actual_on','actual_off','pred_word','pred_sequence','pred_kind','predicted_on','predicted_off','off_greedy_word','off_greedy_sequence','on_top_token','off_top_token','actual_first_on','actual_first_off','on_top_first_on','on_top_first_off','actual_delta_pp','predicted_delta_pp']
with (out/'cases.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader()
    for r in rows:writer.writerow(dict(r,actual_delta_pp=100*(r['actual_off']-r['actual_on']),predicted_delta_pp=100*(r['predicted_off']-r['predicted_on'])))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#273e49','text.color':'#213843','axes.edgecolor':'#9fb4b8','xtick.color':'#4e646b','ytick.color':'#4e646b'})
fig,axes=plt.subplots(1,2,figsize=(10.8,5.1),layout='constrained')
titles=['Recorded next word','Intact model’s greedy continuation']
for ax,stem,title in zip(axes,['actual','predicted'],titles):
    x=np.array([r[stem+'_on'] for r in rows]);y=np.array([r[stem+'_off'] for r in rows])
    ax.plot([0,1],[0,1],ls='--',lw=1,color='#819397',zorder=1)
    for i,r in enumerate(rows):
        active=r['gate']>0
        dot=ax.scatter(x[i],y[i],s=23 if active else 34,marker='o' if active else 's',facecolor='#138877' if active else 'none',linewidths=.5 if active else 1,edgecolor='#15665f' if active else '#a65b32',alpha=.64 if active else .9,zorder=3 if active else 4)
        dot.set_gid(f'{stem}_case_{r["id"]}')
    ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),aspect='equal',xlabel='T2375 intact',ylabel='T2375 off at “water”')
    ax.set_xticks(np.arange(0,1.01,.25));ax.set_yticks(np.arange(0,1.01,.25))
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(True,alpha=.15);ax.set_axisbelow(True);ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=12)
    ax.legend(handles=[Line2D([],[],marker='o',markersize=5,ls='',color='#138877',label='Gate active (157)'),Line2D([],[],marker='s',markersize=5,ls='',markerfacecolor='none',color='#a65b32',label='Gate zero (22)')],loc='upper left',frameon=False,fontsize=9)
fig.savefig(out/'scatter.png',dpi=175,bbox_inches='tight');fig.savefig(out/'scatter.svg',bbox_inches='tight')
plt.close(fig)
# Add native tooltip descriptions to the standalone vector figure.
p=out/'scatter.svg';text=p.read_text()
from html import escape
for stem in ['actual','predicted']:
    for r in rows:
        tag=f'<g id="{stem}_case_{r["id"]}">'
        assert text.count(tag)==1
        word=r['word'] if stem=='actual' else r['pred_sequence']
        label=f"Case {r['id']} · {word.strip()} · intact {r[stem+'_on']:.2%} · off {r[stem+'_off']:.2%} · gate {r['gate']:.2f}"
        text=text.replace(tag,tag+'<title>'+escape(label)+'</title>')
p.write_text(text)
print(json.dumps({k:stats[k] for k in ['all','active_only','strict_adjacent','first_top_changed','greedy_word_changed','pred_nonwords','pred_multitoken','actual_multitoken']},indent=2))
flips=[r for r in rows if r['on_top_id']!=r['off_top_id']]
stats['first_top_flips_with_ties']=sum(r['on_max_ties']>1 or r['off_max_ties']>1 for r in flips)
stats['first_top_flips_unique_both']=sum(r['on_max_ties']==r['off_max_ties']==1 for r in flips)
(out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
def line(stem,title):
    x=stats['all'][stem];lo,hi=x['mean_delta_pp_ci95']
    return f"| {title} | {100*x['intact_mean']:.2f}% | {100*x['off_mean']:.2f}% | {x['mean_delta_pp']:+.2f} | {lo:+.2f} to {hi:+.2f} |"
example=next(r for r in rows if r['id']==885)
report=['# Does removing T2375 at “water” change the next word?', '',
  'Paired Qwen3-4B runs on all 179 occurrences in the original noun-modifier card, spanning 141 conversations. Both runs receive the identical full chat prefix ending at the measured “water” token. The intact run uses Qwen’s native MLP. The off run subtracts T2375’s naturally gated low-rank contribution from the layer-29 MLP output at this one token. Subsequent positions use the resulting attention cache and proceed without further intervention.', '',
  '![Two scatter plots comparing intact and off probabilities](scatter.png)', '',
  'Each point scores the same token sequence under both conditions. The first panel scores the recorded continuation through the end of its first word; the second chooses that continuation by greedy decoding from the intact model. Where punctuation precedes the word, the score includes those punctuation tokens. A score multiplies the conditional probabilities of the tokens in that path. It ends at the word’s last token; it does not include the probability of a later word boundary or sum alternate tokenizations. The CSV also supplies the immediate next-token version of each comparison.', '',
  '| Target | Mean intact probability | Mean off probability | Off minus intact, pp | 95% interval, pp |','|---|---:|---:|---:|---:|',
  line('actual','Recorded continuation'),line('predicted','Intact model’s greedy continuation'),'',
  'Intervals are paired percentile bootstraps over 141 whole conversations with 10,000 draws. The 22 native-zero gates reproduced identical probabilities. Among the 157 firing cases, the mean change for the recorded word was −0.046 percentage points. In 40 of the 179 cases the absolute change for the recorded continuation was at least one percentage point.', '',
  f"Five greedy predictions changed. In {stats['first_top_flips_with_ties']} of these a BF16 next-token logit tie occurred in at least one condition. In case 885, after “Sydney’s water”, the intact favorite “management” fell from {example['predicted_on']:.2%} to {example['predicted_off']:.2%}; the recorded word “supply” rose from {example['actual_on']:.2%} to {example['actual_off']:.2%} and became the off favorite.", '',
  'Four frozen labels involve coordination (785, 787, 934, 973). The recorded next word in these positions is “and” or “or”. After excluding those cases, the mean change for the recorded continuation was −0.016 percentage points (95% interval −0.305 to +0.295). Seventeen recorded and 18 greedy target sequences contain multiple tokens. The results describe a local subtraction using the learned MOLT decomposition, with sensitivity to bfloat16 precision and the contexts drawn.', '',
  '- [Per-case results](cases.csv) and [group estimates](summary.json)',
  '- [Zoomable vector figure](scatter.svg)',
  '- [Full records with token IDs](paired.json) and [verification hashes](verification.json)',
  '- [Preparation, intervention and plotting code](../../experiments/water-modifier-ablation-20261121/)']
(out/'README.md').write_text('\n'.join(report)+'\n')
