"""Aggregate paired full-vocabulary KL by offset, with conversation-cluster uncertainty."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,colors,ticker
from matplotlib.patches import Patch
out=Path(__file__).resolve().parents[2]/'results/2026-11-21-water-modifier-kl'
def load(name):return json.loads((out/name).read_text())
def write_csv(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
verify=load('verification.json')
assert hashlib.sha256((out/'measurements.json').read_bytes()).hexdigest()==verify['sha256']
rs=load('measurements.json');cases=load('cases.json');protocol=load('protocol.json')
ids={c['id']:i for i,c in enumerate(cases)};conv,cluster=np.unique([c['conversation'] for c in cases],return_inverse=True)
n=len(cases);offset=np.arange(protocol['horizon']+1);k=np.full((n,len(offset)),np.nan);tv=k.copy();flip=k.copy()
for r in rs:
    ix=(ids[r['id']],r['offset'])
    assert np.isnan(k[ix])
    k[ix]=r['kl_nats'];tv[ix]=r['total_variation'];flip[ix]=int(r['intact_top_id']!=r['off_top_id'])
valid=np.isfinite(k);active=np.array([c['original_gate']>0 for c in cases])
assert np.all(k[~active][valid[~active]]==0)
full=valid[:,-1];strict=np.array([c['strict_adjacent'] for c in cases])
num=np.zeros((len(conv),len(offset)));den=np.zeros_like(num)
np.add.at(num,cluster,np.nan_to_num(k));np.add.at(den,cluster,valid.astype(float))
rng=np.random.default_rng(20261121)
boot=[]
for _ in range(100):
    indices=rng.integers(len(conv),size=(100,len(conv)))
    boot.append(num[indices].sum(axis=1)/den[indices].sum(axis=1))
lo,hi=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
mean=np.nanmean(k,axis=0);med=np.nanmedian(k,axis=0)
by=[]
for j in offset:
    v=k[valid[:,j],j]
    by.append(dict(offset=int(j),n=len(v),conversations=len(np.unique(cluster[valid[:,j]])),active_n=int(np.sum(active&valid[:,j])),mean_kl=float(mean[j]),lo95=float(lo[j]),hi95=float(hi[j]),median_kl=float(med[j]),p90_kl=float(np.quantile(v,.9)),max_kl=float(v.max()),active_only_mean=float(np.nanmean(k[active,j])),complete_window_mean=float(k[full,j].mean()),strict_mean=float(np.nanmean(k[strict,j])),mean_total_variation=float(np.nanmean(tv[:,j])),top_id_changes=int(np.nansum(flip[:,j]))))
write_csv('by-offset.csv',by);write_csv('per-case-offset.csv',rs)
max_offsets=np.nanargmax(k[active],axis=1)
summary=dict(cases=n,conversations=len(conv),horizon=int(offset[-1]),full_window=int(full.sum()),controls=int((~active).sum()),positions=len(rs),
    peak_mean_offset=int(np.argmax(mean)),largest_delayed_mean_offset=int(np.argmax(mean[1:])+1),peak_at_water_active=int((max_offsets==0).sum()),active_cases=int(active.sum()),
    delayed_peak_active=int((max_offsets>0).sum()),bootstrap='10,000 conversation-cluster resamples; pointwise 95% percentile intervals',by_offset=by)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Left panel emphasizes the mean and its uncertainty. Right panel preserves all cases,
# exact zeros and missing suffixes. The log color scale reveals small delayed changes.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'water-kl-20261121','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8ba1aa','text.color':'#1e3941','axes.labelcolor':'#26454c','xtick.color':'#49626a','ytick.color':'#49626a'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.85),gridspec_kw={'width_ratios':[1,1.05]},layout='constrained')
ax.fill_between(offset,lo,hi,color='#badfd8',label='Mean · 95% pointwise interval')
ax.plot(offset,mean,color='#106e64',marker='o',ms=3.3,lw=2,zorder=4)
ax.plot(offset,med,color='#786ba0',ls='--',lw=1.6,label='Median')
ax.set_yscale('symlog',linthresh=.00001,linscale=.65);ax.set_ylim(bottom=0,top=max(hi)*1.35)
ticks=[float(10.**j) for j in range(-5,2) if 10.**j<=hi.max()*1.3]
ax.set_yticks([0]+ticks,labels=['0']+[f'{v:.5f}'.rstrip('0').rstrip('.') if v<1 else str(int(v)) for v in ticks])
ax.set_xlim(-.65,32.5);ax.set_xticks([0,1,4,8,16,24,32]);ax.set_xticklabels(['0','','4','8','16','24','32'])
ax.grid(axis='y',color='#d9e2e2',lw=.8);ax.legend(loc='upper right',frameon=False,fontsize=9)
ax.set_title('Average change by position',loc='left',fontweight='bold',fontsize=14);ax.set_xlabel('Recorded tokens read after “water”');ax.set_ylabel('KL(intact ‖ off), nats')
ordered=np.argsort(-k[:,0],kind='stable')
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#cad4d9');cmap.set_under('white')
min_color=1e-6;vmax=max(.01,10.**math.ceil(np.log10(np.nanmax(k))))
values=k[ordered].copy();values[(values>0)&(values<min_color)]=min_color;values[values==0]=min_color/10
img=bx.imshow(np.ma.masked_invalid(values),aspect='auto',interpolation='nearest',origin='upper',cmap=cmap,norm=colors.LogNorm(vmin=min_color,vmax=vmax),rasterized=True)
bx.set_title('Individual occurrences',loc='left',fontweight='bold',fontsize=14)
bx.set_xlabel('Recorded tokens read after “water”');bx.set_ylabel('Cases ranked by KL at “water”')
bx.set_xticks([0,4,8,16,24,32]);bx.set_yticks([0,49,99,149,178],labels=['1','50','100','150','179'])
cb=fig.colorbar(img,ax=bx,shrink=.86,pad=.02);cb.set_label('KL (nats)');cb.ax.tick_params(labelsize=8)
bx.legend(handles=[Patch(facecolor='white',edgecolor='#7c969e',label='Exact zero'),Patch(facecolor='#cad4d9',label='Turn ended')],loc='lower center',bbox_to_anchor=(.5,-.28),ncol=2,frameon=False,fontsize=9)
fig.savefig(out/'kl-by-position.png',dpi=175,bbox_inches='tight');fig.savefig(out/'kl-by-position.svg',metadata={'Date':None},bbox_inches='tight');plt.close(fig)
print('peak',summary['peak_mean_offset'],'delayed',summary['largest_delayed_mean_offset'],'active peaks at water',summary['peak_at_water_active'])
for j in [0,1,2,3,4,8,16,32]:print(j,{key:by[j][key] for key in ['n','mean_kl','median_kl','lo95','hi95','mean_total_variation','complete_window_mean','top_id_changes']})
immediate=by[0]; delayed=by[summary['largest_delayed_mean_offset']]
lines=['# T2375 ablation and next-token KL after “water”','',
  f'The largest mean change occurred immediately, with mean KL {immediate["mean_kl"]:.5f} nats at offset 0 and median {immediate["median_kl"]:.6f}. The largest mean over the following positions was {delayed["mean_kl"]:.6f} nats at +{delayed["offset"]}. The immediate mean is {immediate["mean_kl"]/delayed["mean_kl"]:.1f} times that later mean. The immediate mean total variation was {100*immediate["mean_total_variation"]:.2f}%, the fraction of probability mass redistributed on average.','',
 '![Mean and median KL by offset and individual heatmap](kl-by-position.png)','',
 '| Offset after “water” | Cases | Mean KL (nats) | 95% interval | Median KL | Mean total variation |',
 '|---:|---:|---:|---:|---:|---:|']
for j in (0,1,2,4,8,16,32):
    r=by[j];lines.append(f"| {j} | {r['n']} | {r['mean_kl']:.6f} | {r['lo95']:.6f}–{r['hi95']:.6f} | {r['median_kl']:.6f} | {100*r['mean_total_variation']:.2f}% |")
lines+=['','## Measurement','',
 f'The cohort comprises the same 179 frozen noun-modifier cases in {len(conv)} conversations. For each case the model received the original full chat prefix through “water”. The intervention subtracted the naturally gated L29–T2375 contribution only at that token. Both runs then received identical recorded continuations under causal attention, with their own resulting caches. At +1, for example, the model has read one additional token and is predicting the token after it. Windows stop before the next chat special token or after 32 following tokens. Four windows extend past the original 1,024-token sampling boundary to recover the recorded continuation.','',
 f'Each measurement uses all {verify["vocabulary"]:,} logit entries: sum over tokens of p × ln(p / q), where p is the intact probability and q the ablated probability. Model outputs are BF16, and log-softmax and KL arithmetic use FP64. The intervals resample whole conversations 10,000 times and are pointwise. Case order in the heatmap uses immediate KL. {summary["peak_at_water_active"]} of the {summary["active_cases"]} firing cases reached their largest observed KL at “water”; the remaining cases each had up to 32 later opportunities to exceed it.','',
 f'All {summary["controls"]} zero-gate controls produced bit-identical logits across both trajectories at every available offset. The immediate recorded-token probabilities and top-token IDs matched the earlier paired experiment. The 163 complete-window cases also showed an immediate peak (mean {by[0]["complete_window_mean"]:.5f} nats at offset 0). The downloadable summary includes active-only and strict-adjacency means, as well as counts of top-token changes.','',
 '- [Per-offset data](by-offset.csv) and [all paired measurements](per-case-offset.csv)',
 '- [Full-size figure](kl-by-position.svg), [frozen protocol](protocol.json), and [verification](verification.json)',
 '- [Preparation, model run and plotting code](../../experiments/water-modifier-kl-20261121/)']
(out/'README.md').write_text('\n'.join(lines)+'\n')
