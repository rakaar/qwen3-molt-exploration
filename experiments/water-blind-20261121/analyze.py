"""Compare frozen text labels using all sampled occurrences and cluster bootstrap intervals."""
import hashlib
import json
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parents[2]/'results/2026-11-21-water-blind'
def read(name): return json.loads((out/name).read_text())
def sha(name): return hashlib.sha256((out/name).read_bytes()).hexdigest()
def save(name,x): (out/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lock=read('label-freeze.json')
assert sha('labels.json')==lock['labels_sha256'] and sha('sample.json')==lock['sample_sha256']
assert sha('gates.json')==read('measurement.json')['gates_sha256']
s=read('sample.json'); labels=read('labels.json'); gates=read('gates.json')
rows=[]
for r,l,g in zip(s,labels,gates,strict=True):
    for k in ['id','conversation','position']: assert r[k]==l[k]==g[k]
    assert r['token_id']==g['token_id'] and g['gate']>=0 and np.isfinite(g['gate'])
    rows.append({**r,'modifier':l['next_noun_modifier'],'label_note':l['note'],'gate':g['gate'],'raw':g['raw']})
assert len({(r['conversation'],r['position']) for r in rows})==len(rows)==240
save('observations.json',rows)
conv,ci=np.unique([r['conversation'] for r in rows],return_inverse=True)
y=np.array([r['gate'] for r in rows])
x=np.array([r['modifier'] for r in rows])
def group_summary(yes):
    v=y[x==yes]; fired=v[v>0]
    return dict(n=len(v),conversations=int(len(set(ci[x==yes]))),fires=len(fired),
                rate=float(np.mean(v>0)),mean=float(v.mean()),median=float(np.median(v)),
                mean_when_firing=float(fired.mean()),median_when_firing=float(np.median(fired)))
def effects(sums):
    # totals ordered as [N, firing count, gate sum] by group
    den,events,total=sums[...,0],sums[...,1],sums[...,2]
    metrics=np.stack([events/den,total/den,total/events],axis=-1)
    return metrics[...,1,:]-metrics[...,0,:]
totals=np.zeros((len(conv),2,3))
for i in range(len(y)): totals[ci[i],int(x[i])]+=[1,int(y[i]>0),y[i]]
rng=np.random.default_rng(20261121)
boot=[]
for j in range(100):
    indices=rng.integers(0,len(conv),size=(100,len(conv)))
    boot.extend(effects(totals[indices].sum(axis=1)))
ci_bounds=np.quantile(boot,[.025,.975],axis=0)
diff=effects(totals.sum(axis=0))
keys=['rate','mean','mean_when_firing']
result=dict(modifier=group_summary(True),other=group_summary(False),differences={
    k:dict(estimate=float(diff[j]),lower=float(ci_bounds[0,j]),upper=float(ci_bounds[1,j])) for j,k in enumerate(keys)},
    bootstrap=dict(unit='conversation',resamples=10000,seed=20261121,interval='95% percentile',conversations=len(conv)))
save('summary.json',result)
print(json.dumps(result,indent=2))
review=read('annotation-review.json')
alt_ids=set(review['positive_ids'])
alt=np.array([r['id'] in alt_ids for r in rows])
lower=np.array([r['token'].strip()=='water' for r in rows])
sensitivity={}
for key,group,mask in [('reviewer_labels',alt,np.ones(len(x),dtype=bool)),('lowercase_only',x,lower)]:
    values=[y[(group==j)&mask] for j in [False,True]]
    sensitivity[key]=dict(modifier_n=len(values[1]),other_n=len(values[0]),
                          rate_difference=float(np.mean(values[1]>0)-np.mean(values[0]>0)),
                          mean_difference=float(values[1].mean()-values[0].mean()))
save('sensitivity.json',sensitivity)
def describe(group):
    return (f"| {group['fires']} / {group['n']} | {100*group['rate']:.1f}% | "
            f"{group['mean']:.3f} | {group['median']:.3f} | {group['mean_when_firing']:.3f} |")
report=[
    '# Does T2375 fire more often when “water” modifies the following noun?', '',
    'We sampled 240 occurrences without replacement from 5,989 eligible occurrences in the final 11,765 conversations of UltraChat test_sft. Each case fell within the first 1,024 Qwen tokens of its conversation. Joined words and lexically hyphenated uses were excluded. All 240 causal prefixes were distinct. The sample spanned 199 prompt IDs, all distinct from those in the original dashboard collection.', '',
    'One Codex text-only annotation pass labeled the grammatical role before measurement. A second independent Codex pass agreed on 239 of 240 labels. The primary analysis keeps the original coding of “Denver Water proposal” in other uses.', '',
    '| Use | Firing / sampled | Firing rate | Mean gate, all | Median gate, all | Mean gate when firing |',
    '|---|---:|---:|---:|---:|---:|',
    '| Modifies adjacent noun '+describe(result['modifier']),
    '| Other uses '+describe(result['other']), '',
    'Differences below use modifier minus other. Intervals resample whole conversations 10,000 times.',
    '', '| Measure | Difference | 95% interval |','|---|---:|---:|']
for k,title in [('rate','Firing rate (percentage points)'),('mean','Mean gate'),('mean_when_firing','Mean gate among firings')]:
    d=result['differences'][k];scale=100 if k=='rate' else 1
    report.append(f"| {title} | {scale*d['estimate']:.3f} | {scale*d['lower']:.3f} to {scale*d['upper']:.3f} |")
report += ['', 'Measurement used the full native Qwen3-4B causal prefix ending exactly on “water,” the residual input to layer 29’s post-attention layer norm, and the released T2375 gate threshold. A gate above zero counts as firing. The subsequent words were available to the labeler only. Topic, wording and sentence structure can all differ between groups.', '',
    '- [All 240 observed contexts, labels and gates](observations.json)',
    '- [Sampling frame and rubric](sampling.json)',
    '- [Frozen labels](labels.json) and [timestamped hashes](label-freeze.json)',
    '- [Independent text-only review](annotation-review.json) and [sensitivity checks](sensitivity.json)',
    '- [Measurement versions and checksums](measurement.json) and [36 calibration positions](calibration.json)',
    '- [Sampling, annotation, measurement and analysis code](../../experiments/water-blind-20261121/)',
    '- [Earlier activation-selected pilot](../2026-11-21-t2375-water-continuations/README.md)']
(out/'README.md').write_text('\n'.join(report)+'\n')
