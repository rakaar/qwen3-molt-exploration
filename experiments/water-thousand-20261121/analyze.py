"""Join frozen labels and measured gates; cluster-bootstrap the two planned rate contrasts."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parents[2]/'results/2026-11-21-water-thousand'
def read(file):return json.loads((out/file).read_text())
def save(file,x):(out/file).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(file):return hashlib.sha256((out/file).read_bytes()).hexdigest()
lock=read('label-freeze.json');run=read('measurement.json')
assert sha('labels.json')==lock['labels_sha256'] and sha('sample.json')==lock['sample_sha256'] and sha('gates.json')==run['gates_sha256']
rows=[]
for r,l,g in zip(read('sample.json'),read('labels.json'),read('gates.json'),strict=True):
    assert all(r[k]==g[k]==l[k] for k in ['id','conversation','position'])
    assert r['token_id']==g['token_id'] and g['gate']>=0 and np.isfinite(g['gate'])
    rows.append({**r,'category':l['category'],'raw':g['raw'],'gate':g['gate']})
assert len(rows)==len({r['id'] for r in rows})==1000
save('observations.json',rows)
groups={}
for k in 'MSJHFO':
    sel=[r for r in rows if r['category']==k]
    v=np.array([r['gate'] for r in sel]);n=len(v);fires=int((v>0).sum())
    groups[k]=dict(n=n,fires=fires,rate=fires/n if n else None,median=float(np.median(v)) if n else None,
        mean=float(v.mean()) if n else None,median_when_firing=float(np.median(v[v>0])) if fires else None)
conv,ci=np.unique([r['conversation'] for r in rows],return_inverse=True)
rng=np.random.default_rng(2026112102)
counts=np.zeros((len(conv),3,2))
for i,r in enumerate(rows):
    j=0 if r['category']=='M' else 1 if r['category']=='S' else 2
    counts[ci[i],j]+=[r['gate']>0,1]
def diffs(total):
    fraction=lambda x:x[...,0]/x[...,1]
    m,s,oth=[total[...,i,:] for i in range(3)]
    return np.stack([fraction(m)-fraction(s),fraction(m)-fraction(s+oth)],axis=-1)
observed=diffs(counts.sum(axis=0))
boots=np.concatenate([diffs(counts[rng.integers(0,len(conv),size=(100,len(conv)))].sum(axis=1)) for _ in range(100)])
bounds=np.quantile(boots,[.025,.975],axis=0)
contrasts={k:dict(estimate=float(observed[i]),lower=float(bounds[0,i]),upper=float(bounds[1,i])) for i,k in enumerate(['M_minus_S','M_minus_other'])}
result=dict(groups=groups,contrasts=contrasts,n=1000,conversations=len(conv),fires=sum(g['fires'] for g in groups.values()),bootstrap='10,000 resamples, conversation-cluster percentile 95% intervals')
save('summary.json',result)
print(json.dumps(result,indent=2))
names={'M':'Noun modifier','S':'Standalone noun','J':'Joined or inflected word','H':'Hyphenated adjective','F':'Fixed expression','O':'Other uses'}
sample=read('sampling.json')
lines=['# T2375 on 1,000 random water occurrences','',
 f'The frame contained {sample["population"]:,} occurrences in the first 1,024 rendered tokens of {sample["eligible_conversations"]:,} eligible UltraChat test_sft conversations. The uniform sample without replacement contained {len(conv)} distinct conversations. All 1,000 model prefixes were distinct. The original dashboard corpus rows and all 199 conversations from the 240-case study were excluded. The dashboard evaluation supplied 50 activating windows and 50 zero-gate windows for the earlier pilot.', '',
 'Each category summarizes its own denominator. Firing means a native T2375 gate above zero. The median includes every sampled occurrence in the category. The examples on the local cards illustrate the observed range.', '',
 '| Use | Sample share | Fired / sampled | Firing rate | Median gate |','|---|---:|---:|---:|---:|']
for k,v in groups.items():
    if v['n']: lines.append(f"| {names[k]} | {v['n']/10:.1f}% | {v['fires']} / {v['n']} | {v['rate']:.1%} | {v['median']:.2f} |")
lines.extend(['',f"{result['fires']} of the 1,000 positions fired. The modifier-minus-standalone firing-rate difference was {100*contrasts['M_minus_S']['estimate']:.1f} percentage points (95% interval {100*contrasts['M_minus_S']['lower']:.1f} to {100*contrasts['M_minus_S']['upper']:.1f}, resampling whole conversations).", '',
 'Two Codex annotators each labeled a separate half of the sample using text alone. Labels were frozen before measurement. Fixed expressions include figurative and title uses whose classification is debatable. The joined category includes compounds, plurals, inflections and names. BPE tokens can span a whole joined word; the gate is read on the token containing the final r of water. An activation can therefore coincide with part or all of a following spelling. The measured relationship is observational.', '',
 '- [Local results cards](../../sites/t2375-water-cards/index.html)',
 '- [All observations and gates](observations.json)',
 '- [Sampling frame](sampling.json) and [category rubric](../../experiments/water-thousand-20261121/rubric.txt)',
 '- [Labels](labels.json), [premeasurement hashes](label-freeze.json), and [measurement verification](measurement.json)',
 '- [Group results and uncertainty intervals](summary.json)'])
(out/'README.md').write_text('\n'.join(lines)+'\n')
# Exploratory spelling split noticed during review of the joined-word examples.
joined=[r for r in rows if r['category']=='J']
split={}
for key,interior in [('word_initial',False),('after_a_letter',True)]:
    subset=[r for r in joined if r['word_before'][-1:].isalpha()==interior]
    split[key]=dict(n=len(subset),fired=sum(r['gate']>0 for r in subset))
save('joined-position-exploratory.json',split)
initial=split['word_initial'];later=split['after_a_letter']
with (out/'README.md').open('a') as f:
    f.write(f"\n## Exploratory spelling comparison\n\nWithin the joined or inflected category, {initial['fired']}/{initial['n']} positions fired when “water” began the word and {later['fired']}/{later['n']} fired when a letter preceded it. This follow-up was chosen after examining the results. The groups also differ in tokenizer boundaries and vocabulary.\n")

with (out/'README.md').open('a') as f:
    f.write('\n## Follow-up check of noun adjacency\n\nDuring the next-word ablation, an audit of the 179 frozen modifier labels identified four coordinated phrases (IDs 785, 787, 934, 973) in which punctuation or a conjunction intervenes before the head noun. All four fired. Restricting the modifier group to the 175 directly adjacent cases gives 153/175 firing (87.4%), compared with 157/179 (87.7%) under the frozen labels. The [follow-up ablation](../2026-11-21-water-modifier-ablation/README.md) reports both sets.\n')
