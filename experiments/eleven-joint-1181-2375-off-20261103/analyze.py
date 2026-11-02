"""Independent audit of the joint natural-contribution removal and 2x2 comparison."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
NAME='eleven-joint-1181-2375-off-20261103'
SOURCE=ROOT/'comparisons/2026-11-03-eleven-joint-1181-2375-off/remote-export'/NAME
OUT=ROOT/'results/2026-11-03-eleven-joint-1181-2375-off';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bf16(x):
    b=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((b+(0x7fff+((b>>16)&1)))&0xffff0000).view(np.float32)
p=SOURCE/'artifacts/results.json';d=json.loads(p.read_text());assert d['status']=='complete' and d['hooks_remaining']==0
assert sha(SOURCE/'run.py')==d['runtime']['source_sha256'] and sha(SOURCE/'protocol.json')==d['runtime']['protocol_sha256']
weights={}
for index in [1181,2375]:
    path=ROOT/f'experiments/eleven-active-controls-20261103/weights/L29-T{index}.npz'
    assert sha(path)==d['runtime']['feature_sha256'][str(index)]
    with np.load(path,allow_pickle=False) as a:weights[index]={k:a[k].copy() for k in a.files}
rows=list(d['controls'].values())+d['runs']+d['repeats'];error_max=0.;edits=0;contribution_vectors={};native_mlp=None
for r in rows:
    ap=SOURCE/'artifacts'/r['arrays']['file'];assert sha(ap)==r['arrays']['sha256']
    with np.load(ap,allow_pickle=False) as a:
        logits=a['first_logits'];assert hashlib.sha256(logits.tobytes()).hexdigest()==r['steps'][0]['logits_sha256']
        assert int(logits.argmax())==r['steps'][0]['token_id']
        for ci,(text,tids) in enumerate(d['probe_token_ids'].items()):
            score=r['candidate_scores'][text];assert score['token_ids']==tids;values=[]
            for j,tid in enumerate(tids):
                l=logits if j==0 else a[f'candidate_{ci}_step_{j}'];pr=np.exp(l.astype(float)-float(l.max()));pr/=pr.sum()
                value=float(pr[tid]);error=abs(value-score['conditional_probabilities'][j]);assert error<2e-6;error_max=max(error_max,error);values.append(value)
            assert np.isclose(np.prod(values),score['probability'],rtol=5e-6,atol=1e-14)
        if 'z' not in a:continue
        pos=r['position'];assert pos==7 and r['prompt_tokens'][pos]==' is'
        contributions={}
        for index,w in weights.items():
            z=(a['residual']-w['input_mean'])/w['input_std'];assert np.allclose(z,a['z'],atol=1e-6,rtol=1e-6)
            raw=z@w['e'].reshape(-1)+w['bias'][0];gate=np.where((raw>w['theta'][0])&(raw>0),raw,0.)
            assert np.allclose(raw,a[f'raw_T{index}'],atol=1e-5,rtol=1e-5)
            assert np.allclose(gate,a[f'gate_T{index}'],atol=1e-5,rtol=1e-5)
            op=(z@w['V']@w['U'])*w['output_std']
            assert np.linalg.norm((op-a[f'operator_T{index}']).astype(float))/np.linalg.norm(a[f'operator_T{index}'].astype(float))<3e-6
            contributions[index]=a[f'gate_T{index}'][pos]*a[f'operator_T{index}'][pos]
            if index in contribution_vectors:assert np.array_equal(contribution_vectors[index],contributions[index])
            contribution_vectors[index]=contributions[index].copy()
        if r['edit_calls']==0:continue
        assert r['selected_input_positions']==[7] and r['generated_positions_edited']==[]
        assert r['edit_calls']==1 and r['generated_passes_without_edit']==len(r['steps'])-1
        total=np.zeros(2560,dtype=np.float32)
        for index in r['off_transforms']:total=total+contributions[index]
        delta=np.zeros_like(a['native_mlp']);delta[pos]=-np.float32(r['alpha'])*total
        assert np.array_equal(delta,a['intended_delta'])
        changed=a['native_mlp'].copy();changed[pos]=bf16(changed[pos]+delta[pos])
        assert np.array_equal(changed,a['edited_mlp']) and np.array_equal(changed[:pos],a['native_mlp'][:pos])
        assert np.array_equal(a['actual_delta'],changed-a['native_mlp'])
        if native_mlp is None:native_mlp=a['native_mlp'].copy()
        else:assert np.array_equal(native_mlp,a['native_mlp'])
        r['intended_delta_l2']=float(np.linalg.norm(delta[pos].astype(float)))
        r['actual_delta_l2']=float(np.linalg.norm(a['actual_delta'][pos].astype(float)));edits+=1
native=d['controls']['native']
for key in ['observed','zero_strength']:
    assert d['controls'][key]['steps']==native['steps'] and d['controls'][key]['candidate_scores']==native['candidate_scores']
refpath=ROOT/d['protocol']['reference_source_path'];assert sha(refpath)==d['protocol']['reference_source_sha256']
for label,r in [('native',native)]+[(str(r['off_transforms'][0]),r) for r in d['runs'] if len(r['off_transforms'])==1]:
    ref=d['protocol']['references'][label]
    assert r['output_token_ids']==ref['output_token_ids'] and [s['logits_sha256'] for s in r['steps']]==ref['logit_hashes']
assert d['runs'][-1]['off_transforms']==[1181,2375]
assert d['runs'][-1]['steps']==d['repeats'][0]['steps'] and d['runs'][-1]['candidate_scores']==d['repeats'][0]['candidate_scores']
months=['january','february','march','april','may','june','july','august','september','october','november','december']
def record(r):
    r=dict(r);r['month_probabilities']={m:sum(r['candidate_scores'][' '+s]['probability'] for s in [m,m.title()]) for m in months}
    r['initial_word']=r['response'].lstrip().split()[0].strip('.,:;!?');return r
native=record(native);runs=[record(r) for r in d['runs']]
validation={'status':'pass','arrays_audited':len(rows),'edit_arrays_audited':edits,'probability_max_absolute_error':error_max,
            'native_observer_zero_exact':True,'native_and_both_single_references_exact':True,'joint_repeat_exact':True,
            'both_contribution_vectors_identical_across_conditions':True,'FP32_sum_before_single_BF16_cast_exact':True,
            'native_local_MLP_same_across_conditions':True,'only_final_input_edited':True,'no_generated_token_edits':True,'all_feature_hashes_verified':True}
summary={'protocol':d['protocol'],'native':native,'runs':runs,'runtime':d['runtime'],'validation':validation,
         'probability_definition':d['probability_definition'],'source_results_sha256':sha(p),
         'natural_contribution_l2':{str(i):float(np.linalg.norm(c.astype(float))) for i,c in contribution_vectors.items()}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
conditions=[('Native',native),('T1181 off',runs[0]),('T2375 off',runs[1]),('Both off',runs[2])]
with (OUT/'month-probabilities.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['condition','month','probability'])
    for label,r in conditions:
        for month in months:wr.writerow([label,month,r['month_probabilities'][month]])
lines=['# Joint T1181/T2375 ablation on the eleventh-month prompt','',
       'Prompt: `the eleventh month of the year is`. Only final input position 7 (` is`) edited. Natural contributions removed independently or together; no generated-token edit.',
       '','| Condition | Initial word | P(November) | P(December) | P(January) |','|---|---|---:|---:|---:|']
for label,r in conditions:lines.append(f'| {label} | {r["initial_word"]} | {100*r["month_probabilities"]["november"]:.6f}% | {100*r["month_probabilities"]["december"]:.6f}% | {100*r["month_probabilities"]["january"]:.6f}% |')
lines+=['','Probabilities sum exact leading-space lowercase/title-case complete answer prefixes, excluding following punctuation/EOS and other answer formats. Multi-token month names use complete sequence probabilities. All twelve months are retained in month-probabilities.csv.',
        '','## Full recorded responses','']
for label,r in conditions:lines += ['### '+label,'','```text',r['response'],'```','']
lines+=['## Method and validation','',
        'Same pinned native BF16 Qwen3-4B and MOLT checkpoint. Contributions use native pre-MLP residual and checkpoint input/output standardizers. For the joint edit, sum c1181 and c2375 in FP32, subtract from native local MLP output, then cast once to BF16. No sequential BF16 edits or extra vector normalization. Native MLP remainder and normal downstream computation retained.',
        '','Native, observer and zero-strength controls match exactly. Fresh native/single-ablation runs reproduce all earlier generated token IDs and full-vocabulary logit hashes. Joint ablation repeats exactly. Independent CPU auditing reconstructs every candidate score, both gates/operators, selected-position masks and exact FP32-sum/BF16 edit. Raw archive/files are hash-verified.',
        '','One exact lowercase raw prefix; greedy 32-token continuations, eager attention, TF32 off. This tests dependence of this December result on T1181, not a universal ordinal circuit or direct connection between the MOLTs.',
        '','```json',json.dumps(validation,indent=2),'```','',
        'Reproduce: `python3 experiments/eleven-joint-1181-2375-off-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-eleven-joint-1181-2375-off/`. No publication or GPU cleanup was performed.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'conditions':[{'label':label,'initial_word':r['initial_word'],'response':r['response'],'month_probabilities':r['month_probabilities']} for label,r in conditions],'natural_contribution_l2':summary['natural_contribution_l2'],'validation':validation},indent=2))
