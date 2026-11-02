"""Independent audit and complete active-MOLT control report."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
NAME='eleven-active-controls-20261103'
SOURCE=ROOT/'comparisons/2026-11-03-eleven-active-controls/remote-export'/NAME
OUT=ROOT/'results/2026-11-03-eleven-active-controls';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+(0x7fff+((bits>>16)&1)))&0xffff0000).view(np.float32)
p=SOURCE/'artifacts/results.json';d=json.loads(p.read_text())
assert d['status']=='complete' and d['hooks_remaining']==0
for file,key in [('run.py','source_sha256'),('protocol.json','protocol_sha256'),('selection.json','selection_sha256'),('reference.json','reference_sha256'),('weights/manifest.json','weights_manifest_sha256')]:assert sha(SOURCE/file)==d['runtime'][key]
manifest=json.loads((SOURCE/'weights/manifest.json').read_text());weights={}
for f in manifest['features']:
    path=SOURCE/'weights'/f['file'];assert sha(path)==f['sha256']
    with np.load(path,allow_pickle=False) as a:weights[f['transform']]={k:a[k].copy() for k in a.files}
assert list(weights)==d['protocol']['selected_transforms']==[34,587,799,858,1181,2375]
assert [r['transform'] for r in d['runs']]==list(weights)
assert sha(SOURCE/'weights/all-gates.npz')==manifest['all_gate_file_sha256']
with np.load(SOURCE/'weights/all-gates.npz',allow_pickle=False) as a:gw={k:a[k].copy() for k in a.files}
rows=list(d['controls'].values())+d['runs']+d['repeats']
prob_error=0.;edit_count=0;vectors={};native_mlp=None
for row in rows:
    path=SOURCE/'artifacts'/row['arrays']['file'];assert sha(path)==row['arrays']['sha256']
    with np.load(path,allow_pickle=False) as a:
        logits=a['first_logits'];assert hashlib.sha256(logits.tobytes()).hexdigest()==row['steps'][0]['logits_sha256']
        assert int(logits.argmax())==row['steps'][0]['token_id']
        probs=np.exp(logits.astype(np.float64)-float(logits.max()));probs/=probs.sum()
        for ci,(text,tids) in enumerate(d['probe_token_ids'].items()):
            score=row['candidate_scores'][text];assert score['token_ids']==tids
            conditional=[]
            for j,tid in enumerate(tids):
                l=logits if j==0 else a[f'candidate_{ci}_step_{j}']
                pr=np.exp(l.astype(float)-float(l.max()));pr/=pr.sum()
                value=float(pr[tid]);error=abs(value-score['conditional_probabilities'][j])
                assert error<2e-6;prob_error=max(prob_error,error);conditional.append(value)
            assert np.isclose(np.prod(conditional),score['probability'],rtol=5e-6,atol=1e-14)
        if 'z' not in a:continue
        w=weights[row['transform']];z=(a['residual']-w['input_mean'])/w['input_std'];assert np.allclose(z,a['z'],atol=1e-6,rtol=1e-6)
        raw=z@w['e'].reshape(-1)+w['bias'][0];assert np.allclose(raw,a['raw'],atol=1e-5,rtol=1e-5)
        gate=np.where((raw>w['theta'][0])&(raw>0),raw,0.);assert np.allclose(gate,a['gate'],atol=1e-5,rtol=1e-5)
        all_raw=z@gw['encoder'].T+gw['bias'];assert np.allclose(all_raw,a['all_raw'],atol=1e-5,rtol=1e-5)
        all_gate=np.where((all_raw>gw['threshold'])&(all_raw>0),all_raw,0.)
        assert np.allclose(all_gate,a['all_gate'],atol=1e-5,rtol=1e-5)
        pos=row['position'];assert pos==7 and row['prompt_tokens'][pos]==' is'
        assert np.flatnonzero(all_gate[pos]).tolist()==d['protocol']['selected_transforms']
        op=(z@w['V']@w['U'])*w['output_std']
        assert np.linalg.norm((op-a['operator']).astype(float))/np.linalg.norm(a['operator'].astype(float))<4e-6
        if row['target_gate'] is None:continue
        assert row['selected_input_positions']==[7] and row['generated_positions_edited']==[]
        assert row['edit_calls']==1 and row['generated_passes_without_edit']==len(row['steps'])-1
        delta=np.zeros_like(a['native_mlp']);delta[pos]=np.float32(row['alpha'])*(np.float32(row['target_gate'])-a['gate'][pos])*a['operator'][pos]
        assert np.array_equal(delta,a['intended_delta'])
        changed=a['native_mlp'].copy();changed[pos]=bf16(changed[pos]+delta[pos])
        assert np.array_equal(changed,a['edited_mlp']) and np.array_equal(changed[:pos],a['native_mlp'][:pos])
        assert np.array_equal(a['actual_delta'],changed-a['native_mlp'])
        if native_mlp is None:native_mlp=a['native_mlp'].copy()
        else:assert np.array_equal(native_mlp,a['native_mlp'])
        contribution=(a['gate'][pos]*a['operator'][pos]).astype(float)
        row['contribution_l2']=float(np.linalg.norm(contribution))
        row['intended_delta_l2']=float(np.linalg.norm(delta[pos].astype(float)))
        row['actual_delta_l2']=float(np.linalg.norm(a['actual_delta'][pos].astype(float)))
        if row['alpha']!=0:
            if row['transform'] in vectors:assert np.array_equal(vectors[row['transform']],contribution)
            vectors[row['transform']]=contribution
        edit_count+=1
assert d['controls']['native']['steps']==d['controls']['observed']['steps']==d['controls']['zero_strength']['steps']
for r in d['repeats']:
    original=next(a for a in d['runs'] if a['transform']==r['transform'])
    assert r['steps']==original['steps'] and r['output_token_ids']==original['output_token_ids']

def record(row):
    r=dict(row);r['month_probabilities']={m:sum(r['candidate_scores'][' '+s]['probability'] for s in [m,m.title()]) for m in ['january','february','march','april','may','june','july','august','september','october','november','december']}
    r['initial_word']=r['response'].lstrip().split()[0].strip('.,:;!?')
    r['output_tokens_exactly_native']=r['output_token_ids']==d['controls']['native']['output_token_ids']
    r['rank']=next(f['rank'] for f in manifest['features'] if f['transform']==r['transform'])
    if 'contribution_l2' in r:
        c=vectors[r['transform']];ref=vectors[2375]
        r['contribution_cosine_to_T2375']=float(c@ref/(np.linalg.norm(c)*np.linalg.norm(ref)))
    return r
runs=[record(r) for r in d['runs']];native=record(d['controls']['native'])
validation={'status':'pass','all_layer_transforms_scanned':2480,'active_at_final_input':6,'other_active_controls':5,
            'array_files_audited':len(rows),'audited_edit_arrays':edit_count,'probability_max_absolute_error':prob_error,
            'native_observer_zero_controls_exact':True,'native_prior_reference_exact':d['native_prior_reference_exact'],
            'T2375_prior_reference_exact':d['T2375_prior_reference_exact'],'all_six_ablation_repeats_exact':True,
            'all_gate_active_set_reproduced':True,'bf16_edits_exact':True,'unselected_local_outputs_unchanged':True,
            'no_generated_token_edits':True,'full_local_native_MLP_equal_across_edits':True}
out={'protocol':d['protocol'],'selection':d['selection'],'native':native,'runs':runs,'runtime':d['runtime'],
     'validation':validation,'probability_definition':d['probability_definition'],'source_results_sha256':sha(p),
     'december_initial_transforms':[r['transform'] for r in runs if r['initial_word'].lower()=='december'],
     'contribution_cosine_transform_order':list(vectors)}
labels_path=ROOT/'experiments'/NAME/'cached-labels.json'
if labels_path.exists():
    out['cached_gate_labels']=json.loads(labels_path.read_text())
(OUT/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
fields=['transform','natural_gate','contribution_l2','actual_delta_l2','contribution_cosine_to_T2375','initial_word','p_november','p_december','response','stop_reason']
with (OUT/'controls.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
    for r in runs:wr.writerow({**{k:r[k] for k in fields if k not in ['p_november','p_december']},'p_november':r['month_probabilities']['november'],'p_december':r['month_probabilities']['december']})
np.savez_compressed(OUT/'contributions.npz',**{f'T{k}':v for k,v in vectors.items()})
lines=['# Active L29 MOLT ablation controls','',
       'Prompt: `the eleventh month of the year is`. Only final input position 7 (` is`) is edited. Every active transform is removed separately at its own natural magnitude. No generated-token edits or joint removal.',
       '',f'All 2,480 L29 gates were scanned on the saved native state. Exactly six are active: {d["protocol"]["selected_transforms"]}. Selection preceded all ablation outcomes. Runtime reproduces the same active set.',
       '',f'Native initial answer: {native["initial_word"]}. P(November variants) {100*native["month_probabilities"]["november"]:.4f}%; P(December variants) {100*native["month_probabilities"]["december"]:.4f}%.',
       '','| Transform removed | Natural gate | Contribution L2 | Initial answer | P(November) | P(December) |','|---|---:|---:|---|---:|---:|']
for r in runs:lines.append(f'| T{r["transform"]} | {r["natural_gate"]:.6f} | {r["contribution_l2"]:.4f} | {r["initial_word"]} | {100*r["month_probabilities"]["november"]:.4f}% | {100*r["month_probabilities"]["december"]:.4f}% |')
lines+=['',f'December-initial removals: {out["december_initial_transforms"]}. This is a one-prompt specificity control, not a broad causal-label validation.',
        '',f'Other controls with the entire generated token sequence identical to native: {[r["transform"] for r in runs if r["transform"]!=2375 and r["output_tokens_exactly_native"]]}. The full-vocabulary logits can change while greedy output remains identical.',
        '','The natural removed contributions have different norms. These are natural-strength ablations, not equal-norm perturbations. Contribution norm and actual rounded edit norm are retained. Equal or larger unrelated edits that preserve the answer provide a useful comparison; smaller edits do not establish equal-strength specificity.',
        '','Probabilities sum exact leading-space lowercase/title-case month-name prefixes, multiplying token probabilities when needed; punctuation/EOS and other renderings excluded. All 12 month-name probabilities are saved.',
        '','## Full recorded responses','', '### Native','','```text',native['response'],'```','']
for r in runs:lines += [f'### T{r["transform"]} off','','```text',r['response'],'```','']
lines+=['## Method and checks','',
        'Same pinned BF16 Qwen3-4B and MOLT checkpoint. Natural contribution = gate * ((z @ V) @ U) * output_std, retaining checkpoint standardizers. Subtract in FP32, then cast selected output once to BF16. Native MLP remainder retained. No extra vector normalization. Eager attention, TF32 off, greedy decoding, 32 generated-token maximum.',
        '','Every ablation repeats exactly in token IDs and full-vocabulary logit hashes. Native/observer/zero controls are exact. Native and T2375-off reproduce prior records. The CPU audit reconstructs all gates, operators, probabilities, exact BF16 edits and token masks. All raw files and the archive are hash-verified.',
        '','```json',json.dumps(validation,indent=2),'```','',
        'Reproduce: `python3 experiments/eleven-active-controls-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-eleven-active-controls/`. No publication or GPU cleanup was performed.']
if 'cached_gate_labels' in out:
    lines+=['','## Cached dashboard labels','',
            'These descriptions were cached on 21 October, not refreshed live. They describe activation conditions and are not causal explanations of the ablation results.','']
    for r in out['cached_gate_labels']['rows']:lines.append(f'- T{r["feature_id"]}: {r["title"]}.')
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'native':native['month_probabilities'],'rows':[{k:r[k] for k in ['transform','natural_gate','contribution_l2','initial_word','month_probabilities','response']} for r in runs],'validation':validation},indent=2))
