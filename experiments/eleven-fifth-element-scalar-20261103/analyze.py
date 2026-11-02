"""Audit saved native/edited vectors, BF16 mask and exact answer-prefix probabilities."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='eleven-fifth-element-scalar-20261103'
SOURCE=ROOT/'comparisons/2026-11-03-eleven-fifth-element-scalar/remote-export'/NAME
OUT=ROOT/'results/2026-11-03-eleven-fifth-element-scalar'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+(0x7fff+((bits>>16)&1)))&0xffff0000).view(np.float32)
def softmax(x):
    p=np.exp(x.astype(np.float64)-float(x.max()));return p/p.sum()
p=SOURCE/'artifacts/results.json'
d=json.loads(p.read_text())
assert d['status']=='complete' and d['hooks_remaining']==0
assert sha(SOURCE/'run.py')==d['runtime']['source_sha256']
assert sha(SOURCE/'protocol.json')==d['runtime']['protocol_sha256']
assert d['protocol']['scalar_only'] and not d['protocol']['copy_donor_operator']
assert [r['target_gate'] for r in d['runs']]==d['protocol']['fixed_gates']
wpath=ROOT/'experiments/eleven-activation-20261102/weights/L29-T2375.npz'
assert sha(wpath)==d['runtime']['feature_sha256']
with np.load(wpath,allow_pickle=False) as a:w={k:a[k].copy() for k in a.files}
all_rows=[d['controls'][label][kind] for label in ['fifth','eleventh','thirteenth'] for kind in ['native','observed']]+[d['controls']['zero_strength']]+d['runs']+d['repeats']
max_probability_error=0.;edit_count=0;recipient_operator=None
for row in all_rows:
    ap=SOURCE/'artifacts'/row['arrays']['file'];assert sha(ap)==row['arrays']['sha256']
    with np.load(ap,allow_pickle=False) as a:
        logits=a['first_logits']
        assert hashlib.sha256(logits.tobytes()).hexdigest()==row['steps'][0]['logits_sha256']
        assert int(logits.argmax())==row['steps'][0]['token_id']
        for ci,(text,tids) in enumerate(d['probe_token_ids'].items()):
            score=row['candidate_scores'][text]
            assert score['token_ids']==tids
            expected=[]
            for j,tid in enumerate(tids):
                v=float(softmax(a['first_logits'] if j==0 else a[f'candidate_{ci}_step_{j}'])[tid])
                error=abs(v-score['conditional_probabilities'][j]);assert error<2e-6
                max_probability_error=max(max_probability_error,error);expected.append(v)
            assert np.isclose(np.prod(expected),score['probability'],rtol=5e-6,atol=1e-14)
            assert score['conditional_probabilities'][0]==row['first_token_probabilities'][text]
        if 'z' not in a:continue
        z=(a['residual']-w['input_mean'])/w['input_std'];assert np.allclose(z,a['z'],atol=1e-6,rtol=1e-6)
        raw=z@w['e'].reshape(-1)+w['bias'].reshape(-1)[0];assert np.allclose(raw,a['raw'],atol=1e-5,rtol=1e-5)
        gate=np.where((raw>w['theta'].reshape(-1)[0])&(raw>0),raw,0.);assert np.allclose(gate,a['gate'],atol=1e-5,rtol=1e-5)
        op=(z@w['V']@w['U'])*w['output_std']
        assert np.linalg.norm((op-a['operator']).astype(np.float64))/np.linalg.norm(a['operator'].astype(np.float64))<2e-6
        pos=row['position'];assert row['prompt_tokens'][pos]==' is' and pos==len(row['prompt_token_ids'])-1
        assert row['natural_gate']==float(a['gate'][pos])
        if row['prompt']==d['protocol']['controls']['fifth']['prompt']:
            if recipient_operator is None:recipient_operator=a['operator'].copy()
            else:assert np.array_equal(recipient_operator,a['operator'])
        if row['target_gate'] is None:continue
        assert row['selected_input_positions']==[pos] and row['generated_positions_edited']==[]
        assert row['edit_calls']==1 and row['generated_passes_without_edit']==len(row['steps'])-1
        delta=np.zeros_like(a['native_mlp'],dtype=np.float32)
        delta[pos]=np.float32(row['alpha'])*(np.float32(row['target_gate'])-a['gate'][pos])*a['operator'][pos]
        assert np.array_equal(delta,a['intended_delta'])
        changed=a['native_mlp'].copy();changed[pos]=bf16(changed[pos]+delta[pos])
        assert np.array_equal(changed,a['edited_mlp'])
        assert np.array_equal(a['actual_delta'],changed-a['native_mlp'])
        assert np.array_equal(changed[:pos],a['native_mlp'][:pos])
        row['intended_delta_l2']=float(np.linalg.norm(delta[pos].astype(np.float64)))
        row['actual_delta_l2']=float(np.linalg.norm(a['actual_delta'][pos].astype(np.float64)))
        edit_count+=1
for label,case in d['protocol']['controls'].items():
    c=d['controls'][label]
    assert c['native']['steps']==c['observed']['steps'] and c['native']['candidate_scores']==c['observed']['candidate_scores']
    assert re.match(case['expected_initial_pattern'],c['native']['response'],re.I)
assert d['controls']['zero_strength']['steps']==d['controls']['fifth']['native']['steps']
for repeated in d['repeats']:
    original=next(r for r in d['runs'] if r['target_gate']==repeated['target_gate'])
    assert repeated['steps']==original['steps'] and repeated['candidate_scores']==original['candidate_scores']

def record(row):
    r=dict(row);m=re.match(r'\s*([A-Za-z]+|[0-9]+)',r['response'])
    r['initial_word']=m.group(1) if m else r['steps'][0]['token']
    for name in ['boron','sodium','aluminum','aluminium']:
        r['p_'+name]=sum(r['candidate_scores'][text]['probability'] for text in [' '+name,' '+name.title()])
    r['p_aluminum_spellings']=r['p_aluminum']+r['p_aluminium']
    return r
rows=[record(r) for r in d['runs']]
native=record(d['controls']['fifth']['native'])
peaks={key:{'gate':max(rows,key=lambda r:r[key])['target_gate'],'probability':max(r[key] for r in rows)} for key in ['p_sodium','p_aluminum_spellings']}
assert not any(r['initial_word'].lower() in ['sodium','aluminum','aluminium'] for r in rows)
validation={'status':'pass','sweep_conditions':len(rows),'array_files_checked':len(all_rows),'audited_edit_arrays':edit_count,
            'probability_max_absolute_error':max_probability_error,'native_capability_controls':3,
            'native_observer_and_zero_exact':True,'repeated_gates_exact':[r['target_gate'] for r in d['repeats']],
            'recipient_operator_identical_across_gates':True,'bf16_edits_exact':True,'input_only_mask_exact':True,
            'candidate_prefix_scores_reconstructed':True,'no_generated_token_edits':True}
summary={'feature':d['feature'],'protocol':d['protocol'],'native':native,'controls':d['controls'],'runs':rows,
         'runtime':d['runtime'],'probe_token_ids':d['probe_token_ids'],'validation':validation,
         'probability_definition':d['probability_definition'],'source_results_sha256':sha(p),
         'outcome':{'sodium_or_aluminum_initial_answers':0,'peaks_among_tested_strengths':peaks}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
fields=['target_gate','initial_word','p_boron','p_sodium','p_aluminum','p_aluminium','p_aluminum_spellings','intended_delta_l2','actual_delta_l2','response','stop_reason']
with (OUT/'sweep.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:row[k] for k in fields})
lines=['# Fifth-element scalar-gate sweep','','Prompt: `'+d['protocol']['controls']['fifth']['prompt']+'`. Feature: L29–T2375.',
       '',f'No tested strength gives an initial sodium or aluminum/aluminium answer. Sodium peaks among tested values at gate {peaks["p_sodium"]["gate"]}: {100*peaks["p_sodium"]["probability"]:.6f}%, versus {100*native["p_sodium"]:.6f}% natively. Aluminum/aluminium peaks at {100*peaks["p_aluminum_spellings"]["probability"]:.6f}%. High strengths produce unrelated starts such as `eletrum`, followed by a return to boron.',
       '',f'Only final input token ` is`, position {rows[0]["position"]}, is edited. Keep recipient UVx; no donor vector and no generated-token edits.',
       '',f'Natural fifth-element gate: {rows[0]["natural_gate"]:.9f}. Scalar replacement uses the same 17 fixed gates as the preceding month sweep.',
       '','The fifth element is boron, the eleventh is sodium, and the thirteenth is aluminum/aluminium. All three native capability checks pass. These are spelled-ordinal raw completion prefixes, not chat questions.',
       '','| Gate | Initial word | P(boron) | P(sodium) | P(aluminum + aluminium) |','|---:|---|---:|---:|---:|']
for r in rows:
    shown=r['initial_word'].replace('\n',r'\n')
    lines.append(f'| {r["target_gate"]:.6f} | {shown} | {100*r["p_boron"]:.6f}% | {100*r["p_sodium"]:.6f}% | {100*r["p_aluminum_spellings"]:.6f}% |')
lines+=['','Probabilities sum exact lowercase/title-case leading-space answer prefixes, multiplying token probabilities for multi-token spellings; no subsequent punctuation or EOS. Chemical symbols and other phrasings are excluded. They are not semantic answer probabilities.',
        '','Native BF16 Qwen3-4B; pinned model/MOLT; FP32 gate/operator and edit with one BF16 cast; eager attention; TF32 off; greedy 32-token continuations. No extra normalization or donor operator copy.',
        '','## Native capability controls','']
for label in ['fifth','eleventh','thirteenth']:
    c=d['controls'][label];lines += ['### '+label,'','Prompt: `'+c['native']['prompt']+'`.',f'Final input gate: {c["observed"]["natural_gate"]:.9f}.','','```text',c['native']['response'],'```','']
lines+=['## Full sweep responses','']
for r in rows:lines += [f'### Gate {r["target_gate"]:.9f}','','```text',r['response'],'```','']
lines+=['## Validation and scope','',json.dumps(validation,indent=2),'',
        'All arrays and the remote archive are SHA-256 verified. CPU analysis reconstructs gate/operator values, exact BF16 edits, input masks and exact-prefix probabilities. Native/observer/zero controls and three repeated strengths match exactly.',
        '','This deterministic exploratory sweep covers one prompt. An initial word change is not a stable abstract ordinal mechanism; full continuations are preserved.',
        '','Reproduce: `python3 experiments/eleven-fifth-element-scalar-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-eleven-fifth-element-scalar/`. No publication or GPU cleanup was performed.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'validation':validation,'controls':{k:{'gate':v['observed']['natural_gate'],'response':v['native']['response']} for k,v in d['controls'].items() if k!='zero_strength'},'rows':[{k:r[k] for k in fields if k not in ['response','stop_reason']} for r in rows]},indent=2,ensure_ascii=False))
