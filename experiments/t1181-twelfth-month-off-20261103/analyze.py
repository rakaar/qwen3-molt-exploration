"""Independent CPU verification of the twelfth-month single-position ablation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
NAME='t1181-twelfth-month-off-20261103'
SOURCE=ROOT/'comparisons/2026-11-03-t1181-twelfth-month-off/remote-export'/NAME
OUT=ROOT/'results/2026-11-03-t1181-twelfth-month-off';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+(0x7fff+((bits>>16)&1)))&0xffff0000).view(np.float32)
p=SOURCE/'artifacts/results.json';d=json.loads(p.read_text())
assert d['status']=='complete' and d['hooks_remaining']==0 and d['feature']=='L29-T1181'
assert sha(SOURCE/'run.py')==d['runtime']['source_sha256']
assert sha(SOURCE/'protocol.json')==d['runtime']['protocol_sha256']
wp=ROOT/'experiments/eleven-active-controls-20261103/weights/L29-T1181.npz';assert sha(wp)==d['runtime']['feature_sha256']
with np.load(wp,allow_pickle=False) as a:w={k:a[k].copy() for k in a.files}
rows=list(d['controls'].values())+d['runs']+d['repeats']
max_error=0.;edits=0;operator=None
for r in rows:
    ap=SOURCE/'artifacts'/r['arrays']['file'];assert sha(ap)==r['arrays']['sha256']
    with np.load(ap,allow_pickle=False) as a:
        logits=a['first_logits'];assert hashlib.sha256(logits.tobytes()).hexdigest()==r['steps'][0]['logits_sha256']
        assert int(logits.argmax())==r['steps'][0]['token_id']
        for ci,(text,tids) in enumerate(d['probe_token_ids'].items()):
            score=r['candidate_scores'][text];assert score['token_ids']==tids
            conditional=[]
            for j,tid in enumerate(tids):
                l=logits if j==0 else a[f'candidate_{ci}_step_{j}'];pr=np.exp(l.astype(float)-float(l.max()));pr/=pr.sum()
                value=float(pr[tid]);error=abs(value-score['conditional_probabilities'][j]);assert error<2e-6;max_error=max(max_error,error);conditional.append(value)
            assert np.isclose(np.prod(conditional),score['probability'],rtol=5e-6,atol=1e-14)
        if 'z' not in a:continue
        z=(a['residual']-w['input_mean'])/w['input_std'];assert np.allclose(z,a['z'],atol=1e-6,rtol=1e-6)
        raw=z@w['e'].reshape(-1)+w['bias'][0];assert np.allclose(raw,a['raw'],atol=1e-5,rtol=1e-5)
        gate=np.where((raw>w['theta'][0])&(raw>0),raw,0.);assert np.allclose(gate,a['gate'],atol=1e-5,rtol=1e-5)
        op=(z@w['V']@w['U'])*w['output_std'];assert np.linalg.norm((op-a['operator']).astype(float))/np.linalg.norm(a['operator'].astype(float))<3e-6
        if operator is None:operator=a['operator'].copy()
        else:assert np.array_equal(operator,a['operator'])
        pos=r['position'];assert pos==7 and r['prompt_tokens'][pos]==' is'
        r['contribution_l2']=float(np.linalg.norm((a['gate'][pos]*a['operator'][pos]).astype(float)))
        if r['target_gate'] is None:continue
        assert r['selected_input_positions']==[7] and r['generated_positions_edited']==[]
        assert r['edit_calls']==1 and r['generated_passes_without_edit']==len(r['steps'])-1
        delta=np.zeros_like(a['native_mlp']);delta[pos]=np.float32(r['alpha'])*(np.float32(r['target_gate'])-a['gate'][pos])*a['operator'][pos]
        assert np.array_equal(delta,a['intended_delta'])
        changed=a['native_mlp'].copy();changed[pos]=bf16(changed[pos]+delta[pos])
        assert np.array_equal(changed,a['edited_mlp']) and np.array_equal(changed[:pos],a['native_mlp'][:pos])
        assert np.array_equal(a['actual_delta'],changed-a['native_mlp'])
        r['intended_delta_l2']=float(np.linalg.norm(delta[pos].astype(float)))
        r['actual_delta_l2']=float(np.linalg.norm(a['actual_delta'][pos].astype(float)));edits+=1
native=d['controls']['native']
for key in ['observed','zero_strength','natural_gate_replay']:
    assert d['controls'][key]['steps']==native['steps'] and d['controls'][key]['candidate_scores']==native['candidate_scores']
assert d['repeats'][0]['steps']==d['runs'][0]['steps'] and d['repeats'][0]['candidate_scores']==d['runs'][0]['candidate_scores']
protocol=d['protocol'];assert native['output_token_ids']==protocol['reference_native_first_32_tokens']
reference=ROOT/protocol['reference_controls_path'];assert sha(reference)==protocol['reference_controls_sha256']
old=json.loads(reference.read_text());old_native=next(r for r in old['controls'] if r['case_id']=='month')['ordinals']['12']['native']
assert native['prompt_token_ids']==old_native['prompt_token_ids']
for text,tid in d['probe_token_ids'].items():
    if len(tid)==1 and str(tid[0]) in old_native['steps'][0]['probe_probabilities']:
        assert abs(native['candidate_scores'][text]['probability']-old_native['steps'][0]['probe_probabilities'][str(tid[0])])<2e-6
months=['january','february','march','april','may','june','july','august','september','october','november','december']
def record(r):
    r=dict(r);r['month_probabilities']={m:sum(r['candidate_scores'][' '+s]['probability'] for s in [m,m.title()]) for m in months}
    r['initial_word']=r['response'].lstrip().split()[0].strip('.,:;!?');return r
native=record(native);off=record(d['runs'][0]);observed=d['controls']['observed']
validation={'status':'pass','arrays_audited':len(rows),'edit_arrays_audited':edits,'probability_max_absolute_error':max_error,
            'native_observer_zero_and_replay_exact':True,'ablation_repeat_exact':True,'native_prior_first_32_tokens_exact':True,
            'native_prior_probe_probabilities_match':True,'operator_same_across_conditions':True,'bf16_edits_exact':True,
            'only_final_input_edited':True,'no_generated_token_edits':True,'feature_hash_verified':True}
summary={'feature':'L29-T1181','protocol':protocol,'native':native,'observed':observed,'gate_off':off,
         'p_december_change':off['month_probabilities']['december']-native['month_probabilities']['december'],
         'runtime':d['runtime'],'probability_definition':d['probability_definition'],'validation':validation,'source_results_sha256':sha(p)}
summary['output_token_sequence_unchanged']=native['output_token_ids']==off['output_token_ids']
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
with (OUT/'month-probabilities.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['month','native_probability','T1181_off_probability'])
    for m in months:wr.writerow([m,native['month_probabilities'][m],off['month_probabilities'][m]])
lines=['# T1181 off on the genuine twelfth-month prompt','',
       f'December-prefix probability changes from {100*native["month_probabilities"]["december"]:.6f}% to {100*off["month_probabilities"]["december"]:.6f}%. Initial answers: {native["initial_word"]} to {off["initial_word"]}. All generated token IDs unchanged: {summary["output_token_sequence_unchanged"]}.',
       '',
       'Prompt: `the twelfth month of the year is`. Remove only L29–T1181 at final input position 7 (` is`). No other MOLT or generated token edited.',
       '',f'Natural gate: {observed["natural_gate"]:.9f}. Natural contribution L2: {off["contribution_l2"]:.6f}. Actual BF16-rounded edit L2: {off["actual_delta_l2"]:.6f}.',
       '','| Condition | Initial word | P(December) | P(November) | P(January) |','|---|---|---:|---:|---:|']
for label,r in [('Native',native),('T1181 off',off)]:lines.append(f'| {label} | {r["initial_word"]} | {100*r["month_probabilities"]["december"]:.6f}% | {100*r["month_probabilities"]["november"]:.6f}% | {100*r["month_probabilities"]["january"]:.6f}% |')
lines+=['','Probabilities sum exact leading-space lowercase/title-case answer prefixes, multiplying token probabilities for multi-token names. Other answer formats and following punctuation/EOS excluded. All twelve month probabilities are in month-probabilities.csv.',
        '','## Full native continuation','','```text',native['response'],'```','',
        '## Full T1181-off continuation','','```text',off['response'],'```','',
        '## Checks and limits','',
        'Pinned native BF16 Qwen3-4B; FP32 gate/operator and subtraction; one BF16 cast; normal downstream computation and normalization; eager attention; TF32 off; greedy 32-token continuations. The selected contribution retains checkpoint input/output standardizers and receives no extra normalization.',
        '','Native observation, zero-strength and exact natural-gate replay reproduce the native run bit-exactly in recorded full-vocabulary logit hashes and tokens. Native output reproduces the earlier first 32 tokens. Repeated ablation matches exactly. CPU auditing reconstructs probabilities, gate/operator values, selected-position masks and exact BF16 edits. Raw archive/files are hash-verified.',
        '','One exact raw completion prefix; this does not test numeric `12th` wording, all-input/output edits, or joint removal with T2375.',
        '','```json',json.dumps(validation,indent=2),'```','',
        'Reproduce: `python3 experiments/t1181-twelfth-month-off-20261103/analyze.py`. Raw evidence: `comparisons/2026-11-03-t1181-twelfth-month-off/`. No publication or GPU cleanup was performed.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'native_response':native['response'],'off_response':off['response'],'gate':observed['natural_gate'],'contribution_l2':off['contribution_l2'],'native_probabilities':native['month_probabilities'],'off_probabilities':off['month_probabilities'],'validation':validation},indent=2))
