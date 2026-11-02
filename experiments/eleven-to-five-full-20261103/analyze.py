"""Independent CPU checks for donor-operator and scalar transplantation."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'comparisons/2026-11-03-eleven-to-five-full/remote-export/eleven-to-five-full-20261103'
OUT=ROOT/'results/2026-11-03-eleven-to-five-full'
OUT.mkdir(exist_ok=True)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bf16(value):
    bits=np.asarray(value,dtype=np.float32).view(np.uint32)
    return ((bits+(0x7fff+((bits>>16)&1)))&0xffff0000).view(np.float32)


path=SOURCE/'artifacts/results.json'
d=json.loads(path.read_text())
assert d['status']=='complete' and d['hooks_remaining']==0
assert sha(SOURCE/'run.py')==d['runtime']['source_sha256']
assert sha(SOURCE/'protocol.json')==d['runtime']['protocol_sha256']
p=d['protocol']
assert not p['scalar_only'] and p['copy_donor_operator'] and p['copy_donor_contribution']
assert [r['target_gate'] for r in d['runs']]==p['fixed_gates']
wp=ROOT/'experiments/eleven-activation-20261102/weights/L29-T2375.npz'
assert sha(wp)==d['runtime']['feature_sha256']
with np.load(wp,allow_pickle=False) as a:w={k:a[k].copy() for k in a.files}
native=d['controls']['recipient']['native']
assert d['controls']['zero_strength']['steps']==native['steps']
assert d['controls']['recipient']['observed']['steps']==native['steps']
assert d['controls']['donor']['observed']['steps']==d['controls']['donor']['native']['steps']
assert d['donor_gate_reproduced_exactly']
assert next(r for r in d['runs'] if r['target_gate']==0)['steps']==native['steps']
all_runs=[d['controls'][label][kind] for label in ['donor','recipient'] for kind in ['native','observed']]+[d['controls']['zero_strength']]+[d['controls']['scalar_only_same_gate']]+d['runs']+d['repeats']+d['norm_matched_controls']
encoder_error,operator_error,prob_error,audited_edits=0.,0.,0.,0
recipient_operator=None
with np.load(SOURCE/'artifacts'/d['controls']['donor']['observed']['arrays']['file'],allow_pickle=False) as a:
    donor_operator=a['operator'][7].copy()
for row in all_runs:
    ap=SOURCE/'artifacts'/row['arrays']['file']
    assert sha(ap)==row['arrays']['sha256']
    with np.load(ap,allow_pickle=False) as a:
        logits=a['first_logits']
        assert hashlib.sha256(logits.tobytes()).hexdigest()==row['steps'][0]['logits_sha256']
        assert int(logits.argmax())==row['steps'][0]['token_id']
        probs=np.exp(logits.astype(np.float64)-float(logits.max()));probs/=probs.sum()
        for text,tid in d['probe_token_ids'].items():
            error=abs(float(probs[tid])-row['first_token_probabilities'][text])
            assert error<2e-6
            prob_error=max(prob_error,error)
        if 'z' not in a:continue
        z=(a['residual']-w['input_mean'])/w['input_std']
        assert np.allclose(z,a['z'],atol=1e-6,rtol=1e-6)
        raw=z@w['e'].reshape(-1)+w['bias'].reshape(-1)[0]
        error=float(np.max(np.abs(raw-a['raw'])));assert error<1e-5
        encoder_error=max(encoder_error,error)
        gate=np.where((raw>w['theta'].reshape(-1)[0])&(raw>0),raw,0.)
        assert np.allclose(gate,a['gate'],atol=1e-5,rtol=1e-5)
        op=(z@w['V']@w['U'])*w['output_std']
        error=float(np.linalg.norm((op-a['operator']).astype(np.float64))/np.linalg.norm(a['operator'].astype(np.float64)))
        assert error<2e-6
        operator_error=max(operator_error,error)
        if row['prompt']==p['recipient_prompt']:
            if recipient_operator is None:recipient_operator=a['operator'].copy()
            else:assert np.array_equal(recipient_operator,a['operator']),'Recipient operator changed across targets'
        if row['target_gate'] is None:continue
        assert row['position']==6 and row['token']==' is' and row['selected_input_positions']==[6]
        assert row['generated_positions_edited']==[] and row['edit_calls']==1
        assert row['generated_passes_without_edit']==len(row['steps'])-1
        delta=np.zeros_like(a['native_mlp'],dtype=np.float32)
        assert np.array_equal(a['frozen_donor_operator'],donor_operator)
        replacement=donor_operator if row['operator_source']=='donor' else a['operator'][6]
        assert np.array_equal(a['replacement_operator'],replacement)
        delta[6]=np.float32(row['alpha'])*(np.float32(row['target_gate'])*replacement-a['gate'][6]*a['operator'][6])
        assert np.array_equal(delta,a['intended_delta'])
        changed=a['native_mlp'].copy();changed[6]=bf16(changed[6]+delta[6])
        assert np.array_equal(changed,a['edited_mlp'])
        assert np.array_equal(a['actual_delta'],a['edited_mlp']-a['native_mlp'])
        assert np.array_equal(changed[:6],a['native_mlp'][:6])
        row['intended_delta_l2']=float(np.linalg.norm(delta[6].astype(np.float64)))
        row['actual_delta_l2']=float(np.linalg.norm(a['actual_delta'][6].astype(np.float64)))
        audited_edits+=1
for repeat in d['repeats']:
    original=next(r for r in d['runs'] if r['target_gate']==repeat['target_gate'])
    assert repeat['steps']==original['steps'] and repeat['output_token_ids']==original['output_token_ids']

independent_repeat=None


def record(row):
    r=dict(row)
    first=re.match(r'\s*([A-Za-z]+|[0-9]+)',r['response'])
    r['initial_word']=first.group(1) if first else r['steps'][0]['token']
    r['initial_answer_november']=r['initial_word'].lower()=='november'
    pr=r['first_token_probabilities']
    r['p_may_exact']=pr[' may'];r['p_november_exact']=pr[' november']
    r['p_may_case_sum']=pr[' may']+pr[' May'];r['p_november_case_sum']=pr[' november']+pr[' November']
    return r


rows=[record(row) for row in d['runs']]
native=record(native)
donor=next(r for r in rows if r['target_gate']==p['donor_gate'])
with np.load(SOURCE/'artifacts'/donor['arrays']['file'],allow_pickle=False) as a:
    assert np.array_equal(a['intended_delta'][6],np.float32(p['donor_gate'])*donor_operator)
    with np.load(ROOT/'results/2026-11-03-eleven-contribution-similarity/contribution-vectors.npz',allow_pickle=False) as previous:
        i=list(previous['names']).index('Month')
        assert np.array_equal(a['intended_delta'][6],previous['contributions'][i].astype(np.float32))
scalar_copy=record(d['controls']['scalar_only_same_gate'])
matched=[record(r) for r in d['norm_matched_controls']]
for r in matched:
    donor_row=next(a for a in rows if a['target_gate']==r['matched_donor_gate'])
    assert abs(r['intended_delta_l2']-donor_row['intended_delta_l2'])/donor_row['intended_delta_l2']<1e-6
prior_path=ROOT/'results/2026-11-03-eleven-to-five-scalar/summary.json'
prior=json.loads(prior_path.read_text())
assert native['steps']==prior['native']['steps']
assert scalar_copy['steps']==prior['donor_gate_copy']['steps']
comparison=[]
for row in rows:
    scalar=next(r for r in prior['runs'] if r['target_gate']==row['target_gate'])
    comparison.append({'gate':row['target_gate'],'full_initial_word':row['initial_word'],
                       'full_p_november_case_sum':row['p_november_case_sum'],
                       'full_p_may_case_sum':row['p_may_case_sum'],
                       'full_intended_delta_l2':row['intended_delta_l2'],
                       'scalar_initial_word':scalar['initial_word'],
                       'scalar_p_november_case_sum':scalar['p_november_case_sum'],
                       'scalar_intended_delta_l2':scalar['intended_delta_l2']})
successes=[r['target_gate'] for r in rows if r['initial_answer_november']]
validation={'status':'pass','donor_operator_conditions':len(rows),'array_files_checked':len(all_runs),
            'audited_edit_arrays':audited_edits,'encoder_max_absolute_error':encoder_error,
            'operator_max_relative_l2_error':operator_error,'probability_max_absolute_error':prob_error,
            'recipient_operator_identical_across_gates':True,'replacement_operator_matches_native_donor_exactly':True,
            'natural_patch_equals_saved_donor_full_contribution_exactly':True,
            'unselected_local_outputs_bit_exact':True,'bf16_edits_exact':True,
            'native_observer_and_zero_controls_exact':True,'natural_first_success_and_maximum_repeats_exact':True,
            'norm_matched_controls_checked':len(matched),'scalar_copy_reproduces_previous_experiment_exactly':True,
            'no_generated_token_edits':True,'hooks_remaining':0}
report={'feature':'L29-T2375','date_local':'2026-11-03','protocol':p,'native':native,
        'natural_full_contribution_copy':donor,'scalar_only_same_gate':scalar_copy,
        'runs':rows,'norm_matched_controls':matched,'operator_norm_ratio':d['operator_norm_ratio'],
        'scalar_only_comparison':comparison,'prior_scalar_summary_sha256':sha(prior_path),
        'november_initial_answer_gates':successes,'lowest_successful_gate_tested':min(successes) if successes else None,
        'probability_definition':'Conditional next-token probabilities given the original raw prompt and selected edit, before any output prefix. Exact strings include a leading space and exclude following punctuation/EOS.',
        'case_sum_definition':'Sum of lowercase and title-case leading-space next-token events, without punctuation/EOS. Does not include other possible answer formats.',
        'runtime':d['runtime'],'validation':validation,'source_results_sha256':sha(path)}
(OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
fields=['target_gate','operator_source','initial_word','p_may_exact','p_november_exact','p_may_case_sum','p_november_case_sum','intended_delta_l2','actual_delta_l2','stop_reason','response']
with (OUT/'sweep.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:row[k] for k in fields})
with (OUT/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
lines=['# Eleventh-to-fifth full MOLT contribution transplant','',
       'Recipient: `'+p['recipient_prompt']+'`. Donor: `'+p['donor_prompt']+'`.',
       '',f'Natural donor gate: {p["donor_gate"]:.9f}. The donor UVx is frozen at the donor final input token and used unchanged in every transplant condition.',
       '',"At recipient input position 6 (` is`), replace only this MOLT contribution with target_gate * donor_UVx. Retain the native recipient MLP remainder. No generated-token edit.",
       '',f'The exact natural full contribution gives initial word **{donor["initial_word"]}**, versus **{scalar_copy["initial_word"]}** for the scalar-only donor-value control.',
       '',f'Initial-November gates tested: {successes}. Lowest successful value tested: {min(successes) if successes else None}. This is not a global threshold.',
       '','| Donor-operator gate | Initial word | P(May variants) | P(November variants) | Contribution L2 |',
       '|---:|---|---:|---:|---:|']
for r in rows:lines.append(f'| {r["target_gate"]:.6f} | {r["initial_word"]} | {100*r["p_may_case_sum"]:.6f}% | {100*r["p_november_case_sum"]:.6f}% | {r["intended_delta_l2"]:.6f} |')
lines+=['','The probabilities sum two exact one-token spellings per month: ` may` plus ` May`, and ` november` plus ` November`. They exclude following punctuation/EOS and do not cover answers beginning with other words.',
        '','## Equal-norm direction controls','',f'The donor operator is {d["operator_norm_ratio"]:.6f} times the norm of the recipient operator. These controls use recipient UVx at an adjusted scalar to match the donor-patch intended vector norm.']
for r in matched:
    corresponding=next(a for a in rows if a['target_gate']==r['matched_donor_gate'])
    lines += ['',f'Donor gate {r["matched_donor_gate"]:.9f} versus recipient gate {r["target_gate"]:.9f}; intended L2 {r["intended_delta_l2"]:.6f}.',
              '',f'Donor-operator response (P(November variants) {100*corresponding["p_november_case_sum"]:.6f}%):','', '```text',corresponding['response'],'```',
              '',f'Equal-norm recipient-operator response (P(November variants) {100*r["p_november_case_sum"]:.6f}%):','','```text',r['response'],'```']
lines += ['','## Every donor-operator continuation','']
for r in rows:lines += [f'### Gate {r["target_gate"]:.9f}','','```text',r['response'],'```','']
lines += ['## Verification and limits','',
          'Hash-verified pinned Qwen3-4B and MOLT weights; BF16 native model; FP32 contribution edits and one BF16 cast; eager attention; TF32 off; greedy decoding; 32-token limit.',
          '', 'The donor gate and native response controls reproduce saved references. Native/observer and zero controls are exact. The scalar-only donor-value control reproduces the preceding experiment exactly. Natural donor value, first initial-November value if any, and maximum gate are repeated exactly.',
          '', 'CPU reconstruction verifies gates, operators, frozen donor operator identity, mask, actual BF16 edits, probability calculations and equal intended norms. The recipient operator itself stays identical across conditions. All responses and source hashes are preserved.',
          '', 'This tests one prompt and one MOLT. Initial November is judged separately from subsequent self-correction or coherent full answers. Equal-norm controls match intended vectors before BF16 rounding; actual rounded edit norms are also retained.',
          '', 'Reproduce with `python3 experiments/eleven-to-five-full-20261103/analyze.py`. Evidence: `comparisons/2026-11-03-eleven-to-five-full/`.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'natural_full_copy_response':donor['response'],'natural_full_p_november':donor['p_november_case_sum'],
                  'initial_november_gates':successes,'matched_controls':[{'donor_gate':r['matched_donor_gate'],'scalar_gate':r['target_gate'],'response':r['response'],'p_november':r['p_november_case_sum']} for r in matched],
                  'rows':comparison,'validation':validation},indent=2,ensure_ascii=False))
