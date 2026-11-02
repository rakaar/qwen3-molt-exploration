"""Independent CPU checks and complete report for the scalar-only sweep."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'comparisons/2026-11-03-eleven-to-five-scalar/remote-export/eleven-to-five-scalar-20261103'
OUT=ROOT/'results/2026-11-03-eleven-to-five-scalar'
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
assert p['scalar_only'] and not p['copy_donor_operator'] and not p['copy_donor_contribution']
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
all_runs=[d['controls'][label][kind] for label in ['donor','recipient'] for kind in ['native','observed']]+[d['controls']['zero_strength']]+d['runs']+d['repeats']
encoder_error,operator_error,prob_error,audited_edits=0.,0.,0.,0
recipient_operator=None
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
        delta[6]=np.float32(row['alpha'])*(np.float32(row['target_gate'])-a['gate'][6])*a['operator'][6]
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

repeat_path=ROOT/'comparisons/2026-11-03-eleven-to-five-scalar-repeat/remote-export/eleven-to-five-scalar-repeat-20261103/artifacts/results.json'
independent_repeat=None
if repeat_path.exists():
    repeated=json.loads(repeat_path.read_text())
    assert repeated['status']=='complete' and repeated['hooks_remaining']==0
    assert sha(repeat_path.parent.parent/'run.py')==repeated['runtime']['source_sha256']
    assert sha(repeat_path.parent.parent/'protocol.json')==repeated['runtime']['protocol_sha256']
    assert repeated['runtime']['source_sha256']==d['runtime']['source_sha256']
    assert repeated['protocol']['recipient_prompt']==p['recipient_prompt']
    assert repeated['protocol']['scalar_only'] and not repeated['protocol']['copy_donor_operator']
    for row in repeated['runs']+repeated['repeats']:
        original=next(r for r in d['runs'] if r['target_gate']==row['target_gate'])
        assert row['steps']==original['steps'] and row['output_token_ids']==original['output_token_ids']
        assert row['selected_input_positions']==[6] and row['generated_positions_edited']==[]
        assert sha(repeat_path.parent/row['arrays']['file'])==row['arrays']['sha256']
    independent_repeat={'status':'pass','model_reloaded':True,'gates':[r['target_gate'] for r in repeated['runs']],
                        'all_token_ids_and_full_logit_hashes_exact':True,'source':str(repeat_path.relative_to(ROOT)),
                        'source_results_sha256':sha(repeat_path)}


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
successes=[r['target_gate'] for r in rows if r['initial_answer_november']]
validation={'status':'pass','sweep_conditions':len(rows),'array_files_checked':len(all_runs),
            'audited_edit_arrays':audited_edits,'encoder_max_absolute_error':encoder_error,
            'operator_max_relative_l2_error':operator_error,'probability_max_absolute_error':prob_error,
            'recipient_operator_identical_across_gates':True,'unselected_local_outputs_bit_exact':True,
            'bf16_edits_exact':True,'native_observer_and_zero_controls_exact':True,
            'donor_and_maximum_gate_repeats_exact':True,'no_generated_token_edits':True,'hooks_remaining':0}
validation['independent_model_reload_repetition_exact']=independent_repeat is not None
report={'feature':'L29-T2375','date_local':'2026-11-03','protocol':p,'native':native,
        'donor_gate_copy':donor,'runs':rows,'november_initial_answer_gates':successes,
        'lowest_successful_gate_tested':min(successes) if successes else None,
        'probability_definition':d['probability_definition'],
        'case_sum_definition':'Sum of the two case-sensitive leading-space next-token events: lowercase and title case. No answer-ending punctuation/EOS probability is included.',
        'runtime':d['runtime'],'validation':validation,'independent_repetition':independent_repeat,'source_results_sha256':sha(path)}
(OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
fields=['target_gate','initial_word','p_may_exact','p_november_exact','p_may_case_sum','p_november_case_sum','intended_delta_l2','actual_delta_l2','stop_reason','response']
with (OUT/'sweep.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:row[k] for k in fields})
lines=['# Eleventh-to-fifth scalar gate transplant and sweep','','Prompt: `'+p['recipient_prompt']+'`.',
       '',f'Donor: `{p["donor_prompt"]}`; natural final-input gate {p["donor_gate"]:.9f}.',
       '','Only recipient input position 6 (` is`) is edited. The recipient’s own current UVx is retained; neither donor operator nor donor contribution is transferred. No generated token is edited.',
       '','Native initial answer: **'+native['initial_word']+'**. Natural recipient gate: 0.',
       '',f'Copying the donor scalar gives initial answer **{donor["initial_word"]}**.',
       '',f'November-initial gates: {successes}. These are values tested, not a global threshold.',
       '','| Fixed gate | Initial word | P(May variants) | P(November variants) | Stop |','|---:|---|---:|---:|---|']
for r in rows:lines.append(f'| {r["target_gate"]:.6f} | {r["initial_word"]} | {100*r["p_may_case_sum"]:.6f}% | {100*r["p_november_case_sum"]:.6f}% | {r["stop_reason"]} |')
lines+=['','Probabilities sum the lowercase and title-case leading-space one-token forms: ` may` plus ` May`, and ` november` plus ` November`. Following punctuation/EOS is not included. These events do not cover answers beginning with other text such as ` the month of may`. All four exact case-sensitive probabilities are retained in summary.json; lowercase probabilities and case sums are in sweep.csv.','',
        '## Full recorded responses','']
for r in rows:lines += [f'### Gate {r["target_gate"]:.9f}','','```text',r['response'],'```','']
lines+=['## Checks and limits','',
        'Pinned native BF16 Qwen3-4B; FP32 gate/operator and contribution edit; one BF16 cast; eager attention; TF32 off; raw lowercase prefixes; greedy decoding; 32 generated-token cap.',
        '','Native donor and recipient responses reproduce their previously saved first 12 tokens. Native/observer, zero-strength and gate-zero controls match exactly. Donor-scalar and maximum-gate repetitions match exactly. An independent CPU audit verifies hashes, gates, operators, full recipient-operator identity, intervention mask, exact BF16 edits and probability reconstruction.',
        '','This is one prompt and one feature. A high-gate initial-word flip is not a guarantee of coherent full continuation, ordinary-strength sufficiency or broad ordinal transfer. Full continuations and failures are preserved.',
        '','Reproduce local analysis: `python3 experiments/eleven-to-five-scalar-20261103/analyze.py`. Scientific evidence: `comparisons/2026-11-03-eleven-to-five-scalar/`.']
if independent_repeat:
    lines+=['','An independent follow-up reloaded the model and repeated gates 0, 1.158553600 and 20. All generated token IDs, every recorded full-vocabulary logit hash and the gate-20 self-correction reproduced exactly. This follow-up used no new strengths.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'donor_copy_response':donor['response'],'november_initial_gates':successes,
                  'rows':[{k:r[k] for k in ['target_gate','initial_word','p_may_exact','p_november_exact']} for r in rows],
                  'validation':validation},indent=2,ensure_ascii=False))
