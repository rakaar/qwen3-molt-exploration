"""Independent edit checks, prior-run reproduction, and factorial summary."""
import hashlib,html,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/'results/2026-10-19-ch-child-joint-ablation';DEST.mkdir(exist_ok=True)

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def rounded_bf16(a):
    u=np.asarray(a,dtype=np.float32).view(np.uint32)
    return (((u+(0x7fff+((u>>16)&1)))>>16)<<16).view(np.float32)

def main():
    data=json.loads((ROOT/'artifacts/results.json').read_text());assert data['status']=='complete' and len(data['rows'])==10
    weights={}
    for f in data['assets']['features']:
        p=ROOT/'weights'/f['file'];assert sha(p)==f['sha256']
        with np.load(p) as a:weights[f['layer']]={k:a[k] for k in a.files}
    audits=[];vectors={};summary=[];interactions=[];dynamic=[];reproduction=[]
    old_B=json.loads((ROOT.parents[1]/'results/2026-10-19-ch-child-input-ablation/summary.json').read_text())
    old_A=json.loads((ROOT.parent/'ch-alllayer-causal-screen/confirm/results.json').read_text())
    for r in data['rows']:
        cid,condition=r['case_id'],r['condition']['id']
        assert not r['free']['generated_tokens_edited'] and not r['forced']['generated_tokens_edited']
        for mode in ['free','forced']:
            for o in r[mode]['input_observations']:
                assert o['position']==27 and o['segment']=='input' and o['other_positions_byte_identical']
                p=ROOT/'artifacts'/o['vector_file'];assert sha(p)==o['vector_sha256']
                with np.load(p) as a:v={k:a[k] for k in a.files}
                w=weights[o['layer']];z=(v['input']-w['mean'])/w['std']
                raw=float(z@w['encoder']+w['bias'][0]);gate=raw if raw>w['threshold'][0] and raw>0 else 0.
                low=(z@w['V']@w['U'])*w['outstd'];delta=-o['removed_fraction']*gate*low
                assert abs(raw-o['raw'])<5e-5 and abs(gate-o['natural_gate'])<5e-5
                assert np.allclose(z,v['z'],atol=2e-6,rtol=2e-6)
                assert np.allclose(low,v['lowrank'],atol=5e-5,rtol=5e-4)
                assert np.allclose(delta,v['intended_delta'],atol=5e-5,rtol=5e-4)
                assert np.array_equal(rounded_bf16(v['native_mlp']+v['intended_delta']),v['edited_mlp'])
                assert np.array_equal(v['edited_mlp']-v['native_mlp'],v['actual_delta'])
                if o['removed_fraction']==1 and gate>0:assert o['actual_delta_norm']>0 and o['estimated_remaining_gate']==0
                if o['removed_fraction']==0:assert o['actual_delta_norm']==0
                vectors[(cid,condition,mode,o['layer'])]=v
                audits.append({'case_id':cid,'condition':condition,'mode':mode,'layer':o['layer'],
                    'CPU_delta_max_abs_error':float(np.max(np.abs(delta-v['intended_delta'])))})
        score=r['forced']['target_score'];obs={o['layer']:o for o in r['free']['input_observations']}
        summary.append({'case_id':cid,'condition':condition,'answer':r['free']['answer'],
            'output_token_ids':r['free']['output_token_ids'],'target_token':score['target_token'],
            'target_probability':score['target_probability'],'target_log_odds':score['target_log_odds'],
            'target_minus_foil_logit':score['target_minus_foil_logit'],'source_gates':{str(l):o['natural_gate'] for l,o in obs.items()},
            'actual_delta_norms':{str(l):o['actual_delta_norm'] for l,o in obs.items()}})
        if condition=='L3_only':
            old=next(x for x in old_B if x['case_id']==cid and x['condition']=='ablate_B')
            assert r['free']['answer']==old['answer'] and score['target_probability']==old['conditional_gold_letter_probability']
            reproduction.append({'case_id':cid,'condition':condition,'prior_probability_exact':True})
        if condition=='L28_only':
            old=next(x for x in old_A['rows'] if x['case_id']==cid and x['condition']['id']=='L28-T1123-source')
            assert r['free']['output_token_ids']==old['generation']['output_token_ids']
            assert score['target_probability']==old['score']['metrics']['target_probability']
            reproduction.append({'case_id':cid,'condition':condition,'prior_probability_exact':True})
    for cid in sorted({x['case_id'] for x in summary}):
        rows={r['condition']['id']:r for r in data['rows'] if r['case_id']==cid}
        assert rows['native']['free']['scores']==rows['zero_control']['free']['scores']
        assert rows['native']['forced']['scores']==rows['zero_control']['forced']['scores']
        for layer in [3,28]:
            for condition in ['zero_control','L3_only','L28_only','joint']:
                assert np.array_equal(vectors[cid,condition,'free',layer]['input'],vectors[cid,condition,'forced',layer]['input'])
        for layer in [3,28]:
            assert np.array_equal(vectors[cid,'L3_only','free',layer]['input'],vectors[cid,'joint','free',layer]['input'])
        assert np.array_equal(vectors[cid,'zero_control','free',28]['input'],vectors[cid,'L28_only','free',28]['input'])
        for condition in ['L3_only','L28_only','joint']:
            assert np.array_equal(vectors[cid,'zero_control','free',3]['input'],vectors[cid,condition,'free',3]['input'])
        a=vectors[cid,'zero_control','free',28];b=vectors[cid,'joint','free',28]
        dynamic.append({'case_id':cid,'L28_native_gate':float(a['gate'][0]),'L28_gate_after_L3_ablation':float(b['gate'][0]),
            'L28_current_input_change_L2':float(np.linalg.norm(b['input']-a['input'])),'joint_uses_current_L28_input':True})
        def score(c,key):return rows[c]['forced']['target_score'][key]
        interactions.append({'case_id':cid,
            'joint_minus_additive_effect_log_odds':score('joint','target_log_odds')-score('L3_only','target_log_odds')-score('L28_only','target_log_odds')+score('native','target_log_odds'),
            'joint_minus_additive_effect_target_foil_margin':score('joint','target_minus_foil_logit')-score('L3_only','target_minus_foil_logit')-score('L28_only','target_minus_foil_logit')+score('native','target_minus_foil_logit')})
    assert len(audits)==32 and len(reproduction)==4
    verification={'all_10_conditions_complete':True,'all_32_source_observations_independently_verified':True,
        'all_actual_BF16_edits_exact':True,'no_generated_tokens_edited':True,
        'zero_control_scores_bit_exact':True,'all_four_prior_single_ablation_results_exact':True,
        'dynamic_L28_checks':dynamic,'prior_reproduction':reproduction,'edit_checks':audits}
    (DEST/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (DEST/'summary.json').write_text(json.dumps({'rows':summary,'dynamic_L28':dynamic,'factorial_interactions':interactions},indent=2)+'\n')
    body=''
    for cid in ['fewshot_upper_child','fewshot_lower_child']:
        body+=f'<h2>{html.escape(cid)}</h2><table><tr><th>Condition</th><th>Full greedy answer</th><th>Letter probability</th><th>Letter vs foil logit margin</th><th>L28 natural gate before removal</th></tr>'
        for r in summary:
            if r['case_id']!=cid:continue
            body+=f'<tr><td>{r["condition"]}</td><td>{html.escape(r["answer"])}</td><td>{r["target_probability"]:.6%}</td><td>{r["target_minus_foil_logit"]:.6f}</td><td>{r["source_gates"].get("28","—")}</td></tr>'
        body+='</table>'
    cases=json.loads((ROOT/'cases.json').read_text())
    prompt_html=''.join(f'<h3>{html.escape(c["id"])}</h3><pre>{html.escape(c["user"])}</pre>' for c in cases)
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>Joint Child / CH MOLT ablation</title>
    <style>body{{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.55 system-ui;color:#172236}}table{{border-collapse:collapse;width:100%;margin:22px 0;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #ddd}}th{{background:#edf2f8}}pre{{background:#f4f6f8;padding:18px;white-space:pre-wrap}}.note{{padding:16px;border-left:4px solid #3976b7;background:#edf5ff}}</style></head><body>
    <h1>Joint ablation of L3-T1372 and L28-T1123</h1><p>19 October 2026 · Exact two earlier few-shot tasks · Input child token only</p>
    <p class="note">Native Qwen3-4B retained. Subtract each selected current-trajectory estimated MOLT contribution at input token 27, at its own layer. L28 input, gate and transform output are recomputed after the L3 edit. Generated child tokens are unchanged.</p>
    {body}<p>Uppercase probability is P(CH) at the first answer token; the foil is CI. Lowercase probability is P( ch | the native generated prefix child:), evaluated in a separate cached teacher-forced replay; the foil is space + ci. Full free greedy answers are recorded independently.</p>
    <h2>Dynamic L28 state</h2><pre>{html.escape(json.dumps(dynamic,indent=2))}</pre>
    <h2>Exploratory interaction contrast</h2><p>Joint minus L3-only minus L28-only plus baseline, on log-odds and letter-versus-foil logit scales. A nonzero contrast is a finite perturbation interaction on this prompt; it is not proof of mechanistic cooperation or absorption.</p><pre>{html.escape(json.dumps(interactions,indent=2))}</pre>
    <h2>Exact prompts</h2>{prompt_html}
    <h2>Verification and limits</h2><p>Ten conditions completed. All four prior single-feature answer/probability results reproduce exactly. Zero-strength hooks and restored unhooked baselines reproduce native scores exactly. All 32 saved source observations independently reconstruct the gates, low-rank outputs and applied edits on CPU; BF16 rounding matches exactly. Downstream attention remains native.</p>
    <p>These remove approximate MOLT components from the native MLP while retaining its remainder. This pilot tests combined necessity on two prompts. Correct answers after removal do not disprove absorption, redundancy elsewhere, or information carried at other positions. In the lowercase task, the generated child token can supply information again because this test edits only the input word.</p>
    <p><a href="summary.json">Summary</a> · <a href="verification.json">Verification</a> · <a href="results.json">Full records</a> · <a href="protocol.json">Protocol</a></p></body></html>'''
    (DEST/'index.html').write_text(page)
    shutil.copy2(ROOT/'artifacts/results.json',DEST/'results.json');shutil.copy2(ROOT/'protocol.json',DEST/'protocol.json')
    manifest={p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'rows':summary,'dynamic_L28':dynamic,'interactions':interactions},indent=2))

if __name__=='__main__':main()
