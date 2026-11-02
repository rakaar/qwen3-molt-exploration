"""CPU reconstruction of recorded gates, operators, edits, masks and BF16 rounding."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def load(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bf16(x):
    a=np.ascontiguousarray(x,dtype=np.float32);u=a.view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def audit(root,controls_only=False):
    out=root/'artifacts';p=load(root/'protocol.json');cases={c['id']:c for c in load(root/'cases.json')}
    manifest=load(root/'weights/manifest.json');feature=manifest['features'][0]
    assert (feature['layer'],feature['transform'])==(29,2375)
    assert manifest['revision']==p['molt_revision']
    weights=root/'weights'/feature['file'];assert sha(weights)==feature['sha256']
    with np.load(weights,allow_pickle=False) as a:w={k:a[k] for k in a.files}
    ctl=load(out/'controls.json');assert ctl['status']=='complete' and ctl['hooks_remaining']==0
    for f,h in ctl['source_sha256'].items():assert sha(root/f)==h,f
    by_id={r['case_id']:r for r in ctl['controls']};assert set(by_id)==set(cases)
    runs=[];zero_pairs=0;noops=0
    for cid,row in by_id.items():
        for ordinal in ['11','12']:
            r=row['ordinals'][ordinal];native=r['native'];runs.append(native)
            assert native['prompt']==cases[cid]['prompts'][ordinal]
            assert hashlib.sha256(native['answer'].encode()).hexdigest()==r['answer_sha256']
            if ordinal=='11':
                zero=r['zero_strength'];runs.append(zero)
                assert native['steps']==zero['steps'] and native['answer']==zero['answer'] and native['output_token_ids']==zero['output_token_ids']
                assert zero['alpha']==0 and all(t['actual_delta_norm']==0 for t in zero['trace'])
                zero_pairs+=1
    if not controls_only:
        result=load(out/'results.json');review=load(root/'eligibility.json')
        assert result['status']=='complete' and result['hooks_remaining']==0
        assert result['controls_sha256']==review['controls_sha256']==sha(out/'controls.json')
        assert result['review']==review
        eligible={r['case_id'] for r in review['cases'] if r['eligible']}
        for r in review['cases']:
            assert r['eligible']==all(r['correct'].values()) and all(type(v) is bool for v in r['correct'].values())
            for n in ['11','12']:assert r['answer_sha256'][n]==by_id[r['case_id']]['ordinals'][n]['answer_sha256']
        expected={(cid,s) for cid in eligible for s in p['scopes']}
        assert {(r['case_id'],r['scope']) for r in result['runs']}==expected
        assert len(result['runs'])==len(expected)
        assert {r['case_id'] for r in result['specificity_controls']}==eligible
        assert len(result['specificity_controls'])==len(eligible)
        for r in result['runs']+result['specificity_controls']:
            native=by_id[r['case_id']]['ordinals'][str(r['ordinal'])]['native']
            assert r['prompt']==native['prompt'] and r['prompt_token_ids']==native['prompt_token_ids']
            n=len(r['prompt_token_ids']);scope=r['scope']
            spec={'last_input_only':{'kind':'positions','positions':[n-1]},'input_only':{'kind':'input'},'all_positions':{'kind':'all'},'output_only':{'kind':'output'}}[scope]
            assert r['spec']==spec and r['target']==0 and r['alpha']==1
            if scope=='output_only':assert r['steps'][0]==native['steps'][0]
            if not any(t['actual_delta_norm'] for t in r['trace']):
                assert r['steps']==native['steps'] and r['output_token_ids']==native['output_token_ids']
                noops+=1
            runs.append(r)
    vectors=set();positions=0;max_encoder=0.;max_operator=0.;max_probability=0.
    for r in runs:
        assert r['hooks_remaining']==0
        for step in r['steps']:
            error=abs(math.exp(step['selected_logit']-step['logsumexp'])-step['selected_probability'])
            max_probability=max(max_probability,error);assert error<2e-5
        if r['spec'] is None:assert not r['trace'];continue
        n=len(r['prompt_token_ids']);assert len(r['trace'])==n+len(r['output_token_ids'])-1
        assert [t['position'] for t in r['trace']]==list(range(len(r['trace'])))
        for t in r['trace']:
            pos=t['position'];phase='input' if pos<n else 'output';spec=r['spec']
            selected=spec['kind']=='all' or spec['kind']==phase or (spec['kind']=='positions' and pos in spec['positions'])
            assert t['phase']==phase and t['selected']==selected and t['natural_gate']>=0
            if not selected:assert t['actual_delta_norm']==0
        for v in r['vectors']:
            path=out/v['file'];assert sha(path)==v['sha256']
            assert v['file'] not in vectors;vectors.add(v['file'])
            with np.load(path,allow_pickle=False) as a:
                z=a['z'];raw=z@w['e'].reshape(-1)+w['bias'].reshape(-1)[0]
                error=float(np.max(np.abs(raw-a['raw'])));max_encoder=max(max_encoder,error)
                assert np.allclose(raw,a['raw'],atol=2e-5,rtol=2e-5)
                gate=np.where((raw>w['theta'].reshape(-1)[0])&(raw>0),raw,0)
                assert np.allclose(gate,a['gate'],atol=2e-5,rtol=2e-5)
                op=(z@w['V']@w['U'])*w['output_std']
                relative=float(np.linalg.norm(op-a['operator'])/max(np.linalg.norm(a['operator']),1e-12))
                max_operator=max(max_operator,relative);assert relative<2e-5
                delta=np.float32(r['alpha'])*(a['fixed']-a['gate'])[...,None]*a['operator']
                delta=np.where(a['selected'][...,None],delta,np.float32(0))
                assert np.array_equal(delta,a['intended_delta'])
                changed=bf16(a['native_mlp']+delta)
                assert np.array_equal(changed,a['edited_mlp'])
                assert np.array_equal(changed-a['native_mlp'],a['actual_delta'])
                assert np.array_equal(a['native_mlp'][~a['selected']],a['edited_mlp'][~a['selected']])
                positions+=len(a['positions'])
    report={'status':'passed','device':'CPU','controls_only':controls_only,'runs':len(runs),'native_zero_pairs':zero_pairs,
        'exact_noop_comparisons':noops,'vector_files':len(vectors),'sampled_positions':positions,
        'maximum_encoder_absolute_error':max_encoder,'maximum_operator_relative_l2_error':max_operator,
        'maximum_probability_reconstruction_error':max_probability,'english_anchor_checks':ctl['english_anchor_checks'],
        'limits':['Saved vectors start from standardized residual z; this audit does not independently reconstruct pre-normalization residuals.','Vector samples do not independently reproduce the full model forward pass.','Manual native and ablated initial-answer judgments are separate from numerical auditing.']}
    (out/('audit-controls.json' if controls_only else 'audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args();print(json.dumps(audit(args.root,args.controls_only),indent=2))
