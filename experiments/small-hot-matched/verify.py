"""Independent CPU arithmetic audit of saved source vectors and applied edits."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'
torch.set_num_threads(4)
r=json.loads((OUT/'results.json').read_text())
p=r['protocol'];recipe=json.loads((ROOT/'assets/recipe.json').read_text())
sources=torch.load(OUT/'selected-contributions.pt',map_location='cpu',weights_only=True)
traces=torch.load(OUT/'intervention-traces.pt',map_location='cpu',weights_only=True)
rows={x['key']:x for x in r['trials'] if 'key' in x}
groups={'transcoder':{'small':[(f['layer'],f['feature']) for f in recipe['selection']['small (multilingual)']],
                      'hot':[(f['layer'],f['feature']) for f in recipe['selection']['hot (multilingual)']]},**p['molt_groups']}
checks=[]

def selected(case,group,l,i,which):return case['selected'][str((group,l,i))][which]

def expected(case,group,mode,s,component='combined',published=False):
    ds={}
    for role,pairs in groups[group].items():
        if component=='suppress_only' and role!='small':continue
        if component=='inject_only' and role!='hot':continue
        for l,i in pairs:
            a=selected(case,group,l,i,'recipient');b=selected(case,group,l,i,'donor')
            target=b['gate']
            if published:
                if role=='small':target=target*0
                else:target=torch.tensor(next(f['acts']['hot_en'] for f in recipe['selection']['hot (multilingual)'] if f['layer']==l and f['feature']==i))
            change=(target-a['gate'])*a['ungated'] if mode=='scalar' else target*b['ungated']-a['contribution']
            if component=='suppress_only':change=-a['contribution']
            ds[l]=ds.get(l,torch.zeros(2560))+s*change
    return ds

def norm(ds):return torch.stack([d.square().sum() for d in ds.values()]).sum().sqrt()

for key,record in traces.items():
    row=rows[key];case=sources[row['prompt_id']];group=row['group'];mode=row['mode']
    if group=='random_pair22':
        exact=expected(case,'pair22','full',row['strength'])
    elif group=='whole_mlp22':
        exact={22:case['donor_mlp_operand'][22].float()-case['recipient_mlp_operand'][22].float()}
    else:
        actual_group='transcoder' if group=='transcoder_published' else group
        exact=expected(case,actual_group,'scalar' if mode=='scalar' else 'full',row['strength'],row['component'],group=='transcoder_published')
        if mode=='norm_matched_full':
            factor=norm(expected(case,'transcoder','full',row['strength']))/norm(exact).clamp_min(1e-20)
            exact={l:d*factor for l,d in exact.items()}
    assert set(record)==set(exact),(key,set(record),set(exact))
    for l,x in record.items():
        positions=x['intended_positions']
        assert positions in [[],[case['position']]],(key,positions)
        assert x['outside_positions_exactly_unchanged']
        assert torch.equal(x['after'],x['before']+x['delta'].to(x['before'].dtype)),key
        if positions:
            if group=='random_pair22':
                torch.testing.assert_close(x['delta'][0].norm(),exact[l].norm(),rtol=3e-6,atol=2e-5)
            else:torch.testing.assert_close(x['delta'][0],exact[l],rtol=2e-5,atol=2e-5,msg=lambda msg:key+' '+msg)
        else:assert torch.count_nonzero(exact[l])==0,key
    checks.append({'key':key,'positions_verified':True,'written_output_arithmetic_exact':True,'delta_matches_independent_source_calculation':True})

# Recompute MOLT gates and low-rank outputs directly from tensor slices, without
# importing the runner or the upstream MOLT implementation.
manifest=json.loads((OUT/'download-manifest.json').read_text())
weights={s['layer']:load_file(ROOT/'weights'/s['file']) for s in manifest['selected_molt']}
max_gate_error=max_vector_error=0.
for case in sources.values():
    pos=case['position']
    for k,value in case['selected'].items():
        group,l,i=ast.literal_eval(k)
        if group=='transcoder':continue
        w=weights[l]
        for which in ['recipient','donor']:
            raw=case[which+'_prenorm'][l][pos].float()
            x=(raw-w['input_standardizer.mean'][l])/w['input_standardizer.std'][l]
            pre=(w['e.weight'][i]*x).sum()+w['e.bias'][i]
            g=pre if pre>max(float(w['nonlinearity.theta'][0,i]),0.) else pre*0
            uv=(x @ w[f'V_{i}']) @ w[f'U_{i}'];uv=uv*w['output_standardizer.std'][l]
            a=value[which]
            max_gate_error=max(max_gate_error,float((g-a['gate']).abs()))
            max_vector_error=max(max_vector_error,float((uv-a['ungated']).abs().max()))
            torch.testing.assert_close(g,a['gate'],rtol=3e-4,atol=3e-4)
            torch.testing.assert_close(uv,a['ungated'],rtol=3e-4,atol=3e-4)
            torch.testing.assert_close(a['contribution'],a['gate']*a['ungated'],rtol=0,atol=0)
assert all(v==0 for v in r['checks'].values())
assert r['status']=='completed'
report={'passed':True,'trace_count':len(checks),'checks':checks,
    'max_cpu_gate_difference':max_gate_error,'max_cpu_lowrank_vector_difference':max_vector_error,
    'zero_self_restoration_all_exact':True,'full_molt22_selected_weights_match_checkpoint':r['reconstruction22']['selected_UV_exact_match_full_checkpoint'],
    'scope':'CPU audit in a separate process; independent gate, low-rank projection, delta and BF16 output arithmetic. Random controls audited for positions and norm. No runner imports.'}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('INDEPENDENT_AUDIT_PASSED',len(checks),max_gate_error,max_vector_error)
