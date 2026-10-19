"""Audit natural gates on CPU, then plot the exact input-token measurements."""
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'remote-export';ART=RUN/'artifacts'
D=json.loads((ART/'measurements.json').read_text())
P=json.loads((RUN/'protocol.json').read_text())
assert D['status']=='complete' and len(D['rows'])==10
assert D['native_attention_and_mlp_unmodified'] and D['all_hooks_return_none']
assert D['all_prior_baselines_reproduced'] and D['all_replays_token_identical']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


for item in json.loads((RUN/'download-manifest.json').read_text()):
    path=RUN/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
for item in json.loads((ROOT/'source-manifest-before-execution.json').read_text()):
    assert sha(RUN/item['file'])==item['sha256']
for name,digest in D['source_sha256'].items():assert sha(RUN/name)==digest
assets={f"L{f['layer']}-T{f['transform']}":f for f in D['assets']['features']}
weights={}
for name,a in assets.items():
    path=RUN/'weights'/a['file'];assert sha(path)==a['sha256']
    with np.load(path) as arrays:weights[name]={k:arrays[k].copy() for k in arrays.files}
old=json.loads((ROOT.parent/'2026-10-19-ch-prefix-activation/remote-export/artifacts/measurements.json').read_text())
old_rows={r['id']:r for r in old['rows']}
audit=[];target_rows=[];all_rows=[]
for row in D['rows']:
    assert row['read_only_replay_matches_native_tokens'] and row['native_generation_unhooked']
    assert row['output_token_ids']==old_rows[row['id']]['output_token_ids']
    targets=[t for t in row['prompt_tokens'] if t['target_token']]
    assert [t['position'] for t in targets]==row['target_positions']
    assert ''.join(t['token'] for t in targets)==row['target']
    old_targets=[t for t in old_rows[row['id']]['prompt_tokens'] if t['target_token']]
    for t,prior in zip(targets,old_targets):
        for name in ['L3-T2334','L3-T1372']:
            assert abs(t['features'][name]['gate']-prior['features'][name]['gate'])<3e-5
    assert len(row['vector_files'])==7
    for vector in row['vector_files']:
        name=vector['feature'];w=weights[name];path=ART/vector['file']
        assert sha(path)==vector['sha256']
        with np.load(path) as arrays:v={k:arrays[k].copy() for k in arrays.files}
        z=(v['native_input']-w['input_mean'])/w['input_std']
        assert np.allclose(z,v['standardized_input'],atol=2e-6,rtol=3e-6)
        raw=z@w['e']+w['bias'][0]
        gate=np.where((raw>w['theta'][0])&(raw>0),raw,np.float32(0))
        assert np.allclose(raw,v['raw'],atol=3e-5,rtol=3e-5)
        assert np.allclose(gate,v['gate'],atol=3e-5,rtol=3e-5)
        assert (gate>0).tolist()==(v['gate']>0).tolist()
        assert v['positions'].tolist()==row['target_positions']
        for j,t in enumerate(targets):
            assert t['features'][name]['raw']==float(v['raw'][j])
            assert t['features'][name]['gate']==float(v['gate'][j])
        audit.append({'case_id':row['id'],'feature':name,'tokens':len(targets),
            'cpu_raw_max_abs_error':float(np.max(np.abs(raw-v['raw']))),
            'cpu_thresholded_gate_verified':True})
    for t in row['prompt_tokens']+row['answer_tokens']:
        for name,value in t['features'].items():
            rec={'case_id':row['id'],'task':'uppercase' if row['id'].startswith('fewshot_upper_') else 'lowercase',
                'word':row['target'],'segment':t['segment'],'position':t['position'],'token_id':t['token_id'],
                'token':t['token'],'target_token':t['target_token'],'feature':name,
                'layer':assets[name]['layer'],'raw':value['raw'],'theta':assets[name]['theta'],
                'gate':value['gate'],'active':value['gate']>0}
            all_rows.append(rec)
            if t['target_token']:target_rows.append(rec)

features=['L3-T2334','L3-T1372','L4-T153','L11-T2226','L29-T2334','L32-T1516','L34-T929']
feature_labels=['L3–T2334 · CH onset','L3–T1372 · Child','L4–T153 · ch / new',
    'L11–T2226 · Child / baby','L29–T2334 · CH onset','L32–T1516 · CH culinary','L34–T929 · CH prefix']
matrices={};column_labels=[];columns=[]
for task in ['uppercase','lowercase']:
    selected=[r for r in D['rows'] if ('uppercase' if r['id'].startswith('fewshot_upper_') else 'lowercase')==task]
    assert [r['target'] for r in selected]==P['target_words']
    matrix=[];current_cols=[]
    for r in selected:
        for t in r['prompt_tokens']:
            if not t['target_token']:continue
            current_cols.append((r['target'],t['token']))
    if columns:assert current_cols==columns
    else:
        columns=current_cols
        column_labels=[word if word==token else word+'\n['+token+']' for word,token in columns]
    for name in features:
        matrix.append([t['features'][name]['gate'] for r in selected for t in r['prompt_tokens'] if t['target_token']])
    matrices[task]=np.array(matrix)
assert columns==[('child','child'),('chin','chin'),('chair','chair'),('chief','chief'),('cheef','che'),('cheef','ef')]
vmax=max(float(m.max()) for m in matrices.values());assert vmax>0
cmap=LinearSegmentedColormap.from_list('gates',['#f1f3f5','#c8e8ef','#55afc1','#06657e'])
fig,axes=plt.subplots(1,2,figsize=(14.8,5.7),sharey=True,layout='constrained')
for ax,task in zip(axes,['uppercase','lowercase']):
    matrix=matrices[task]
    image=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
    ax.set_title('Few-shot '+task+' task',fontsize=14,pad=12)
    ax.set_xticks(np.arange(len(columns)),column_labels,fontsize=11)
    ax.set_yticks(np.arange(len(features)),feature_labels,fontsize=11)
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(features),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.6);ax.tick_params(which='minor',bottom=False,left=False)
    ax.axvline(3.5,color='#6c7b83',lw=1.1)
    for i in range(len(features)):
        for j in range(len(columns)):
            g=matrix[i,j];text='0' if g==0 else f'{g:.2f}'
            ax.text(j,i,text,ha='center',va='center',fontsize=11,color='white' if g>vmax*.63 else '#172f3b')
    for spine in ax.spines.values():spine.set_visible(False)
bar=fig.colorbar(image,ax=axes,shrink=.9,pad=.025);bar.set_label('Natural gate value',fontsize=11)
fig.suptitle('MOLT gates on the input CH-word tokens',fontsize=17)
fig.supxlabel('Each cell is one input token.  cheef = che + ef; shown separately.  No intervention.',fontsize=11)
fig.savefig(ROOT/'input-word-gates.png',dpi=180);fig.savefig(ROOT/'input-word-gates.svg');plt.close(fig)
for filename,records in [('input-word-gates.csv',target_rows),('all-token-gates.csv',all_rows)]:
    with (ROOT/filename).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
stats={name:{'label':assets[name]['label'],'source_tokens_active':sum(r['active'] for r in target_rows if r['feature']==name),
    'source_tokens_total':sum(r['feature']==name for r in target_rows),
    'max_source_gate':max(r['gate'] for r in target_rows if r['feature']==name),
    'child_gates':{task:float(matrices[task][features.index(name),0]) for task in matrices}}
    for name in features}
proof={'status':'passed','cases':10,'target_spellings':5,'selected_features':7,'target_token_gate_values':len(target_rows),
    'independent_cpu_vector_audits':len(audit),'source_weight_and_download_hashes_verified':True,
    'native_generation_matches_prior_baselines':True,'read_only_replays_token_identical':True,
    'layer3_target_gates_match_prior_measurements':True,'every_source_raw_and_gate_independently_checked':True,
    'cheef_subtokens_kept_separate':True,'vectors':audit}
(ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
summary={'status':'complete','features':stats,'columns':[{'word':w,'token':t} for w,t in columns],
    'matrices':{k:v.tolist() for k,v in matrices.items()},'feature_order':features,
    'answers':[{'case_id':r['id'],'word':r['target'],'answer':r['answer'],'expected':r['expected']} for r in D['rows']],
    'interpretation_limits':'Only five CH-starting spellings, four real words and nonce cheef; no non-CH control group. Gates are input- and position-dependent; absence/presence is not a causal conclusion. Raw gate scales are not calibrated across transforms.'}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def esc(value):return html.escape(str(value))
trs=[]
for name in features:
    s=stats[name]
    trs.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [name,s['label'],s['source_tokens_active'],s['source_tokens_total'],f"{s['max_source_gate']:.6f}",f"{s['child_gates']['uppercase']:.6f}",f"{s['child_gates']['lowercase']:.6f}"])+'</tr>')
answer_trs=['<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['id'],r['answer'],r['expected']])+'</tr>' for r in D['rows']]
details=[]
for row in D['rows']:
    entries=[t for t in row['prompt_tokens'] if t['target_token'] or t['position']==len(row['prompt_token_ids'])-1]+row['answer_tokens']
    body=[]
    for t in entries:
        body.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [t['segment'],t['position'],repr(t['token']),*[f"{t['features'][f]['gate']:.6f}" for f in features]])+'</tr>')
    details.append('<details><summary>'+esc(row['id'])+'</summary><pre>'+esc(row['user'])+'</pre><p>Native answer: <code>'+esc(row['answer'])+'</code></p><div class="wrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in ['Segment','Position','Token']+features)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div></details>')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CH word gates across selected MOLT layers</title><style>body{max-width:1250px;margin:32px auto;padding:0 20px;font:16px/1.55 system-ui;color:#172f3b;background:#fbfcfd}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d4dfe6;padding:8px;text-align:left;white-space:pre-wrap}th{background:#eaf1f5}.wrap{overflow-x:auto}pre{white-space:pre-wrap;background:#eef3f6;padding:12px}details{background:white;border:1px solid #d4dfe6;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:650}</style><h1>CH word gates across selected MOLT layers</h1><p>19 October 2026 · five target spellings · two unchanged few-shot templates · seven selected features</p><p>Read-only observations in the pinned native Qwen3-4B. Each selected MOLT gate is computed at its own layer from the residual after attention and before post-attention normalization, using the released standardization, encoder and threshold. No output contribution is edited.</p><p><strong>At the source child token, L3–T1372 is the only active selected gate: 3.212067 in the uppercase task and 3.243369 in the lowercase task. All five later-layer candidates are inactive at that input position in both tasks.</strong></p><img src="input-word-gates.png" alt="Natural MOLT gates at each input word subtoken for the uppercase and lowercase few-shot tasks"><p>Numbers are natural thresholded scalar gates. Zero means the released gate is inactive, not that its raw preactivation or all relevant model information is zero. Raw gate scales across transforms are not calibrated measures of causal importance. cheef splits into che and ef; no average, sum or maximum hides that distinction.</p><h2>Source-token measurements</h2><div class="wrap"><table><thead><tr><th>Feature</th><th>Gate label hypothesis</th><th>Active source subtokens</th><th>Source subtokens measured</th><th>Maximum source gate</th><th>child: uppercase task</th><th>child: lowercase task</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table></div><h2>Native answers</h2><p>The new unhooked generations reproduce every earlier baseline, including any earlier mistakes. Hooked observation replays reproduce all native output token IDs.</p><table><thead><tr><th>Case</th><th>Actual native answer</th><th>Expected letters</th></tr></thead><tbody>'''+''.join(answer_trs)+'''</tbody></table><h2>Scope and interpretation</h2><p>This compares child with four other CH-starting spellings, including the nonce cheef. It has no non-CH control group and does not establish broad CH selectivity. Gate presence does not identify what a transform writes. Gate absence does not show that the model lacks CH information or cannot compute it elsewhere.</p><p>The main figure measures the input word only. Additional traces below include the prompt-final prediction position and already generated answer tokens. A gate measured while processing an emitted CH/ch token cannot explain choosing that same token.</p><h2>Exact prompts and additional token traces</h2>'''+''.join(details)+'''<p><a href="input-word-gates.svg">Vector figure</a> · <a href="input-word-gates.csv">Input-token values CSV</a> · <a href="all-token-gates.csv">All observed token values CSV</a> · <a href="verification.json">Independent verification</a> · <a href="summary.json">Summary</a> · <a href="remote-export/artifacts/measurements.json">Full measurements</a></p></html>'''
(ROOT/'index.html').write_text(page)
print(json.dumps({'verification':'passed','target_token_gate_values':len(target_rows),'features':stats},indent=2))
