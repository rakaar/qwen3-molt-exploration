"""Report active controls, repeated strengths, source gates, and actual write norms."""
import argparse,csv,html,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('result_dir',type=Path);root=ap.parse_args().result_dir
 R=json.loads((root/'artifacts/results.json').read_text());assert R['status']=='complete'
 verification=json.loads((root/'verification.json').read_text());assert verification['status']=='verified'
 for row in R['rows']:row.update(verification['probabilities'][row['id']])
 fixed=json.loads((root/'fixed-site/artifacts/results.json').read_text())
 fixed_verification=json.loads((root/'fixed-site/verification.json').read_text());assert fixed_verification['status']=='verified'
 for row in fixed['rows']:row.update(fixed_verification['probabilities'][row['id']])
 rows=R['rows']+fixed['rows'];features=R['selection']['features'];assert len(rows)==790
 for r in rows:r.setdefault('edit_layer',r['layer'])
 controls=[r for r in rows if r['stratum'] in ['same_layer','other_layers_fixed_site']]
 assert len(controls)==400 and all(r['nonzero_actual_edit'] for r in rows)
 for r in rows:
  assert all(g['gate']>0 for g in r['source_gates'])
 def describe(r):
  descriptions=[]
  for source,f,coefficient in r['terms']:
   v=features[f'{r["layer"]}:{f}'];gate=v[source+'_gate_gpu'];scale=r.get('norm_match_scale',1)
   descriptions.append(f'{coefficient*scale:+g} × {source} L{r["layer"]}–T{f} — {v["label"]} [source gate={gate:.6f}]')
  return ' ; '.join(descriptions)
 summary={'conditions':len(rows),'primary_active_control_conditions':len(controls),'primary_edit_layer':22,
  'primary_conditions_with_changed_final_logits':sum(r['logits_changed_from_clean'] for r in controls),
  'primary_cold_top_token_count':sum(r['top_token']=='cold' for r in controls),
  'unique_control_transforms':len({(r['layer'],f) for r in controls for src,f,c in r['terms']}),
  'all_source_gates_positive':True,'all_actual_edits_nonzero':True,'groups':[]}
 for r in rows:
  r['report_group']='norm_matched_at_layer22' if r['stratum'].startswith('norm_matched') and r['edit_layer']==22 else 'norm_matched_at_native_layer' if r['stratum']=='norm_matched' else r['stratum']
 for stratum in ['same_layer','other_layers_fixed_site','norm_matched_at_layer22','other_layers','norm_matched_at_native_layer']:
  for kind in ['positive','negative','pair','pair_reverse']:
   group=[r for r in rows if r['report_group']==stratum and r['kind']==kind]
   best=max(group,key=lambda r:r['P_cold'])
   summary['groups'].append({'stratum':stratum,'kind':kind,'conditions':len(group),
      'max_P_cold':best['P_cold'],'best_id':best['id'],'best_strength':best['strength'],
      'count_above_99_percent':sum(r['P_cold']>.99 for r in group),'cold_top_token_count':sum(r['top_token']=='cold' for r in group),
      'final_logits_changed_from_clean':sum(r['logits_changed_from_clean'] for r in group)})
 selected=next(r for r in rows if r['id']=='selected_pair_4.5');summary['selected_pair_4.5']=selected['P_cold']
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 cmap=plt.get_cmap('viridis');norm=LogNorm(vmin=.5,vmax=64)
 fig,axes=plt.subplots(1,2,figsize=(16,5.8),sharey=True)
 for ax,kinds in zip(axes,[['positive','negative'],['pair','pair_reverse']]):
  for stratum,offset in [('same_layer',0),('other_layers_fixed_site',50)]:
   ax.axvspan(offset-.5,offset+49.5,color='#dfeafa' if offset==0 else '#ffefd9',alpha=.45,zorder=0)
   for kind,marker in zip(kinds,['o','v']):
    group=sorted([r for r in controls if r['stratum']==stratum and r['kind']==kind],key=lambda r:r['sample_index'])
    assert len(group)==50
    ax.scatter([offset+r['sample_index'] for r in group],[r['P_cold']*100 for r in group],c=[r['strength'] for r in group],cmap=cmap,norm=norm,s=28,marker=marker,edgecolors='none',alpha=.85)
  references=[next(r for r in rows if r['id']==f'selected_{kind}_4.5') for kind in kinds]
  for r,x,color,marker in zip(references,[103,105],['#b05131','#7e338d'],['*','v']):
   ref_label={'positive':'Selected donor L22–T941, 4.5×','negative':'Selected receiver L22–T2339, −4.5×','pair':'Selected pair, 4.5×','pair_reverse':'Selected pair, reversed signs'}[r['kind']]
   ax.scatter(x,r['P_cold']*100,c=color,marker=marker,s=190,zorder=5,label=ref_label)
  ax.set_xlim(-3,108);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2)
  ax.set_xlabel('Trial index · 0–49: layer-22 sources; 50–99: other-layer sources')
  ax.set_title('Individual edits: + donor / − receiver' if kinds[0]=='positive' else 'Paired edits: − receiver + donor / reversed signs',fontsize=12)
  ax.scatter([],[],marker='o',c='#607487',label=kinds[0]);ax.scatter([],[],marker='v',c='#607487',label=kinds[1])
  ax.legend(loc='upper left',fontsize=9)
 axes[0].set_ylabel('P(cold), % of full vocabulary')
 fig.suptitle('Active MOLT controls · every primary edit at layer 22, token 9 · attention unfrozen\nReference pair: −4.5 receiver L22–T2339 +4.5 donor L22–T941\nOnly positive source gates; every actual control edit is nonzero',fontsize=13)
 fig.subplots_adjust(top=.80,bottom=.19,right=.89,wspace=.12)
 cax=fig.add_axes([.92,.24,.015,.48]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Contribution strength')
 fig.text(.07,.045,'Layer 22 reuses its 2 donor-active transforms, 2 receiver-active transforms, and 3 pairs across strengths.\nStars at right: original selected edits at 4.5×. Colour = strength; repeated trials are not independent feature samples.',fontsize=10)
 fig.savefig(root/'active-controls-scatter.png',dpi=190);fig.savefig(root/'active-controls-scatter.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(15,5.5))
 for ax,kinds in zip(axes,[['positive','negative'],['pair','pair_reverse']]):
  for stratum,offset in [('same_layer',0),('other_layers_fixed_site',50)]:
   for kind,marker in zip(kinds,['o','v']):
    group=[r for r in controls if r['stratum']==stratum and r['kind']==kind]
    ax.scatter([offset+r['sample_index'] for r in group],[r['P_cold']*100 for r in group],c=[r['strength'] for r in group],cmap=cmap,norm=norm,s=30,marker=marker,edgecolors='none',alpha=.85)
  ax.axvline(49.5,color='#8696a7',ls='--',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('0–49: layer 22; 50–99: other layers');ax.set_ylabel('P(cold), % of full vocabulary')
  ax.set_title('Individual active controls' if kinds[0]=='positive' else 'Paired active controls')
 fig.suptitle('Zoom on non-target controls · different y-axis scales · selected reference 99.8653% is off scale')
 fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(root/'active-controls-zoom.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(15,9),sharey=True)
 for y,stratum in enumerate(['same_layer','other_layers_fixed_site']):
  for x,kinds in enumerate([['positive','negative'],['pair','pair_reverse']]):
   ax=axes[y,x]
   for kind in kinds:
    group=[r for r in controls if r['stratum']==stratum and r['kind']==kind]
    recipes={tuple((src,f) for src,f,c in r['terms']) for r in group}
    for recipe in sorted(recipes):
     subset=sorted([r for r in group if tuple((src,f) for src,f,c in r['terms'])==recipe],key=lambda r:r['strength'])
     ax.plot([r['strength'] for r in subset],[r['P_cold']*100 for r in subset],marker='o' if kind==kinds[0] else 'v',ms=3,lw=.9,alpha=.6)
   for kind in kinds:
    subset=sorted([r for r in rows if r['stratum']=='selected' and r['kind']==kind],key=lambda r:r['strength'])
    ax.plot([r['strength'] for r in subset],[r['P_cold']*100 for r in subset],color='black' if kind==kinds[0] else '#9b2875',lw=2,ls='-' if kind==kinds[0] else '--',label='Selected '+kind)
   ax.set_xscale('log',base=2);ax.set_ylim(-3,103);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Contribution strength')
   ax.set_title(('Layer 22' if y==0 else 'Other sampled layers')+' · '+('individual edits' if x==0 else 'paired edits'))
   if x==0:ax.set_ylabel('P(cold), % of full vocabulary')
 fig.suptitle('Active controls across strengths · all edits at layer 22, token 9\nColoured curves: control recipes; bold curves: selected features',fontsize=13)
 fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(root/'active-controls-strengths.png',dpi=180);plt.close(fig)
 fields=['id','stratum','kind','layer','edit_layer','strength','recipe_with_source_gates','mean_source_gate','intended_delta_norm','actual_delta_norm','native_mlp_norm','P_cold','P_large','top_token','norm_match_scale','logits_changed_from_clean']
 with (root/'conditions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for r in rows:w.writerow({**r,'recipe_with_source_gates':describe(r)})
 esc=html.escape
 group_rows=''.join(f'<tr><td>{g["stratum"]}</td><td>{g["kind"]}</td><td>{g["conditions"]}</td><td>{100*g["max_P_cold"]:.6g}%</td><td>{g["count_above_99_percent"]}</td></tr>' for g in summary['groups'])
 feature_rows=[]
 for v in sorted(features.values(),key=lambda v:(v['layer'],v['id'])):
  rg=v['receiver_gate_gpu'];dg=v['donor_gate_gpu'];positive=[g for g in [rg,dg] if g>0];mean=sum(positive)/len(positive)
  role='Reference' if v['layer']==22 and v['id'] in [2339,941] else 'Active control'
  feature_rows.append(f'<tr><td>L{v["layer"]}–T{v["id"]}<small>{role}</small></td><td>{esc(v["label"])} <b>[mean active gate={mean:.6f}]</b></td><td>{rg:.6f}<small>C norm {v["receiver_contribution_norm"]:.5g}</small></td><td>{dg:.6f}<small>C norm {v["donor_contribution_norm"]:.5g}</small></td></tr>')
 tr=[]
 for r in rows:
  tr.append(f'<tr data-search="{esc((r["stratum"]+" "+r["kind"]+" "+describe(r)).lower(),quote=True)}"><td>{esc(r["id"])}<small>{r["stratum"]} · {r["kind"]}</small><small>source L{r["layer"]} → edit L{r["edit_layer"]}, token 9</small></td><td>{r["strength"]:g}'+(f'<small>norm-match multiplier {r["norm_match_scale"]:.5g}</small>' if 'norm_match_scale' in r else '')+f'</td><td class="recipe">{esc(describe(r))}</td><td>{r["mean_source_gate"]:.6f}</td><td>{r["intended_delta_norm"]:.5g}<small>actual {r["actual_delta_norm"]:.5g}; native MLP {r["native_mlp_norm"]:.5g}</small></td><td>{100*r["P_cold"]:.7g}%</td><td>{100*r["P_large"]:.6g}%<small>top: {esc(r["top_token"])}</small></td></tr>')
 page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Active MOLT controls</title><style>
 *{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#17283b;font:16px/1.6 system-ui}}main{{max-width:1450px;margin:auto;padding:30px 22px}}h1{{line-height:1.2;font-size:34px}}section{{background:white;border:1px solid #d1ddea;border-radius:12px;padding:22px;margin:22px 0}}img{{max-width:100%;height:auto}}a{{color:#155a9e}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ed}}th{{background:#edf3fa}}small{{display:block;color:#53677d;font-size:12px}}.scroll{{overflow:auto}}.conditions{{min-width:1250px}}.recipe{{min-width:390px;max-width:480px;white-space:pre-wrap}}input{{width:100%;padding:12px;border:1px solid #b8cada;border-radius:7px;font:inherit}}code{{font-size:13px;overflow-wrap:anywhere}}.note{{background:#fff4df;padding:16px;border-radius:8px}}
 </style></head><body><main><h1>Active controls for the small → hot MOLT edit</h1>
 <p>The original uniform random controls were inactive at their source tokens. This follow-up tests only gate-active contributions. Native Qwen3-4B, unfrozen attention; all primary controls edit layer 22, token 9, matching the selected intervention site.</p>
 <p><a href="#features">Jump to the 35 picked controls and activation values</a> · <a href="#trials-section">Search individual trials</a></p>
 <section><h2>What was tested</h2><p><b>400 primary active-control conditions:</b> 100 donor additions, 100 receiver subtractions, and 100 pairs tested in both sign directions. For each recipe, 50 trials use layer 22 and 50 use the other-layer pool. Layer 22 has only two non-target active transforms per source and three distinct pairs; strength repetitions are counted as conditions, not unique feature draws.</p>
 <p>There are <b>{summary['unique_control_transforms']} unique control transforms</b>. Another 100 conditions sweep the selected features, and 50 conditions match the selected edit's FP32 vector norm at layer 22. A supplementary native-layer arm tests 240 other-layer conditions at their original layers. There are 790 conditions in total. Every source gate is positive and every actual BF16 MLP edit is nonzero.</p>
 <p><b>Receiver:</b> {esc(R['protocol']['receiver'])}<br><b>Donor:</b> {esc(R['protocol']['donor'])}<br><b>Source and target token:</b> position 9, respectively “hot” and “small,” including the chat template.</p>
 <p><code>MLP_receiver_output[22,9] += −strength × C_receiver,A + strength × C_donor,B</code><br>C is the full gate × low-rank output × released output scaling, computed from the clean source input at its own native layer. Individual controls use one term. Reverse controls flip both signs. Pair members always come from the same source layer. Other-layer controls transfer their completed source contribution vectors to layer 22; they do not pass a layer-22 input into a differently trained gate.</p>
 <p>Other layers screened: {R['protocol']['other_layers']}. Selection uses clean gates and a fixed seed before intervention outcomes. The complete release is pinned; layer 22's whole-checkpoint SHA256 is identical to the original release.</p>
 <p class="note">The strength grids differ between strata and include gains larger than 4.5. Source gates and actual write magnitudes are shown below; equal gain does not mean equal edit norm. The separate norm-matched controls address that difference. These are next-token probability tests, not a measure of semantic specificity across tasks.</p><p>The native-layer arm is retained separately. At the final layer 35, an MLP-output edit at token 9 cannot reach the readout at token 28 because no attention layer follows it. Those native-layer entries are structurally unable to change this readout; they are not included in the primary control comparison. Transferring the same source vectors to layer 22 avoids that missing-path problem.</p></section>
 <section><h2>Corrected scatterplot</h2><a href="active-controls-scatter.png"><img src="active-controls-scatter.png" alt="Active individual and paired control probabilities"></a><p><a href="active-controls-scatter.pdf">PDF</a> · <a href="active-controls-strengths.png">Strength curves</a></p><h3>Zoom on the active controls</h3><a href="active-controls-zoom.png"><img src="active-controls-zoom.png" alt="Zoomed probabilities of non-target controls"></a><p>These probabilities are small, but the edits are nonzero. The two zoom panels use different y-axis scales.</p></section>
 <section><h2>Maximum P(cold) by control group</h2><p>Selected L22–T2339 / T941 pair at 4.5×: <b>{selected['P_cold']*100:.6f}%</b>. Maxima below are exploratory maxima across each stated grid; counts include repeated strengths.</p><div class="scroll"><table><thead><tr><th>Source pool</th><th>Recipe</th><th>Conditions</th><th>Maximum P(cold)</th><th>Conditions above 99%</th></tr></thead><tbody>{group_rows}</tbody></table></div></section>
 <section id="features"><h2>Picked transforms and average active gates</h2><p>35 control transforms and two reference transforms. The bracketed mean averages a transform's <b>nonzero</b> source gates at token 9 across the clean receiver and donor. Zeros are excluded from this active-only average; both raw source values are shown separately. This is not an average over other token positions, steering strengths, or the training corpus.</p><div class="scroll"><table><thead><tr><th>Transform ID</th><th>Dashboard label [average active gate]</th><th>Receiver: small</th><th>Donor: hot</th></tr></thead><tbody>{''.join(feature_rows)}</tbody></table></div></section>
 <section id="trials-section"><h2>Every feature, activation, strength, and result</h2><p>Brackets show the source-specific gate at token 9. The mean column averages gates of the contributing source terms; for a single term it equals that source gate. The eight identical clean source rows are verified identical. Gates are not averaged over unrelated prompt tokens or scaled by the steering strength.</p><input id="filter" placeholder="Filter by layer, transform ID, feature label, or recipe" aria-label="Filter control trials"><p id="count">790 conditions</p><div class="scroll"><table class="conditions"><thead><tr><th>Trial / source and edit layers</th><th>Strength</th><th>Features and applied coefficients [source gate]</th><th>Mean source gate</th><th>Edit L2 norm</th><th>P(cold)</th><th>P(large) / top token</th></tr></thead><tbody id="trials">{''.join(tr)}</tbody></table></div></section>
 <section><h2>Saved evidence</h2><p><a href="conditions.csv">Full combined CSV</a> · <a href="summary.json">Summary counts</a> · <a href="artifacts/results.json">Native-site raw results</a> · <a href="fixed-site/artifacts/results.json">Fixed-site raw results</a> · <a href="verification.json">Native-site verification</a> · <a href="fixed-site/verification.json">Fixed-site verification</a> · <a href="protocol.json">Native default protocol</a> · <a href="fixed-site-declaration.json">Fixed-site protocol override</a> · <a href="selection.json">Fixed feature selection and weight-slice provenance</a> · <a href="conditions.json">Original conditions</a> · <a href="fixed-site/conditions-fixed-site.json">Fixed-site conditions</a> · <a href="worker.py">Native-site model code</a> · <a href="fixed-site/worker.py">Fixed-site model code</a></p><p>Probabilities shown here use CPU float64 softmax of the saved full-vocabulary BF16 logits. The selected reference's logits match the original experiment bit-for-bit. The original BF16 write order is preserved: round the FP32 delta to BF16, then add it to the native BF16 MLP output. A first attempt with FP32 addition failed the reference check; the same conditions were rerun with the original order. The fixed-site arm was subsequently added to hold intervention location constant, reusing exactly the same other-layer source features and strengths.</p><p>Source gate and transformation matrices are taken from pinned public checkpoint slices. Tensor byte ranges and hashes are retained. Dashboard titles are cached step-100000 descriptions; missing labels are explicitly marked. Labels are descriptions of activation contexts, not causal validation.</p></section>
 </main><script>const input=document.getElementById('filter');input.addEventListener('input',()=>{{const q=input.value.toLowerCase();let n=0;for(const r of document.querySelectorAll('#trials tr')){{const show=r.dataset.search.includes(q);r.hidden=!show;if(show)n++;}}document.getElementById('count').textContent=n+' conditions';}});</script></body></html>'''
 (root/'index.html').write_text(page);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
