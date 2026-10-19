"""Audit native identity, token indexing, released gate arithmetic and summaries."""
import hashlib,json,math
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;W=S.parents[1];R=W/'results/2026-10-19-uncertainty-gate-activation'
read=lambda p:json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
d=read(R/'artifacts/measurements.json');native=read(R/'artifacts/native.json');p=read(S/'protocol.json');cases=read(S/'cases.json');assets=read(S/'weights/manifest.json')
assert d['status']==native['status']=='complete' and len(d['rows'])==len(native['rows'])==len(cases)==14
assert d['protocol']==native['protocol']==p
for n,h in d['source_sha256'].items():assert sha(S/n)==h,n
for n,h in p['source_sha256'].items():assert sha(W/n)==h,n
for k,v in p['runtime'].items():assert d['runtime'][k]==v
assert assets==d['assets'] and assets['revision']==p['molt_revision'] and assets['repo']==p['molt_repo']
thresholds={}
for a in assets['features']:
 path=S/'weights'/a['file'];assert sha(path)==a['sha256']
 with np.load(path) as x:
  thresholds[f"L{a['layer']}-T{a['transform']}"]=float(x['theta'][0]);assert a['theta']==float(x['theta'][0])
  assert len(x['e'])==len(x['input_mean'])==len(x['input_std'])==2560
  assert np.isfinite(x['e']).all() and (x['input_std']>0).all()
 assert sha(S/'weights'/f"layer-{a['layer']}-config.json")==a['source_config_sha256']
 assert sha(S/'weights'/f"layer-{a['layer']}-header.json")==a['source_header_sha256']
keys=list(thresholds);predictions=prompt_tokens=answer_tokens=0
for c,n,r in zip(cases,native['rows'],d['rows']):
 assert c['id']==n['id']==r['id'] and c['group']==n['group']==r['group']
 assert n['baseline_generation_no_hooks'] and r['baseline_generation_no_hooks']
 assert n['output_token_ids']==r['output_token_ids'] and n['answer']==r['answer']
 if c['group']!='positive_control':assert r['saved_baseline_tokens_match'] and r['output_token_ids']==c['saved_output_token_ids'] and r['answer']==c['saved_answer']
 assert r['observation_only_predictions_match'] and r['predictions_verified']==len(r['output_token_ids']);predictions+=r['predictions_verified']
 assert [t['token_id'] for t in r['prompt_tokens']]==r['prompt_token_ids']
 assert [t['token_id'] for t in r['answer_tokens']]==r['output_token_ids'][:-1] and r['stop_reason']=='eos'
 assert ''.join(t['token'] for t in r['answer_tokens'] if not t['special'])==r['answer']
 assert len(r['prompt_tokens'])==len(r['prompt_token_ids'])
 assert r['chat'][r['query_character_span'][0]:r['query_character_span'][1]]==r['question']
 assert r['query_tokens']==[r['prompt_tokens'][j] for j in r['query_token_positions']]
 assert r['query_token_positions']==list(range(min(r['query_token_positions']),max(r['query_token_positions'])+1))
 for j,t in enumerate(r['prompt_tokens']):assert t['position']==j and t['full_chat_position']==j and t['segment']=='prompt'
 for j,t in enumerate(r['answer_tokens']):assert t['position']==j and t['full_chat_position']==len(r['prompt_tokens'])+j and t['segment']=='response'
 for t in r['prompt_tokens']+r['answer_tokens']:
  assert set(t['features'])==set(keys)
  for key,x in t['features'].items():
   g=x['raw'] if x['raw']>max(0,thresholds[key]) else 0.;assert math.isclose(g,x['gate'],rel_tol=1e-7,abs_tol=1e-7)
 for key in keys:
  ts=[t for t in r['answer_tokens'] if not t['special']];gs=[t['features'][key]['gate'] for t in ts];sm=r['summary'][key]
  assert sm['visible_tokens']==len(gs) and sm['active_tokens']==sum(g>0 for g in gs) and sm['max_gate']==max(gs)
  assert math.isclose(sm['mean_gate'],sum(gs)/len(gs),rel_tol=1e-7,abs_tol=1e-7)
  for x,y in [('final_prompt_gate','gate'),('final_prompt_raw','raw')]:assert sm[x]==r['prompt_tokens'][-1]['features'][key][y]
  qs=[t['features'][key]['gate'] for t in r['query_tokens']];ps=[t['features'][key]['gate'] for t in r['prompt_tokens'] if not t['special']]
  assert sm['query_tokens']==len(qs) and sm['query_active_tokens']==sum(g>0 for g in qs) and sm['query_max_gate']==max(qs)
  assert math.isclose(sm['query_mean_gate'],sum(qs)/len(qs),abs_tol=1e-7)
  assert sm['prompt_visible_tokens']==len(ps) and sm['prompt_active_tokens']==sum(g>0 for g in ps)
  assert math.isclose(sm['prompt_mean_gate'],sum(ps)/len(ps),abs_tol=1e-7)
 prompt_tokens+=len(r['prompt_tokens']);answer_tokens+=len(r['answer_tokens'])
for group in ['correct','made_up','abstention']:assert sum(r['group']==group for r in d['rows'])==4
pcs=[r for r in d['rows'] if r['group']=='positive_control'];assert len(pcs)==2
positive={key:any(r['summary'][key]['active_tokens']>0 for r in pcs) for key in keys}
nonzero_observed={key:any(t['features'][key]['gate']>0 for r in d['rows'] for t in r['prompt_tokens']+r['answer_tokens']) for key in keys}
result={'passed':True,'qa_cases':12,'separate_wording_controls':2,'gates':keys,'native_saved_baseline_matches':12,'observation_only_replay_matches':14,'verified_next_token_predictions':predictions,'prompt_tokens':prompt_tokens,'answer_tokens':answer_tokens,'positive_response_control_per_gate':positive,'nonzero_observed_anywhere':nonzero_observed,'source_asset_gate_arithmetic_and_summary_checks':True,'attention_and_native_mlp':'No replacement or modification; observation-only hooks return None.'}
(R/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
