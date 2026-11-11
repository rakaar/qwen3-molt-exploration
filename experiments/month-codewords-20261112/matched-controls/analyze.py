"""Independent contribution, baseline-eligibility and probability audit."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE.parents[2]/'results/2026-11-12-month-codewords/matched-controls'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return s.strip().strip('`*_\"\'“”‘’').rstrip('.!?').strip().casefold()
def aggregate(v,target):
 q=[x for x in v['candidates'].values() if x['text'].strip().casefold()==target.casefold()]
 assert len({tuple(x['token_ids']) for x in q})==len(q)
 for i,x in enumerate(q):
  for y in q[i+1:]:assert x['token_ids'][:len(y['token_ids'])]!=y['token_ids'] and y['token_ids'][:len(x['token_ids'])]!=x['token_ids']
 return dict(probability=sum(x['word_probability'] for x in q),word_then_eos=sum(x['exact_word_then_eos_probability'] for x in q),variants=[x['text'] for x in q])
def main():
 p=json.loads((HERE/'protocol.json').read_text());r=json.loads((OUT/'results.json').read_text());baseline=json.loads((OUT/'baseline-results.json').read_text())
 assert r['status']=='complete' and r['hooks_remaining']==0 and r['protocol']==p and r['protocol_sha256']==sha(HERE/'protocol.json')
 assert r['source_sha256']==sha(HERE/'run.py')==sha(OUT/'run.py')
 assert r['baseline_lock']['sha256']==sha(OUT/'baseline-results.json') and baseline['phase']=='baseline' and all(not c['runs'] for c in baseline['cases'])
 assert len(r['cases'])==len(p['cases'])==6
 with np.load(HERE/'weights/L29-T2375.npz',allow_pickle=False) as f:w={k:f[k].astype(np.float32) for k in f.files}
 rows=[];audit_positions=0;edited_positions=0;zero_gate_runs=0;rounding_errors=[]
 for c,b in zip(r['cases'],baseline['cases']):
  ref=c['case'];assert ref==b['case'] and c['native']==b['native'] and c['zero']==b['zero'] and c['eligible']==b['eligible']
  assert c['native']['native']==c['zero']['native'] and c['native']['candidates']==c['zero']['candidates']
  correct=norm(c['native']['native']['response'])==ref['expected'].casefold() and c['native']['native']['stop_reason']=='eos' if not ref['calibration'] else c['native']['native']['response'].lstrip().casefold().startswith(ref['expected'].casefold())
  assert correct==c['native_correct']
  peers=[x for x in r['cases'] if x['case']['mapping_id']==ref['mapping_id'] and x['case']['prompt_format']==ref['prompt_format']];assert len(peers)==2 and c['eligible']==all(x['native_correct'] for x in peers)
  assert len(c['runs'])==(2 if ref['prompt_format']=='raw' else 3) if c['eligible'] else len(c['runs'])==0
  n=len(ref['input_token_ids']);native_input=[t for t in c['zero']['trace'] if t['context']=='generation' and t['phase']=='input'];assert len(native_input)==n and native_input[ref['is_position']]['token']==' is'
  for v in [c['native'],c['zero']]+c['runs']:
   f=OUT/v['arrays']['file'];assert sha(f)==v['arrays']['sha256']
   with np.load(f,allow_pickle=False) as z:
    probes=dict(zip(z['first_probe_ids'].tolist(),z['first_probe_logits'].tolist()));lse=float(z['first_logsumexp'][0])
    for score in v['candidates'].values():
     assert abs(score['conditional'][0]['log_probability']-(probes[score['token_ids'][0]]-lse))<3e-5
     for t in score['conditional']:assert abs(t['log_probability']-(t['logit']-t['logsumexp']))<3e-5
     assert sum(t['log_probability'] for t in score['conditional'])==score['word_log_probability']
     assert math.exp(score['word_log_probability'])==score['word_probability'] and 0<=score['eos_probability_given_word']<=1
    if v['alpha']==1:
     selected=set(range(n)) if v['scope']=='off-all' else {n-1} if v['scope']=='off-last' else {ref['is_position']}
     assert z['selected_positions'].tolist()==sorted(selected)
     before=np.array([t['natural_gate'] for t in native_input],dtype=np.float32);after=before.copy();after[list(selected)]=0
     np.testing.assert_array_equal(z['all_gate_before'],before);np.testing.assert_array_equal(z['all_gate_after'],after)
     positive={i for i in selected if before[i]>0};actual_positive=set(np.flatnonzero(z['all_actual_delta_norm']>0).tolist());assert actual_positive==positive
     np.testing.assert_array_equal(z['audit_positions'],sorted({ref['is_position'],n-1}|positive))
     standardized=(z['residual_input']-w['input_mean'])/w['input_std'];raw=standardized@w['e'].reshape(-1)+w['bias'][0];gate=np.where((raw>w['theta'][0])&(raw>0),raw,0);op=(standardized@w['V']@w['U'])*w['output_std']
     np.testing.assert_allclose(z['gate_before'],gate,rtol=1e-4,atol=1e-4);np.testing.assert_allclose(z['operator'],op,rtol=5e-4,atol=5e-5)
     np.testing.assert_allclose(z['intended_delta'],(z['gate_after']-z['gate_before'])[:,None]*z['operator'],rtol=1e-6,atol=1e-7)
     bits=(z['native_mlp']+z['intended_delta']).astype(np.float32).view(np.uint32);rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
     np.testing.assert_array_equal(z['edited_mlp'],rounded);np.testing.assert_array_equal(z['actual_delta'],z['edited_mlp']-z['native_mlp'])
     audit_positions+=len(z['audit_positions']);edited_positions+=len(positive);zero_gate_runs+=not positive
     for i in positive:rounding_errors.append(float(z['all_quantization_error_norm'][i]/z['all_intended_delta_norm'][i]))
   assert all(t['intended_delta_norm']==t['actual_delta_norm']==0 for t in v['trace'] if not t['edited'])
   for score in v['candidates'].values():
    k=len(score['token_ids'])
    if v['native']['output_token_ids'][:k]==score['token_ids']:assert math.isclose(math.prod(t['probability'] for t in v['native']['steps'][:k]),score['word_probability'],rel_tol=3e-5,abs_tol=1e-40)
   if v['alpha']==1:
    ed=[t for t in v['trace'] if t['edited']];assert {t['position'] for t in ed}==positive and all(t['phase']=='input' and t['context']=='generation' and t['effective_gate']==0 for t in ed)
    row=dict(id=c['id'],mapping_id=ref['mapping_id'],format=ref['prompt_format'],ordinal=ref['ordinal'],calibration=ref['calibration'],scope=v['scope'],expected=ref['expected'],code11=ref['code_11'],code12=ref['code_12'],native_response=c['native']['native']['response'],off_response=v['native']['response'],native_correct=c['native_correct'],pair_eligible=c['eligible'],native_stop=c['native']['native']['stop_reason'],off_stop=v['native']['stop_reason'],edited_positions=sorted(positive),edited_input_tokens=len(positive),generated_tokens_edited=0,query_is_natural_gate=before[ref['is_position']].item(),last_input_natural_gate=before[-1].item(),response_changed=v['native']['response']!=c['native']['native']['response'],off_code11=norm(v['native']['response'])==ref['code_11'].casefold() and v['native']['stop_reason']=='eos',off_code12=norm(v['native']['response'])==ref['code_12'].casefold() and v['native']['stop_reason']=='eos')
    for name,a in [('native',c['native']),('off',v)]:
     for target,value in [('code11',ref['code_11']),('code12',ref['code_12']),('month11','November'),('month12','December')]:
      score=aggregate(a,value);assert score['probability']<=1+1e-5;row[f'{name}_p_{target}']=score['probability'];row[f'{name}_p_{target}_eos']=score['word_then_eos']
    rows.append(row)
 pairs=[]
 for mapping in dict.fromkeys(c['case']['mapping_id'] for c in r['cases']):
  q=[c for c in r['cases'] if c['case']['mapping_id']==mapping];assert len(q)==2
  pairs.append(dict(mapping_id=mapping,format='chat',eligible=all(c['native_correct'] for c in q),native11=next(c['native']['native']['response'] for c in q if c['case']['ordinal']==11),native12=next(c['native']['native']['response'] for c in q if c['case']['ordinal']==12),code11='November',code12='December'))
 groups=[]
 for fmt in ['raw','chat']:
  for scope in ['off-is','off-last','off-all']:
   q=[x for x in rows if x['format']==fmt and x['scope']==scope and x['ordinal']==11]
   if q:groups.append(dict(format=fmt,scope=scope,eligible_pairs=len(q),code11_to_code12=sum(x['off_code12'] for x in q),retains_code11=sum(x['off_code11'] for x in q),changed_responses=sum(x['response_changed'] for x in q),actually_edited_cases=sum(x['edited_input_tokens']>0 for x in q)))
 summary=dict(control_cases=6,native_correct=sum(c['native_correct'] for c in r['cases']),eligible_pairs=sum(x['eligible'] for x in pairs),pairs=pairs,groups=groups,rows=rows,audit=dict(native_zero_exact=True,baseline_lock_verified=True,source_protocol_hashes_verified=True,independent_vector_positions=audit_positions,positive_input_positions_removed=edited_positions,generated_positions_edited=0,zero_gate_interventions=zero_gate_runs,bf16_rounding_exact=True,max_rounding_relative_l2=max(rounding_errors,default=0)),source_sha256=r['source_sha256'],results_sha256=sha(OUT/'results.json'))
 (OUT/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 with (OUT/'probabilities.csv').open('w',newline='') as f:
  wcsv=csv.DictWriter(f,fieldnames=list(rows[0]));wcsv.writeheader();wcsv.writerows(rows)
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2));print('CHANGED RESPONSES')
 for x in rows:
  if x['response_changed']:print(json.dumps(x,ensure_ascii=False))
if __name__=='__main__':main()
