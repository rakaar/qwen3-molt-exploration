"""Audit the ice/water probe and distinguish input from answer-processing activity."""
from pathlib import Path
import hashlib,json,math,re,csv
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE.parents[1]/'results/2026-11-12-water-ice-probe'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(s):return s.strip().strip('`*_\"\'“”‘’').rstrip('.!?').strip().casefold()
def aggregate(v,word):
 a=[x for x in v['candidates'].values() if x['text'].strip().casefold()==word.casefold()]
 if not a:return None
 assert len({tuple(x['token_ids']) for x in a})==len(a)
 for i,x in enumerate(a):
  for y in a[i+1:]:assert x['token_ids'][:len(y['token_ids'])]!=y['token_ids'] and y['token_ids'][:len(x['token_ids'])]!=x['token_ids']
 return dict(probability=sum(x['word_probability'] for x in a),word_then_eos=sum(x['exact_word_then_eos_probability'] for x in a),variants=[x['text'] for x in a])
def main():
 r=json.loads((OUT/'results.json').read_text());p=json.loads((HERE/'protocol.json').read_text());b=json.loads((OUT/'baseline-results.json').read_text())
 assert r['status']=='complete' and r['hooks_remaining']==0 and r['protocol']==p and r['protocol_sha256']==sha(HERE/'protocol.json')
 assert r['source_sha256']==sha(HERE/'run.py')==sha(OUT/'run.py') and r['baseline_lock']['sha256']==sha(OUT/'baseline-results.json')
 assert b['phase']=='baseline' and all(not x['runs'] for x in b['cases']) and len(r['cases'])==8
 with np.load(HERE/'weights/L29-T2375.npz',allow_pickle=False) as z:w={k:z[k].astype(np.float32) for k in z.files}
 rows=[];cases=[];samples=0;removed=0
 for c,bc in zip(r['cases'],b['cases']):
  ref=c['case'];assert ref==bc['case'] and c['native']==bc['native'] and c['zero']==bc['zero'] and c['native_correct']==bc['native_correct']
  assert c['native']['native']==c['zero']['native'] and c['native']['candidates']==c['zero']['candidates']
  n=len(ref['input_token_ids']);tr=[t for t in c['zero']['trace'] if t['context']=='generation'];inp=[t for t in tr if t['phase']=='input'];out=[t for t in tr if t['phase']=='response'];assert len(inp)==n
  complete=None;prefix=''
  for i,t in enumerate(c['native']['native']['steps']):
   prefix+=t['token']
   if prefix.lstrip().casefold().startswith(ref['expected'].casefold()):complete=i;break
  case=dict(id=c['id'],prompt=ref['prompt'],format=ref['prompt_format'],expected=ref['expected'],calibration=ref['calibration'],native_correct=c['native_correct'],native_response=c['native']['native']['response'],native_stop=c['native']['native']['stop_reason'],query_final_position=ref['query_final_position'],input_positive=[dict(position=t['position'],token=t['token'],gate=t['natural_gate'],region=t['region']) for t in inp if t['natural_gate']>0],response_positive=[dict(position=t['position'],response_index=t['position']-n,token=t['token'],gate=t['natural_gate']) for t in out if t['natural_gate']>0],expected_word_complete_at_response_index=complete,native_expected_score=aggregate(c['native'],ref['expected']),native_water_score=aggregate(c['native'],'water'),native_ice_score=aggregate(c['native'],'ice'))
  case['response_activity_before_expected_word_complete']=[x for x in case['response_positive'] if complete is not None and x['response_index']<complete]
  case['response_activity_on_or_after_completed_expected_word']=[x for x in case['response_positive'] if complete is not None and x['response_index']>=complete]
  cases.append(case)
  for v in [c['native'],c['zero']]+c['runs']:
   f=OUT/v['arrays']['file'];assert sha(f)==v['arrays']['sha256']
   with np.load(f,allow_pickle=False) as a:
    probes=dict(zip(a['first_probe_ids'].tolist(),a['first_probe_logits'].tolist()));lse=float(a['first_logsumexp'][0])
    for score in v['candidates'].values():
     assert abs(score['conditional'][0]['log_probability']-(probes[score['token_ids'][0]]-lse))<3e-5
     for t in score['conditional']:assert abs(t['log_probability']-(t['logit']-t['logsumexp']))<3e-5
     assert sum(t['log_probability'] for t in score['conditional'])==score['word_log_probability'] and math.exp(score['word_log_probability'])==score['word_probability']
     k=len(score['token_ids'])
     if v['native']['output_token_ids'][:k]==score['token_ids']:assert math.isclose(math.prod(t['probability'] for t in v['native']['steps'][:k]),score['word_probability'],rel_tol=3e-5,abs_tol=1e-40)
    if v['alpha']==1:
     selected=set(range(n)) if v['scope']=='off-all' else {n-1};before=np.array([t['natural_gate'] for t in inp],dtype=np.float32);after=before.copy();after[list(selected)]=0;positive={i for i in selected if before[i]>0}
     assert a['selected_positions'].tolist()==sorted(selected);np.testing.assert_array_equal(a['all_gate_before'],before);np.testing.assert_array_equal(a['all_gate_after'],after)
     assert set(np.flatnonzero(a['all_actual_delta_norm']>0).tolist())==positive
     z=(a['residual_input']-w['input_mean'])/w['input_std'];raw=z@w['e'].reshape(-1)+w['bias'][0];gate=np.where((raw>w['theta'][0])&(raw>0),raw,0);op=(z@w['V']@w['U'])*w['output_std']
     np.testing.assert_allclose(a['gate_before'],gate,rtol=1e-4,atol=1e-4);np.testing.assert_allclose(a['operator'],op,rtol=5e-4,atol=5e-5);np.testing.assert_allclose(a['intended_delta'],(a['gate_after']-a['gate_before'])[:,None]*a['operator'],rtol=1e-6,atol=1e-7)
     bits=(a['native_mlp']+a['intended_delta']).astype(np.float32).view(np.uint32);rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32);np.testing.assert_array_equal(a['edited_mlp'],rounded);np.testing.assert_array_equal(a['actual_delta'],a['edited_mlp']-a['native_mlp'])
     samples+=len(a['audit_positions']);removed+=len(positive)
   assert all(t['actual_delta_norm']==t['intended_delta_norm']==0 for t in v['trace'] if not t['edited'])
   if v['alpha']==1:
    edits=[t for t in v['trace'] if t['edited']];assert {t['position'] for t in edits}==positive and all(t['phase']=='input' and t['effective_gate']==0 and t['context']=='generation' for t in edits)
    row=dict(id=c['id'],format=ref['prompt_format'],expected=ref['expected'],calibration=ref['calibration'],scope=v['scope'],native_correct=c['native_correct'],native_response=c['native']['native']['response'],off_response=v['native']['response'],off_stop=v['native']['stop_reason'],edited_input_tokens=len(positive),edited_positions=sorted(positive),generated_tokens_edited=0,response_changed=v['native']['response']!=c['native']['native']['response'])
    for label,value in [('native',c['native']),('off',v)]:
     for key,word in [('expected',ref['expected']),('water','water'),('ice','ice')]:
      score=aggregate(value,word);row[label+'_p_'+key]=score['probability'] if score else None;row[label+'_p_'+key+'_eos']=score['word_then_eos'] if score else None
    rows.append(row)
 s=dict(cases=cases,rows=rows,audit=dict(source_hash_verified=True,baseline_lock_verified=True,native_zero_exact=True,independent_vector_samples=samples,positive_input_contributions_removed=removed,generated_positions_edited=0,bf16_rounding_exact=True),source_sha256=r['source_sha256'],results_sha256=sha(OUT/'results.json'))
 (OUT/'analysis.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
 with (OUT/'probabilities.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(s,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
