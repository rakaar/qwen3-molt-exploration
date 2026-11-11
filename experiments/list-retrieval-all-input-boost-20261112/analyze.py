"""Independently audit all-input replacement and retain every response."""
from pathlib import Path
import argparse, csv, hashlib, json, math, re
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'results/2026-11-11-list-retrieval-pilot/all-input-boost'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(s):return s.strip().strip('`*_\"\'“”‘’').rstrip('.!?').strip().casefold()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot','followup']);a=ap.parse_args();out=ROOT/a.phase;src=HERE/a.phase
 p=json.loads((src/'protocol.json').read_text());r=json.loads((out/'results.json').read_text())
 assert r['status']=='complete' and r['hooks_remaining']==0 and r['protocol']==p
 assert r['protocol_sha256']==sha(src/'protocol.json') and r['source_sha256']==sha(src/'run.py')==sha(out/'run.py')
 assert [c['id'] for c in r['cases']]==[c['id'] for c in p['cases']]
 with np.load(src/'weights/L29-T2375.npz',allow_pickle=False) as f:w={k:f[k].astype(np.float32) for k in f.files}
 rows=[];errors=[];audit_count=0;total_positions=0
 for c,ref in zip(r['cases'],p['cases']):
  assert c['native']['native']==ref['native'] and c['zero']['native']==c['native']['native'] and c['zero']['candidates']==c['native']['candidates']
  assert [v['target_gate'] for v in c['runs']]==p['strengths']
  n=len(ref['input_token_ids']);both=c['native_requested_correct'] and c['native_eleventh_question_control']['strict_correct']
  for v in [c['native'],c['zero']]+c['runs']:
   assert v['scope']=='all_supplied_input_tokens_only';file=out/v['arrays']['file'];assert sha(file)==v['arrays']['sha256']
   with np.load(file,allow_pickle=False) as z:
    probes=dict(zip(z['first_probe_ids'].tolist(),z['first_probe_logits'].tolist()));lse=float(z['first_logsumexp'][0])
    for score in v['candidates'].values():
     assert abs(score['conditional'][0]['log_probability']-(probes[score['token_ids'][0]]-lse))<3e-5
     assert sum(t['log_probability'] for t in score['conditional'])==score['word_log_probability']
     assert math.exp(score['word_log_probability'])==score['word_probability']
     for t in score['conditional']:
      assert abs(t['log_probability']-(t['logit']-t['logsumexp']))<3e-5
     assert 0<=score['eos_probability_given_word']<=1
    if v['alpha']==1:
     np.testing.assert_array_equal(z['selected_positions'],np.arange(n));np.testing.assert_array_equal(z['all_gate_after'],np.full(n,v['target_gate'],dtype=np.float32))
     assert np.all(z['all_actual_delta_norm']>0)
     assert z['audit_positions'].tolist()==sorted({0,n//4,n//2,n*3//4,c['prefix_positions'][0]['position'],n-1})
     normalized=(z['residual_input']-w['input_mean'])/w['input_std'];raw=normalized@w['e'].reshape(-1)+w['bias'][0];gate=np.where((raw>w['theta'][0])&(raw>0),raw,0);op=(normalized@w['V']@w['U'])*w['output_std']
     np.testing.assert_allclose(z['gate_before'],gate,rtol=1e-4,atol=1e-4);np.testing.assert_allclose(z['operator'],op,rtol=5e-4,atol=5e-5)
     np.testing.assert_allclose(z['intended_delta'],(z['gate_after']-z['gate_before'])[:,None]*z['operator'],rtol=1e-6,atol=1e-7)
     desired=(z['native_mlp']+z['intended_delta']).astype(np.float32);bits=desired.view(np.uint32);rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
     np.testing.assert_array_equal(z['edited_mlp'],rounded);np.testing.assert_array_equal(z['actual_delta'],z['edited_mlp']-z['native_mlp'])
     errors.extend((z['all_quantization_error_norm']/z['all_intended_delta_norm']).tolist());audit_count+=len(z['audit_positions']);total_positions+=n
   assert all(t['actual_delta_norm']==t['intended_delta_norm']==0 for t in v['trace'] if not t['edited'])
   if v['alpha']==1:
    ed=[t for t in v['trace'] if t['edited']];assert [t['position'] for t in ed]==list(range(n))
    assert all(t['phase']=='input' and t['context']=='generation' and t['effective_gate']==v['target_gate'] and t['actual_delta_norm']>0 for t in ed)
   for score in v['candidates'].values():
    m=len(score['token_ids'])
    if v['native']['output_token_ids'][:m]==score['token_ids']:assert math.isclose(score['word_probability'],math.prod(t['probability'] for t in v['native']['steps'][:m]),rel_tol=3e-5,abs_tol=1e-40)
  for v in c['runs']:
   response=v['native']['response'];mentioned=sorted({i for i,word in enumerate(c['words'],1) if re.search(r'(?<![A-Za-z])'+re.escape(word)+r'(?![A-Za-z])',response,re.I)})
   complete=v['native']['stop_reason']=='eos';strict=clean(response)==c['eleventh_word'] and complete;unique=mentioned==[11] and complete
   row=dict(phase=a.phase,id=c['id'],list_id=c['list_id'],format=c['format'],ordinal=c['ordinal'],gate=v['target_gate'],expected=c['expected'],eleventh_word=c['eleventh_word'],native_response=c['native']['native']['response'],response=response,stop_reason=v['native']['stop_reason'],native_requested_correct=c['native_requested_correct'],native_eleventh_correct=c['native_eleventh_question_control']['strict_correct'],both_native_controls_correct=both,native_already_eleventh=c['native_already_eleventh'],strict_eleventh=strict,only_listed_word_eleventh=unique,new_strict_eleventh=strict and not c['native_already_eleventh'],new_unique_eleventh=unique and not c['native_already_eleventh'],mentioned_listed_positions=mentioned,literal_eleven=clean(response) in ['eleven','11','eleventh','11th'],response_changed=response!=c['native']['native']['response'],retains_requested=clean(response)==c['expected'] and complete,truncated=not complete,input_tokens_edited=n,generated_tokens_edited=0,generated_sequence_probability=math.prod(t['probability'] for t in v['native']['steps']))
   for name,val in [('native',c['native']),('steered',v)]:
    for k in [5,11,12]:
     score=val['candidates'][f'word_{k}'];row[f'{name}_p{k}']=score['word_probability'];row[f'{name}_p{k}_then_eos']=score['exact_word_then_eos_probability']
   rows.append(row)
 groups=[]
 for g in p['strengths']:
  q=[x for x in rows if x['gate']==g];groups.append(dict(gate=g,cases=len(q),new_unique_eleventh=sum(x['new_unique_eleventh'] for x in q),new_strict_eleventh=sum(x['new_strict_eleventh'] for x in q),new_unique_eleventh_native_correct=sum(x['new_unique_eleventh'] and x['native_requested_correct'] for x in q),new_unique_eleventh_both_controls=sum(x['new_unique_eleventh'] and x['both_native_controls_correct'] for x in q),retains_requested=sum(x['retains_requested'] for x in q),responses_changed=sum(x['response_changed'] for x in q),literal_eleven=sum(x['literal_eleven'] for x in q),truncated=sum(x['truncated'] for x in q),mean_p11_both_controls=float(np.mean([x['steered_p11'] for x in q if x['both_native_controls_correct']]))))
 summary=dict(phase=a.phase,cases=len(r['cases']),variants=len(rows),strengths=p['strengths'],native_correct=sum(c['native_requested_correct'] for c in r['cases']),both_controls_correct=sum(c['native_requested_correct'] and c['native_eleventh_question_control']['strict_correct'] for c in r['cases']),native_already_eleventh=sum(c['native_already_eleventh'] for c in r['cases']),groups=groups,rows=rows,audit=dict(native_and_zero_exact=True,all_input_positions_verified=total_positions,generated_positions_edited=0,independently_reconstructed_vector_samples=audit_count,exact_bf16_rounding_verified_on_samples=True,full_input_gate_and_delta_norms_verified=True,max_rounding_relative_l2_error=max(errors)),source_sha256=r['source_sha256'],protocol_sha256=r['protocol_sha256'],results_sha256=sha(out/'results.json'))
 (out/'analysis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 with (out/'probabilities.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
 print('CANDIDATE ITEM-11 ANSWERS:')
 for x in rows:
  if x['new_unique_eleventh']:print(json.dumps(x,ensure_ascii=False))
if __name__=='__main__':main()
