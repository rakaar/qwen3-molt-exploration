"""Verify combined narrative data against immutable saved measurements/generations."""
from pathlib import Path
import json,hashlib,subprocess,re
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'results/2026-10-15-abstention-overview';D=json.loads((R/'evidence.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p,v in D['source_sha256'].items():assert sha(ROOT/p)==v,p
old=ROOT/'results/2026-10-15-vampire-intermediate';new=ROOT/'results/2026-10-15-six-new-abstentions';obs=ROOT/'results/2026-10-14-passage-answer-gates'
oldrows={r['trial_id']:r for r in json.loads((old/'artifacts/all-results.json').read_text())['rows']};newrows={r['trial_id']:r for r in json.loads((new/'artifacts/all-steering.json').read_text())['rows']};oldg=json.loads((old/'reviewed-judgments.json').read_text());newg={r['trial_id']:r for r in json.loads((new/'reviewed-judgments.json').read_text())['rows']};natural={r['id']:r for r in json.loads((obs/'artifacts/answer-gates.json').read_text())['rows']}
assert len(D['cases'])==10 and len({r['id'] for r in D['cases']})==10 and sum(r['success'] for r in D['cases'])==7
for c in D['cases']:
 original=c['id'].startswith('P');r=(oldrows if original else newrows)[c['trial_id']];g=(oldg if original else newg)[c['trial_id']]
 assert r['question']==c['question'] and r['answer']==c['steered'] and r['added_gate']==c['strength']
 assert c['success']==g['clean_coherent_abstention' if original else 'success']
 if not c['success']:assert g['coherent'],'Failure illustration must actually be coherent'
for r in D['observational_examples']:
 original=natural[r['id']];tokens=[t for t in original['answer_tokens'] if not t['special']]
 assert original['saved_greedy_predictions_verified'] and not original['ancillary_explanation_error'] and r['tokens']==tokens and r['answer']==original['answer']
 assert r['mean_gate']==sum(t['gate'] for t in tokens)/len(tokens) and ''.join(t['token'] for t in tokens)==r['answer']
 if r['group']=='made_up':assert all(t['gate']==0 for t in tokens)
 else:assert any(t['gate']>0 for t in tokens)
info=subprocess.check_output(['pdfinfo',str(R/'poster-detailed.pdf')],text=True);assert re.search(r'Pages:\s+1\b',info)
text=subprocess.check_output(['pdftotext','-raw',str(R/'poster-detailed.pdf'),'-'],text=True);compact=' '.join(text.split())
for c in D['cases']:
 if c['success']:assert ' '.join(c['question'].split()) in compact,c['id']
assert '7/10' in compact and D['label'] in compact and 'not strict binary search' in compact
proof={'status':'passed','causal_cases':10,'clear_successes':7,'exact_steered_outputs_and_strengths_match_raw':True,'success_labels_match_saved_reviews':True,'failure_examples_coherent':True,'natural_examples':6,'natural_tokens':sum(r['visible_tokens'] for r in D['observational_examples']),'all_natural_predictions_verified':True,'natural_means_include_zeros_and_exclude_specials':True,'pdf_pages':1,'all_seven_success_queries_in_pdf':True,'source_hashes_verified':True,'evidence_sha256':sha(R/'evidence.json')}
(R/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
