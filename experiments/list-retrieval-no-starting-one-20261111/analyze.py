"""Verify complete passive traces and summarize every frozen pilot case."""
from pathlib import Path
import hashlib,json,re
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'experiments/list-retrieval-no-starting-one-20261111';OUT=ROOT/'results/2026-11-11-list-retrieval-pilot/without-starting-one'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def clean(s):return s.strip().strip('`*_\"\'“”‘’').rstrip('.!?').strip().strip('`*_\"\'“”‘’').casefold()
def stats(trace,region):
 rows=[t for t in trace if t['region']==region and not (region=='response' and t['special'])]
 return dict(tokens=len(rows),active_tokens=sum(t['gate']>0 for t in rows),max_gate=max((t['gate'] for t in rows),default=0),gate_sum=sum(t['gate'] for t in rows))
def main():
 r=json.loads((OUT/'results.json').read_text());p=json.loads((SRC/'protocol.json').read_text())
 assert r['status']=='complete' and r['hooks_remaining']==0 and r['interventions']==0 and len(r['cases'])==30
 assert r['protocol']==p and r['protocol_sha256']==sha(SRC/'protocol.json') and r['source_sha256']==sha(SRC/'run.py')
 assert sha(OUT/'run.py')==sha(SRC/'run.py') and sha(OUT/'protocol.json')==sha(SRC/'protocol.json')
 byid={c['id']:c for c in p['cases']};assert len(byid)==30
 for c in [r['calibration']]+r['cases']:
  assert c['native']==c['observed'] and c['observer_all_logits_and_outputs_exact']
  ids=c['input_token_ids']+c['native']['output_token_ids'];trace=c['trace'];assert len(trace)==len(ids)
  assert [t['token_id'] for t in trace]==ids and [t['position'] for t in trace]==list(range(len(ids)))
  assert sum(t['trace_only_terminal'] for t in trace)==1 and trace[-1]['trace_only_terminal']
  assert all(t['gate']>=0 and (t['gate']==0 or t['gate']==t['raw']) for t in trace)
  assert all((t['raw']>r['runtime']['threshold'] and t['raw']>0)==(t['gate']>0) for t in trace)
  if c['id']=='calibration-month':continue
  assert all(c[k]==v for k,v in byid[c['id']].items());assert c['expected']==c['words'][c['ordinal']-1]
  response=c['native']['response'];normalized=clean(response);exact_position=next((i for i,w in enumerate(c['words'],1) if clean(w)==normalized),None)
  found=sorted((m.start(),i) for i,w in enumerate(c['words'],1) for m in re.finditer(r'(?<![A-Za-z])'+re.escape(w)+r'(?![A-Za-z])',response,re.I))
  first=found[0][1] if found else None
  c['assessment']=dict(exact_answer_position=exact_position,strict_correct=exact_position==c['ordinal'] and c['native']['stop_reason']=='eos',first_recognized_position=first,first_word_correct=first==c['ordinal'],truncated=c['native']['stop_reason']=='token_limit',mentioned_positions=sorted({x[1] for x in found}))
  c['stats']={s:stats(trace,s) for s in ['list','question','answer_prefix','instruction','response']}
  c['stats']['final_input_gate']=trace[len(c['input_token_ids'])-1]['gate']
  c['item_word_boundaries']=[]
  for n in range(1,16):
   positions=[t['position'] for t in trace if t['phase']=='input' and t.get('word_item')==n];assert positions,n
   pos=min(positions);c['item_word_boundaries'].append(dict(item=n,word=c['words'][n-1],first_word_token_position=pos,first_word_token_gate=trace[pos]['gate'],preceding_token=trace[pos-1]['token'] if pos else None,preceding_token_gate=trace[pos-1]['gate'] if pos else None))
 groups=[]
 for fmt in ['numbered','bullets']:
  for n in [5,11,12]:
   rows=[c for c in r['cases'] if c['format']==fmt and c['ordinal']==n]
   groups.append(dict(format=fmt,ordinal=n,cases=len(rows),strict_correct=sum(c['assessment']['strict_correct'] for c in rows),first_word_correct=sum(c['assessment']['first_word_correct'] for c in rows),truncated=sum(c['assessment']['truncated'] for c in rows),active_cases={key:sum(c['stats'][key]['active_tokens']>0 for c in rows) for key in ['list','question','answer_prefix','response']},final_input_active_cases=sum(c['stats']['final_input_gate']>0 for c in rows)))
 shared=[]
 for listid in [x['id'] for x in p['lists']]:
  for fmt in ['numbered','bullets']:
   rows=[c for c in r['cases'] if c['list_id']==listid and c['format']==fmt];size=min(len(c['input_token_ids']) for c in rows);prefix=0
   while prefix<size and len({c['input_token_ids'][prefix] for c in rows})==1:prefix+=1
   differences=[abs(rows[0]['trace'][i]['raw']-c['trace'][i]['raw']) for c in rows[1:] for i in range(prefix)]
   shared.append(dict(list_id=listid,format=fmt,shared_input_prefix_tokens=prefix,max_raw_difference=max(differences,default=0),gates_exact=all(rows[0]['trace'][i]['gate']==c['trace'][i]['gate'] for c in rows[1:] for i in range(prefix)),active_locations_exact=all((rows[0]['trace'][i]['gate']>0)==(c['trace'][i]['gate']>0) for c in rows[1:] for i in range(prefix))))
 summary=dict(pilot_cases=30,lists=5,formats=2,requested_positions=[5,11,12],strict_correct=sum(c['assessment']['strict_correct'] for c in r['cases']),first_word_correct=sum(c['assessment']['first_word_correct'] for c in r['cases']),truncated=sum(c['assessment']['truncated'] for c in r['cases']),observer_exact_cases=31,interventions=0,groups=groups,shared_prefix_checks=shared,calibration=dict(prompt=r['calibration']['prompt'],response=r['calibration']['native']['response'],final_input_gate=r['calibration']['trace'][len(r['calibration']['input_token_ids'])-1]['gate'],observer_exact=True),max_pilot_gate=max(t['gate'] for c in r['cases'] for t in c['trace']),max_shared_prefix_raw_difference=max(x['max_raw_difference'] for x in shared),runtime=r['runtime'],results_sha256=sha(OUT/'results.json'),protocol_sha256=r['protocol_sha256'])
 question_events=[dict(case=c['id'],**t) for c in r['cases'] for t in c['trace'] if t['region']=='question' and t['gate']>0]
 list_events=[dict(case=c['id'],**t) for c in r['cases'] for t in c['trace'] if t['region']=='list' and t['gate']>0]
 summary['question_active_events']=question_events
 summary['list_active_events']=list_events
 summary['comparison']=p['comparison']
 assert all(' starting from 1' not in c['prompt'] for c in r['cases'])
 (OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'analyzed-cases.json').write_text(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 lines=['# Matched follow-up: delete starting from 1','',f"Native word-only correct: {summary['strict_correct']}/30. First recognized listed word correct: {summary['first_word_correct']}/30. Truncated: {summary['truncated']}/30.",'', 'All 30 frozen cases are included. Each native generation and its terminal read-only trace pass were exactly reproduced with the passive observer. Separate month calibration passed. No causal intervention was performed.','', '| Format | Requested item | Word-only correct | Query active | Answer-prefix active | Response text active |','|---|---:|---:|---:|---:|---:|']
 for g in groups:lines.append(f"| {g['format']} | {g['ordinal']} | {g['strict_correct']}/5 | {g['active_cases']['question']}/5 | {g['active_cases']['answer_prefix']}/5 | {g['active_cases']['response']}/5 |")
 lines+=['','## All native responses','','| Case | Expected | Response | Word-only correct |','|---|---|---|---|']
 for c in r['cases']:lines.append('| '+c['id']+' | '+c['expected']+' | '+c['native']['response'].replace('\n',' / ').replace('|','\\|')+' | '+str(c['assessment']['strict_correct'])+' |')
 lines+=['','## Interpretation limits','','Gate color represents the scalar JumpReLU gate, not proof of an indexing function. List activations precede the later question. Response-token gates are measured when the generated token is subsequently processed, not when that token is selected. The last generated token has a separately flagged trace-only extra forward, with no additional generated token. All input template tokens and EOS are retained in raw data.','']
 lines += ['All prompts match their original case IDs, word orders, formats, requested positions and settings. The only prompt change deletes the exact substring " starting from 1". The unchanged native runner checks every generated and terminal-unused logit against its passive-observer repeat.','']
 (OUT/'README.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
