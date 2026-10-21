"""Combine verified source trials, sealed grades and adaptive search outcomes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SEARCH=ROOT/'stages/adaptive-l11-authority'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
trials=[]; sources=[]; checks=[]
baseline={}

def add(stage, resultpath, grades, allowed=None):
    data=read(resultpath)
    assert data['status']=='complete'
    srcsha=sha(resultpath)
    sources.append({'stage':stage,'path':str(resultpath.relative_to(ROOT)),'sha256':srcsha})
    for row in data['rows']:
        if row['kind']=='observe': continue
        if allowed and row['case_id'] not in allowed: continue
        if row['kind']=='additive' and row['group']!='malicious_intent': continue
        grade=grades.get(stage+'::'+row['id'],grades.get(row['id']))
        assert grade is not None, (stage,row['id'])
        assert sha_answer(row['answer'])==row['answer_sha256']
        if 'answer_sha256' in grade: assert grade['answer_sha256']==row['answer_sha256']
        if row['kind']=='native' and data['protocol']['decoding']['max_new_tokens']==512:
            if row['case_id'] in baseline:
                assert baseline[row['case_id']]['output_token_ids']==row['output_token_ids']
                continue
            baseline[row['case_id']]=row
        cap=data['protocol']['decoding']['max_new_tokens']
        old=next((r for r in trials if r['case_id']==row['case_id'] and r['strength']==row['strength'] and r['token_cap']==cap),None)
        if old:
            assert old['output_token_ids']==row['output_token_ids']
            continue
        trials.append({**row,'id':stage+'::'+row['id'],'source_stage':stage,
                       'source_result_sha256':srcsha,'token_cap':cap,'grade':grade})

def sha_answer(s): return hashlib.sha256(s.encode()).hexdigest()

main=ROOT/'stages/power-context-l11-8-to-16'
cases=read(main/'protocol.json')['cases']
cases+=[next(c for c in read(ROOT/'protocol.json')['cases'] if c['id']=='ruler_of_the_world')]
add('power-context-l11-8-to-16',main/'artifacts/results.json',read(main/'review/grades.json'))
add('initial',ROOT/'artifacts/remote-initial/artifacts/results.json',read(ROOT/'review/grades.json'),['ruler_of_the_world'])
exp=ROOT/'stages/expanded-l11-12-to-30'
add('expanded-l11-12-to-30',exp/'artifacts/results.json',read(exp/'review/grades.json'),['ruler_of_the_world'])
newgrades={}
for g in sorted((SEARCH/'reviews').glob('*/grades.json')):
    seal=read(g.parent/'seal.json')
    assert seal['blinded_grades_sha256']==sha(g.parent/'blinded-grades.json')
    assert seal['packet_sha256']==sha(g.parent/'blinded.jsonl')
    newgrades.update(read(g))
for d in sorted(SEARCH.glob('b*')):
    if not (d/'artifacts/results.json').exists(): continue
    if not (d/'backup-verification.json').exists(): continue
    assert read(d/'backup-verification.json')['all_remote_stage_files_hash_verified']
    data=read(d/'artifacts/results.json')
    for name,h in data['source_sha256'].items(): assert h==sha(d/name)
    assert (d/'run.py').read_bytes()==(ROOT/'run.py').read_bytes()
    g=dict(newgrades)
    for row in data['rows']:
        if row['kind']=='native' and data['protocol']['decoding']['max_new_tokens']==512:
            assert baseline[row['case_id']]['output_token_ids']==row['output_token_ids']
            original=next(t for t in trials if t['case_id']==row['case_id'] and t['strength']==0)
            g[d.name+'::'+row['id']]=original['grade']
    add(d.name,d/'artifacts/results.json',g)
    checks.append({'stage':d.name,'verification':read(d/'artifacts/verification.json'),
                   'backup':read(d/'backup-verification.json')})
selected={}
for c in cases:
    pool=[t for t in trials if t['case_id']==c['id'] and t['strength']>0 and
          t['grade']['coherence']=='coherent' and t['grade']['behavior']=='harmful']
    pool.sort(key=lambda t:(t['strength'], -t['token_cap']))
    selected[c['id']]=pool[0]['id'] if pool else None
decision=read(SEARCH/'decisions.json') if (SEARCH/'decisions.json').exists() else {}
protocol=read(main/'protocol.json')
method={k:protocol[k] for k in ['base_repo','base_revision','molt_repo','molt_revision','label_dataset','label_revision','patch','scope','attention']}
method['decoding']=protocol['decoding']
method['long_answer_cap']=1024
prefix_checks=[]
for trial in trials:
    if trial['token_cap']<=512: continue
    old=next(t for t in trials if t['token_cap']==512 and t['case_id']==trial['case_id'] and t['strength']==trial['strength'])
    prefix=old['output_token_ids']
    assert trial['output_token_ids'][:len(prefix)]==prefix
    assert trial['prompt_token_ids']==old['prompt_token_ids']
    if old['stop_reason']=='eos': assert trial['output_token_ids']==prefix
    prefix_checks.append({'trial':trial['id'],'reference':old['id'],'prefix_exact':True,
                          'matched_tokens':len(prefix),'new_tokens':len(trial['output_token_ids']),
                          'stop_reason':trial['stop_reason']})
out={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'plan':read(SEARCH/'search-plan.json'),
     'cases':cases,'trials':trials,'selected':selected,'findings':decision.get('findings',[]),
     'representative':decision.get('representative',{}),
     'stopping_notes':decision.get('stopping_notes',{}),'sources':sources,
     'verification':{'new_stage_checks':checks,'repeated_native_baselines_token_exact':True,
                     'source_runner_unchanged':True,'new_grades_sealed_before_mapping':True,
                     'method':method,
                     'long_prefix_checks':prefix_checks,
                     'adaptive_decisions':[read(f) for f in sorted(SEARCH.glob('decision-b*.json'))],
                     'new_generations':sum(x['verification']['generation_rows'] for x in checks),
                     'new_trace_records':sum(x['verification']['trace_records'] for x in checks),
                     'all_new_remote_stage_files_hash_verified':True}}
(SEARCH/'compiled.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'trials':len(trials),'selected':selected,'new_stages':len(checks)},ensure_ascii=False))
