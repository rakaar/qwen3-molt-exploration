"""Build a self-contained research page from verified saved measurements."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'results/2026-11-10-t2375-evaluation-cutoff'
OUT.mkdir(exist_ok=True)

def read(relative):
    return json.loads((ROOT/relative).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

coverage=read('results/2026-11-10-t2375-pre11-coverage/counts.json')
feature=read('comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json')
original=read('results/2026-11-09-t2375-explanation-refinement/heldout-plan.json')
extended=read('results/2026-11-10-t2375-pre11-lookahead/heldout-plan.json')
verified=read('results/2026-11-10-t2375-pre11-lookahead/verification.json')
dashboard=read('comparisons/2026-10-28-water-month/dashboard-examples.json')
assert verified['complete_requests']==96 and verified['unique_request_bodies']==72
assert verified['no_parse_failures'] and verified['runtime_matches_previous_local_judge']
prefix_index={(c['method'],c['row_index']):c for c in original['cases']}
after_index={(c['method'],c['row_index']):c for c in extended['cases']}
rows_by_fp={m:{c['fingerprint']:c for c in coverage['methods'][m]['cases']} for m in verified['methods']}
order=[49,55,57,68,69,86,90,47,39,98,13,45]
names={49:'Exam advice',55:'Black bean burrito',57:'Garage door instructions',68:'Contract clauses',
       69:'Psychoanalysts',86:'Vehicle survey',90:'WordPress tools',47:'NAB Show dates',
       39:'Diwali',98:'Tuscan festivals',13:'Poetry award',45:'Texas–Texas A&M rivalry'}
cases=[]
for n,i in enumerate(order,1):
    c=next(x for x in coverage['methods']['fuzz']['cases'] if x['row_index']==i)
    methods={}
    for method in verified['methods']:
        row_id=rows_by_fp[method][c['fingerprint']]['row_index']
        row=feature['methods'][method]['rows'][row_id]
        old=prefix_index[(method,row_id)]
        new=after_index[(method,row_id)]
        pair=next(p for p in verified['methods'][method]['pairs'] if p['row_index']==row_id)
        assert pair['activating'] is True
        methods[method]=dict(row=row_id,highlighted=row.get('highlighted_positions',[]),
            before={a:dict(messages=old[a+'_messages'],prediction=pair['prefix_only_predictions'][a]) for a in ('baseline','candidate')},
            after={a:dict(messages=new[a+'_messages'],prediction=pair[a]) for a in ('baseline','candidate')})
    row=feature['methods']['fuzz']['rows'][i]
    suffix=after_index[('fuzz',i)]['continuation_str_tokens']
    assert len(row['str_tokens'])==64 and len(suffix)==16
    cases.append(dict(id=f'{n:02}',name=names[i],family=c['families'][0],
                      tokens=row['str_tokens'],gates=row['activations'],added_tokens=suffix,methods=methods))
observations=[]
for index,title,caption in [
    (2,'A comma before November','The comma after October has gate 3.063; November follows.'),
    (14,'A boundary before item 11','The period and two newlines have gate 1.375; item 11 follows.'),
    (16,'A space before the number 11','The space has gate 1.367; the digits 11 follow.')]:
    example=dashboard['examples'][index]
    tokens=[]
    for t in example['tokens']:
        r=dict(t['rows'])
        tokens.append(dict(text=t['text'].replace('↵','\n'),gate=float(r['Feature activation']),
                           position=r['Dataset position'],token=r['Token']))
    observations.append(dict(index=index,title=title,caption=caption,tokens=tokens))
controls=[]
for pair in verified['methods']['active_fuzz']['pairs']:
    if not pair['activating']:
        c=after_index[('active_fuzz',pair['row_index'])]
        controls.append(dict(row=pair['row_index'],**{a:dict(prediction=pair[a],messages=c[a+'_messages']) for a in ('baseline','candidate')}))
assert len(controls)==12
sources=[
 'comparisons/2026-10-28-water-month/dashboard-examples.json',
 'comparisons/2026-11-09-t2375-explanation/original-campaign/features/2375.json',
 'results/2026-11-10-t2375-pre11-coverage/counts.json',
 'results/2026-11-10-t2375-pre11-lookahead/verification.json',
 'results/2026-11-10-t2375-pre11-lookahead/protocol.json',
 'results/2026-11-10-t2375-pre11-lookahead/heldout-plan.json',
 'results/2026-11-10-t2375-pre11-lookahead/audit.jsonl']
data=dict(feature='L29–T2375',built_at_utc=datetime.now(timezone.utc).isoformat(),
    descriptions={a:original[a+'_explanation'] for a in ('baseline','candidate')},
    observations=observations,dashboard_source=dashboard['source'],cases=cases,
    scores={m:v['arms'] for m,v in verified['methods'].items()},controls=controls,
    provenance=[dict(path=p,sha256=digest(ROOT/p)) for p in sources],
    runtime=read('results/2026-11-10-t2375-pre11-lookahead/runtime.json'))
(OUT/'study-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
html=(HERE/'template.html').read_text().replace('__STUDY_DATA__',encoded)
assert '__STUDY_DATA__' not in html
(OUT/'index.html').write_text(html)
shutil.copyfile(HERE/'build.py',OUT/'build.py')
shutil.copyfile(HERE/'template.html',OUT/'template.html')
manifest=dict(files=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='site-manifest.json'])
(OUT/'site-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(page=str(OUT/'index.html'),cases=len(cases),dashboard_examples=len(observations),bytes=(OUT/'index.html').stat().st_size)))
