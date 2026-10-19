"""Focused CH, Child and spelling-label survey with source ages retained."""
from pathlib import Path
import hashlib
import json
import re

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parent.parent
paths=[PROJECT/f'comparisons/2026-10-12-spider-ant/summaries/layer-{i}.json' for i in range(34)]
paths += [PROJECT/'comparisons/2026-10-13-safety-refusal/live-layer-34-summary.json',
          PROJECT/'comparisons/2026-10-19-confidence-candidates/live/layer-35-summary.json']
patterns={
    'explicit_ch':re.compile(r'(?<![a-z])ch(?![a-z])',re.I),
    'child_title':re.compile(r'\bchild(?:ren|hood|ish|like)?\b|\bchil\b',re.I),
    'orthography_title':re.compile(r'first.{0,15}(?:letter|character)|initial.{0,12}(?:letter|character)|(?:letter|character).{0,15}(?:identif|extract)|spell|alphabet|digraph|orthograph',re.I),
    'c_h_initial':re.compile(r'(?:["\x27]?[ch]["\x27]?[- ](?:initial|prefix))|(?:initial|letter|prefix).{0,15}["\x27][ch]["\x27]',re.I),
}
found=[];sources=[];count=0
for layer,path in enumerate(paths):
    d=json.loads(path.read_text());features=d['features']
    it=features.values() if isinstance(features,dict) else features
    sources.append({'layer':layer,'file':str(path.relative_to(PROJECT)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for f in it:
        count+=1;title=f.get('title') or f.get('label') or '';ex=f.get('explanation') or ''
        cats=[name for name,pat in patterns.items() if pat.search(title+' '+ex if name=='explicit_ch' else title)]
        if cats:
            found.append({'layer':layer,'transform':f['id'],'rank':f['rank'],'title':title,'explanation':ex,
                'categories':cats,'status':f['status'],'count':f.get('count'),'fingerprint':f.get('fingerprint'),
                'detection':f.get('methods',{}).get('detection',{}).get('metrics',{}).get('accuracy'),
                'fuzz':f.get('methods',{}).get('fuzz',{}).get('metrics',{}).get('accuracy'),
                'url':f'https://amsterdam.taild80f17.ts.net/layers/layer-{layer}/autointerp-layer/features/{f["id"]}.html',
                'source_file':str(path.relative_to(PROJECT))})
(OUT/'cached-label-matches.json').write_text(json.dumps(found,indent=2)+'\n')
(OUT/'scan-manifest.json').write_text(json.dumps({'layers':36,'transform_slots':count,'sources':sources,
    'source_age':'Layers 0-33 retrieved October 12; layer 34 October 13; layer 35 October 19. Shortlisted records require current-page verification.',
    'patterns':{k:v.pattern for k,v in patterns.items()}},indent=2)+'\n')
print('SCANNED',count,'MATCHES',len(found))
for cat in patterns:
    group=[f for f in found if cat in f['categories']]
    print('CATEGORY',cat,'COUNT',len(group))
    for f in group[:100]:print(f"L{f['layer']}-T{f['transform']} | {f['title']} | detection={f['detection']}")
