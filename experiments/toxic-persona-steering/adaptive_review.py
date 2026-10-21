"""Response-only review packets and sealed joins for adaptive stages."""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent/'stages/adaptive-l11-authority'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['make','seal'])
parser.add_argument('name')
parser.add_argument('stages',nargs='*')
parser.add_argument('--include-native',action='store_true')
args=parser.parse_args()
dest=ROOT/'reviews'/args.name
if args.mode=='make':
    assert not dest.exists()
    rows=[]
    for stage in args.stages:
        path=ROOT/stage/'artifacts/results.json'
        r=json.loads(path.read_text())
        assert r['status']=='complete'
        rows.extend((stage,x,sha(path)) for x in r['rows'] if x['kind']=='additive' or (args.include_native and x['kind']=='native'))
    dest.mkdir(parents=True)
    random.SystemRandom().shuffle(rows)
    packet=[]; mapping={}
    for i,(stage,row,result_sha) in enumerate(rows,1):
        key=f'R{i:03d}'
        packet.append({'review_id':key,'question':row['question'],'answer':row['answer'],'stop_reason':row['stop_reason']})
        mapping[key]={'id':stage+'::'+row['id'],'answer_sha256':row['answer_sha256'],'source_result_sha256':result_sha}
    (dest/'blinded.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in packet))
    (dest/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print(json.dumps({'packet':str(dest/'blinded.jsonl'),'responses':len(packet)}))
else:
    packet=[json.loads(line) for line in (dest/'blinded.jsonl').read_text().splitlines()]
    grades=json.loads((dest/'blinded-grades.json').read_text())
    assert set(grades)=={r['review_id'] for r in packet}
    for r in grades.values():
        assert r['coherence'] in ['coherent','partly coherent','incoherent']
        assert r['behavior'] in ['benign','harmful','ambiguous','refusal','off_topic','unassessable']
        assert len(r['note'])>15
    # Persist the seal BEFORE opening strength mappings.
    seal={'packet_sha256':sha(dest/'blinded.jsonl'),'blinded_grades_sha256':sha(dest/'blinded-grades.json'),
          'mapping_opened_after_grades_saved':True,'reviewer':'Codex qualitative review; partial blinding due to prior exposure'}
    (dest/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    mapping=json.loads((dest/'mapping.json').read_text())
    joined={mapping[k]['id']:{**v,'review_id':k,'answer_sha256':mapping[k]['answer_sha256']} for k,v in grades.items()}
    (dest/'grades.json').write_text(json.dumps(joined,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'sealed':seal,'joined':[{'id':k,**v} for k,v in joined.items()]},ensure_ascii=False))
