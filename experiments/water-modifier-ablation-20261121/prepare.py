"""Recover each recorded next word from the full corpus tokenization."""
from pathlib import Path
import json,re,hashlib
import pyarrow.parquet as pq
from jinja2 import Environment
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[2]
out=root/'results/2026-11-21-water-modifier-ablation'
prev=root/'results/2026-11-21-water-thousand'
data=root/'comparisons/2026-11-10-t2375-pre11-coverage'
meta=json.loads((data/'source-manifest.json').read_text())
tok=Tokenizer.from_file(str(root/meta['tokenizer_file']))
cfg=json.loads((data/'tokenizer_config.json').read_text())
template=Environment(trim_blocks=True,lstrip_blocks=True).from_string(cfg['chat_template'])
corpus=pq.read_table(data/meta['dataset_file'])
inputs=json.loads((prev/'input_ids.json').read_text())
cases=[]
for r in json.loads((prev/'observations.json').read_text()):
 if r['category']!='M':continue
 c=r['conversation'];p=r['position']
 row=corpus.slice(c,1).to_pylist()[0]
 assert r['prompt_id']==row['prompt_id']
 text=template.render(messages=row['messages'],tools=None,add_generation_prompt=False,bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
 enc=tok.encode(text,add_special_tokens=False)
 ids=enc.ids[:p+1]
 assert ids==inputs[str(c)][:p+1] and ids[-1]==r['token_id']
 end=enc.offsets[p][1]
 assert text[end:end+len(r['after'])]==r['after']
 m=re.match(r'[^\w]*([^\W_]+)',text[end:],re.UNICODE)
 assert m,r
 stop=next(i+1 for i in range(p+1,len(enc.ids)) if enc.offsets[i][1]>=end+m.end())
 assert enc.offsets[stop-1][1]==end+m.end(),(r['id'],m.group(),enc.tokens[p+1:stop])
 cases.append(dict(id=r['id'],conversation=c,position=p,prefix=ids,water_token=r['token'],original_gate=r['gate'],before=r['before'],after=r['after'],word=m.group(1),strict_adjacent=r['id'] not in (785,787,934,973),target_ids=enc.ids[p+1:stop]))
assert len(cases)==179
(out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False)+'\n')
from collections import Counter
print('word target lengths',Counter(len(r['target_ids']) for r in cases))
print('samples',[(r['word'],r['target_ids']) for r in cases if len(r['target_ids'])>1][:15])
