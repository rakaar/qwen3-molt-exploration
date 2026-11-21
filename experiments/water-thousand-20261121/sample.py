"""Draw 1,000 water occurrences, including embedded forms, without inspecting gates."""
from pathlib import Path
import hashlib,json,random,re
from datetime import datetime,timezone
from collections import Counter
import pyarrow.parquet as pq
from jinja2 import Environment
from tokenizers import Tokenizer

root=Path(__file__).resolve().parents[2]
out=root/'results/2026-11-21-water-thousand'
data=root/'comparisons/2026-11-10-t2375-pre11-coverage'
meta=json.loads((data/'source-manifest.json').read_text())
corpus=data/meta['dataset_file']
for p,k in [(corpus,'dataset_sha256'),(root/meta['tokenizer_file'],'tokenizer_sha256'),(data/'tokenizer_config.json','tokenizer_config_sha256')]:
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta[k]
tok=Tokenizer.from_file(str(root/meta['tokenizer_file']))
cfg=json.loads((data/'tokenizer_config.json').read_text())
template=Environment(trim_blocks=True,lstrip_blocks=True).from_string(cfg['chat_template'])
prior={r['conversation'] for r in json.loads((root/'results/2026-11-21-water-blind/sample.json').read_text())}
collect=[]; inputs={};skip=0
for c,row in enumerate(pq.read_table(corpus).to_pylist()):
    if c<11345 or c in prior or not any('water' in m['content'].lower() for m in row['messages']):continue
    chat=template.render(messages=row['messages'],tools=None,add_generation_prompt=False,bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
    enc=tok.encode(chat,add_special_tokens=False)
    ends={j:i for i,(a,b) in enumerate(enc.offsets[:1024]) for j in range(a,b)}
    for hit in re.finditer('water',chat,re.I|re.ASCII):
        p=ends.get(hit.end()-1)
        if p is None:continue
        a,b=enc.offsets[p]; token=tok.decode([enc.ids[p]])
        assert chat[a:b]==token and a<=hit.end()-1<b
        collect.append(dict(conversation=c,position=p,token_id=enc.ids[p],token=token,before=chat[max(0,a-100):a],after=chat[b:b+100],word_before=chat[max(0,hit.start()-100):hit.start()],word_after=chat[hit.end():hit.end()+100],surface=hit.group(),match_offset=hit.start(),prompt_id=row['prompt_id']))
        inputs[c]=enc.ids[:1024]
draw=random.Random(2026112102).sample(collect,1000)
assert len({(r['conversation'],r['position']) for r in draw})==1000
for i,r in enumerate(draw,1):r['id']=i
def save(name,data): (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
save('sample.json',draw)
save('input_ids.json',{c:inputs[c][:max(r['position'] for r in draw if r['conversation']==c)+1] for c in sorted({r['conversation'] for r in draw})})
save('sampling.json',dict(seed=2026112102,population=len(collect),eligible_conversations=len(inputs),sample_size=1000,sampled_conversations=len({r['conversation'] for r in draw}),excluded_original_rows=11345,excluded_previous_random_conversations=len(prior),corpus=meta,frame='All case-insensitive ASCII water occurrences with final r in the first 1024 rendered tokens. Includes compounds, proper names, hyphenated forms and inflections. Gate is read at the token containing the final r, which can already contain more letters. Uniform occurrence sampling without replacement.',sample_sha256=hashlib.sha256((out/'sample.json').read_bytes()).hexdigest(),time=datetime.now(timezone.utc).isoformat()))
print(len(collect),len(inputs),len({r['conversation'] for r in draw}))
