"""Uniform occurrence sample from previously unused UltraChat test_sft conversations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import pyarrow.parquet as pq
from jinja2 import Environment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/2026-11-21-water-blind'
DATA = ROOT / 'comparisons/2026-11-10-t2375-pre11-coverage'
META = json.loads((DATA / 'source-manifest.json').read_text())
corpus = DATA / META['dataset_file']
tokenizer_path = ROOT / META['tokenizer_file']
config_path = DATA / 'tokenizer_config.json'
for path, key in [(corpus,'dataset_sha256'),(tokenizer_path,'tokenizer_sha256'),(config_path,'tokenizer_config_sha256')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META[key]
tok = Tokenizer.from_file(str(tokenizer_path))
cfg = json.loads(config_path.read_text())
template = Environment(trim_blocks=True,lstrip_blocks=True).from_string(cfg['chat_template'])
word = re.compile(r'(?<![\w\-‐‑])water(?![\w\-‐‑])', re.I)
records = []
prefixes = {}
for c, row in enumerate(pq.read_table(corpus).to_pylist()):
    if c < 11345 or not any(word.search(m['content']) for m in row['messages']):
        continue
    chat = template.render(messages=row['messages'],tools=None,add_generation_prompt=False,
                           bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
    enc = tok.encode(chat, add_special_tokens=False)
    by_end = {end:(p,start) for p,(start,end) in enumerate(enc.offsets[:1024])}
    for hit in word.finditer(chat):
        if hit.end() not in by_end:
            continue
        p,start = by_end[hit.end()]
        if start > hit.start() or chat[start:hit.start()].strip() or tok.decode([enc.ids[p]]).strip().casefold() != 'water':
            continue
        records.append(dict(conversation=c,position=p,token_id=enc.ids[p],token=tok.decode([enc.ids[p]]),
                            before=chat[max(0,hit.start()-105):hit.start()],after=chat[hit.end():hit.end()+105]))
        prefixes[c] = enc.ids[:1024]
seed=20261121
draw=random.Random(seed).sample(records,240)
for i,r in enumerate(draw,1): r['id']=i
def save(name,x): (OUT / name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
save('sample.json',draw)
save('input_ids.json',{str(c):prefixes[c][:max(r['position'] for r in draw if r['conversation']==c)+1] for c in sorted({r['conversation'] for r in draw})})
save('sampling.json',dict(seed=seed,corpus=META,first_row=11345,past_last_row=23110,prefix_limit=1024,
     pool=len(records),pool_conversations=len(prefixes),sample=len(draw),sample_conversations=len({r['conversation'] for r in draw}),
     time=datetime.now(timezone.utc).isoformat(),sample_sha256=hashlib.sha256((OUT/'sample.json').read_bytes()).hexdigest(),
     positive='water directly modifies the next noun in the same noun phrase, including gerund nouns such as water recycling. Capitalized names included. An intervening adjective, punctuation, or coordination is negative under the direct adjacency rule. Lexically hyphenated or joined forms are outside the frame.',
     endpoints='Gate > 0 proportion, mean gate across all positions including zeros, mean among firing positions; descriptive medians. Conversation-cluster percentile bootstrap for group differences.'))
print(len(records),len(prefixes),len(draw),len({r['conversation'] for r in draw}))
