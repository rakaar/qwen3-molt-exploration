"""Recover full causal prefixes for 32 known firing and four known zero positions."""
from pathlib import Path
import json,collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2 import Environment
root=Path(__file__).resolve().parents[2]
source=root/'comparisons/2026-11-10-t2375-pre11-coverage'
known=json.loads((root/'results/2026-11-10-t2375-pre11-coverage/recovered-windows.json').read_text())
tok=Tokenizer.from_file(str(root/'comparisons/2026-11-02-eleven-contribution-readout/tokenizer.json'))
cfg=json.loads((source/'tokenizer_config.json').read_text())
template=Environment(trim_blocks=True,lstrip_blocks=True).from_string(cfg['chat_template'])
data=pq.read_table(next(source.glob('*.parquet')))
refs={}
for w in known['windows'].values():
 for m in w['matches']:
  for i,(t,g) in enumerate(zip(w['str_tokens'],w['activations'])):
   if 'water' in t.lower():
    c=m['conversation'];p=m['window_start']+i
    refs[c,p]=(m['window_token_ids'][i],g)
reference=[]
for (c,p),(ident,g) in refs.items():
 row=data.slice(c,1).to_pylist()[0]
 text=template.render(messages=row['messages'],tools=None,add_generation_prompt=False,bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
 ids=tok.encode(text,add_special_tokens=False).ids[:p+1]
 assert ids[-1]==ident
 reference.append(dict(conversation=c,position=p,token_id=ident,gate=g,ids=ids))
assert len(reference)==36
(root/'results/2026-11-21-water-blind/calibration-input.json').write_text(json.dumps(reference)+'\n')
