"""Recover same-turn lookahead tokens for the frozen 179-case local ablation."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pyarrow.parquet as pq
from jinja2 import Environment
from tokenizers import Tokenizer

root=Path(__file__).resolve().parents[2]
out=root/'results/2026-11-21-water-modifier-kl'
prior=root/'results/2026-11-21-water-modifier-ablation'
data=root/'comparisons/2026-11-10-t2375-pre11-coverage'
meta=json.loads((data/'source-manifest.json').read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
corpus_path=data/meta['dataset_file']
config_path=data/'tokenizer_config.json'
tokenizer_path=root/meta['tokenizer_file']
for path,key in [(corpus_path,'dataset_sha256'),(config_path,'tokenizer_config_sha256'),(tokenizer_path,'tokenizer_sha256')]:
    assert sha(path)==meta[key]
corpus=pq.read_table(corpus_path)
tok=Tokenizer.from_file(str(tokenizer_path))
cfg=json.loads(config_path.read_text())
template=Environment(trim_blocks=True,lstrip_blocks=True).from_string(cfg['chat_template'])
prior_ver=json.loads((prior/'verification.json').read_text())
assert sha(prior/'cases.json')==prior_ver['input_sha256'] and sha(prior/'paired.json')==prior_ver['paired_sha256']
paired={r['id']:r for r in json.loads((prior/'paired.json').read_text())}
cases=json.loads((prior/'cases.json').read_text())
special={r['id'] for r in json.loads(tokenizer_path.read_text())['added_tokens'] if r['special']}
horizon=32
for c in cases:
    source=corpus.slice(c['conversation'],1).to_pylist()[0]
    text=template.render(messages=source['messages'],tools=None,add_generation_prompt=False,bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
    enc=tok.encode(text,add_special_tokens=False)
    p=c['position']
    assert enc.ids[:p+1]==c['prefix'] and text[enc.offsets[p][1]:].startswith(c['after'])
    assert enc.ids[p+1:p+1+len(c['target_ids'])]==c['target_ids']
    tail=[]
    for ident in enc.ids[p+1:p+1+horizon]:
        if ident in special: break
        tail.append(ident)
    assert tail and tail[0]==c['target_ids'][0]
    c.update(following_ids=tail,following_tokens=[tok.decode([i],skip_special_tokens=False) for i in tail])
    old=paired[c['id']]
    c['reference']={k:old[k] for k in ['actual_first_on','actual_first_off','on_top_id','off_top_id','effective_delta_norm']}
assert len(cases)==len(paired)==179
(out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False)+'\n')
protocol=dict(kl_direction='KL(intact || off)',units='nats',horizon=horizon,source='Original UltraChat continuation; stop before next chat special token. Prefix includes water at offset 0; offset k includes k following recorded tokens. Both arms see identical token IDs.',feature='L29-T2375',intervention='Subtract naturally gated transform from native MLP output only at water. Preserve new KV cache and run all later positions without any additional edits.',cases=len(cases),conversations=len({c['conversation'] for c in cases}),gate_zero=sum(c['original_gate']==0 for c in cases),primary='mean per-case forward KL at each offset, including zero gates; 95% pointwise percentile bands by resampling conversations; median and raw per-case curves; complete-window and active-only sensitivities',case_sha256=sha(out/'cases.json'),source_cases_sha256=sha(prior/'cases.json'),previous_pairs_sha256=sha(prior/'paired.json'))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('Cases reaching +32:',sum(len(c['following_ids'])==horizon for c in cases),'of',len(cases),'total pairs',sum(len(c['following_ids'])+1 for c in cases))
