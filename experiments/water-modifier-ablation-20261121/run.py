"""Paired word/next-token probabilities after removing L29–T2375 only at water."""
from pathlib import Path
from datetime import datetime, timezone
import copy,hashlib,json,math,re
import numpy as np
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

base=Path('/workspace/water-thousand')
here=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=json.loads((base/'model-verified.json').read_text())
assert spec['revision']=='1cfa9a7208912126459214e8b04321603b3df60c'
for f in spec['files']:assert sha(base/'model'/f['file'])==f['sha256']
weights=base/'L29-T2375.npz'
assert sha(weights)=='8ae225a36fcfd920adf9cd06aacfbcc53011343b0fbb4e8880cc7ff97111e57c'
with np.load(weights) as f:w={k:torch.tensor(f[k],device='cuda',dtype=torch.float32) for k in f.files}
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(0)
model=AutoModelForCausalLM.from_pretrained(base/'model',dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).eval().cuda().requires_grad_(False)
tok=AutoTokenizer.from_pretrained(base/'model',local_files_only=True)
special=set(tok.all_special_ids)
def decode(ids):return tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
def forward(ids,cache=None):
    out=model(input_ids=torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
    return out.logits[0,-1].float(),out.past_key_values
def prefix(case,off):
    rec={}
    def observe(module,args):
        x=args[0][0,-1].float()
        z=(x-w['input_mean'])/w['input_std']
        raw=torch.dot(z,w['e'])+w['bias'][0]
        gate=torch.where((raw>w['theta'][0]) & (raw>0),raw,0.)
        rec['gate']=float(gate)
        rec['contrib']=gate*((z@w['V'])@w['U'])*w['output_std']
    def edit(module,args,out):
        if off:
            changed=out.clone()
            changed[0,-1]=(out[0,-1].float()-rec['contrib']).to(out.dtype)
            assert torch.equal(changed[:,:-1],out[:,:-1])
            rec['delta_norm']=float((changed[0,-1].float()-out[0,-1].float()).norm())
            return changed
    layer=model.model.layers[29]
    handles=[layer.post_attention_layernorm.register_forward_pre_hook(observe),layer.mlp.register_forward_hook(edit)]
    try: logits,cache=forward(case['prefix'])
    finally:
        for h in handles:h.remove()
    return logits,cache,rec
def predicted(logits,cache):
    # Look ahead by one token to catch a continuation piece such as "nuts" in "chestnuts".
    used=[]; ends=[]; cache=copy.deepcopy(cache)
    for step in range(24):
        token=int(logits.argmax())
        if token in special:
            if not used:return [token],decode([token]),'special'
            text=decode(used)
            match=re.search(r'[^\W_]+',text)
            if match:return used,match.group(),'word'
            return used+[token],decode(used+[token]),'special'
        used.append(token);text=decode(used);ends.append(len(text))
        match=re.search(r'[^\W_]+',text)
        if match and match.end()<len(text):
            stop=next(i+1 for i,end in enumerate(ends) if end>=match.end())
            return used[:stop],match.group(),'word'
        logits,cache=forward([token],cache)
    raise ValueError('Greedy continuation did not finish its first word')
def prob_sequence(ids,logits,cache):
    cache=copy.deepcopy(cache) if len(ids)>1 else None
    parts=[]
    for j,token in enumerate(ids):
        parts.append(float(logits.log_softmax(-1)[token]))
        if j+1<len(ids):logits,cache=forward([token],cache)
    return math.exp(sum(parts))
def save(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
cases=json.loads((here/'cases.json').read_text());assert len(cases)==179
rows=[]
with torch.inference_mode():
    for c in cases:
        assert decode(c['prefix'][-1:])==c['water_token']
        on,onc,rec=prefix(c,False);off,offc,changed=prefix(c,True)
        assert rec['gate']==changed['gate'] and abs(rec['gate']-c['original_gate'])<1e-4,(c['id'],rec['gate'],c['original_gate'])
        if rec['gate']==0:assert torch.equal(on,off) and changed['delta_norm']==0
        pred_ids,word,kind=predicted(on,onc)
        off_ids,off_word,off_kind=predicted(off,offc)
        actual=c['target_ids']
        def pair(ids):return prob_sequence(ids,on,onc),prob_sequence(ids,off,offc)
        a,b=pair(actual);d,e=pair(pred_ids)
        on_p=on.softmax(-1);off_p=off.softmax(-1)
        row={k:v for k,v in c.items() if k!='prefix'}
        row.update(gate=rec['gate'],removed_norm=float(rec['contrib'].norm()),effective_delta_norm=changed['delta_norm'],
                   actual_sequence=decode(actual),actual_on=a,actual_off=b,pred_word=word,pred_kind=kind,pred_ids=pred_ids,pred_sequence=decode(pred_ids),
                   predicted_on=d,predicted_off=e,off_greedy_word=off_word,off_greedy_kind=off_kind,off_greedy_sequence=decode(off_ids),
                   on_top_id=pred_ids[0],off_top_id=off_ids[0],on_top_token=decode(pred_ids[:1]),off_top_token=decode(off_ids[:1]),
                   actual_first_on=float(on_p[actual[0]]),actual_first_off=float(off_p[actual[0]]),on_top_first_on=float(on_p[pred_ids[0]]),
                   on_top_first_off=float(off_p[pred_ids[0]]),on_max_ties=int((on==on.max()).sum()),off_max_ties=int((off==off.max()).sum()))
        if rec['gate']==0: assert a==b and d==e and pred_ids==off_ids
        for j in [a,b,d,e]:assert 0<j<=1
        rows.append(row)
        if len(rows)%30==0:print('measured',len(rows),flush=True)
assert len(rows)==179 and sum(r['gate']==0 for r in rows)==22
save(here/'paired.json',rows)
save(here/'verification.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),cases=179,gate_zero_controls=22,model_revision=spec['revision'],weights_sha256=sha(weights),input_sha256=sha(here/'cases.json'),paired_sha256=sha(here/'paired.json'),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),scope='native MLP output minus the learned transform contribution at the water position only; subsequent tokens use the resulting KV cache',score='Probability of the observed or intact-greedy token path through its first word, including any leading punctuation; no terminal boundary token, no marginalization over other tokenizations.'))
print('complete',flush=True)
