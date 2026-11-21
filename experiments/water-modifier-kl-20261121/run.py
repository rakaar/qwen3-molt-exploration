"""Full-vocabulary forward KL along identical recorded suffixes after a one-position ablation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

here=Path(__file__).resolve().parent
base=Path('/workspace/water-thousand')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,x):
    path=here/name
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)
protocol=json.loads((here/'protocol.json').read_text())
assert sha(here/'cases.json')==protocol['case_sha256']
cases=json.loads((here/'cases.json').read_text())
spec=json.loads((base/'model-verified.json').read_text())
assert spec['revision']=='1cfa9a7208912126459214e8b04321603b3df60c'
for row in spec['files']:assert sha(base/'model'/row['file'])==row['sha256']
weights=base/'L29-T2375.npz'
assert sha(weights)=='8ae225a36fcfd920adf9cd06aacfbcc53011343b0fbb4e8880cc7ff97111e57c'
with np.load(weights) as arrays:
    w={k:torch.tensor(arrays[k],device='cuda',dtype=torch.float32) for k in arrays.files}
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(0)
model=AutoModelForCausalLM.from_pretrained(base/'model',dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).eval().cuda().requires_grad_(False)
def forward(ids,cache=None,last=False):
    return model(input_ids=torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1 if last else 0)
def arm(case,off):
    rec={}
    layer=model.model.layers[29]
    def observe(module,args):
        x=args[0][0,-1].float()
        z=(x-w['input_mean'])/w['input_std']
        raw=torch.dot(z,w['e'])+w['bias'][0]
        gate=torch.where((raw>w['theta'][0]) & (raw>0),raw,0.)
        rec['gate']=float(gate)
        rec['contribution']=gate*((z@w['V'])@w['U'])*w['output_std']
    def edit(module,args,output):
        if off:
            edited=output.clone()
            edited[0,-1]=(output[0,-1].float()-rec['contribution']).to(output.dtype)
            assert torch.equal(edited[:,:-1],output[:,:-1])
            rec['effective_norm']=float((edited[0,-1].float()-output[0,-1].float()).norm())
            return edited
    hooks=[layer.post_attention_layernorm.register_forward_pre_hook(observe),layer.mlp.register_forward_hook(edit)]
    try:prefix=forward(case['prefix'],last=True)
    finally:
        for h in hooks:h.remove()
    # The ablated prefix's cache carries the intervention into later positions. All hooks are removed.
    future=forward(case['following_ids'],cache=prefix.past_key_values)
    return torch.cat((prefix.logits[0],future.logits[0])),rec
rows=[];max_prob_err=0.;zeros=0
with torch.inference_mode():
    for c in cases:
        on,a=arm(c,False);off,b=arm(c,True)
        assert on.shape==off.shape==(1+len(c['following_ids']),model.config.vocab_size)
        assert a['gate']==b['gate'] and abs(a['gate']-c['original_gate'])<1e-4
        assert abs(b['effective_norm']-c['reference']['effective_delta_norm'])<1e-4
        for logits,k in [(on,'on'),(off,'off')]:
            assert int(logits[0].argmax())==c['reference'][k+'_top_id']
            err=abs(float(logits[0].float().softmax(-1)[c['following_ids'][0]])-c['reference']['actual_first_'+k])
            assert err<2e-6,(c['id'],k,err)
            max_prob_err=max(max_prob_err,err)
        log_p=on.double().log_softmax(-1);log_q=off.double().log_softmax(-1)
        p=log_p.exp();q=log_q.exp()
        kl=(p*(log_p-log_q)).sum(-1)
        reverse=(q*(log_q-log_p)).sum(-1)
        tv=(p-q).abs().sum(-1)/2
        assert torch.isfinite(kl).all() and float(kl.min())>=-1e-12 and float(reverse.min())>=-1e-12
        assert (2*tv.square()<=kl+1e-12).all()
        if a['gate']==0:
            assert torch.equal(on,off) and (kl==0).all() and (tv==0).all()
            zeros+=1
        metrics=torch.stack([kl.clamp_min(0),reverse.clamp_min(0),tv],1).cpu().tolist()
        top_on=on.argmax(-1).tolist();top_off=off.argmax(-1).tolist()
        for j,(k,rev,v) in enumerate(metrics):
            rows.append(dict(id=c['id'],conversation=c['conversation'],water_position=c['position'],offset=j,
                context_token=c['water_token'] if j==0 else c['following_tokens'][j-1],
                gate=a['gate'],strict_adjacent=c['strict_adjacent'],kl_nats=k,reverse_kl_nats=rev,
                total_variation=v,intact_top_id=top_on[j],off_top_id=top_off[j]))
        if len([1 for r in rows if r['offset']==0])%40==0:print('measured through sample id',c['id'],flush=True)
assert zeros==22 and len(rows)==sum(1+len(c['following_ids']) for c in cases)==5607
save('measurements.json',rows)
save('verification.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),cases=179,positions=len(rows),zero_control_cases=zeros,all_zero_control_logits_identical=True,water_baseline_max_probability_error=max_prob_err,sha256=sha(here/'measurements.json'),input_sha256=sha(here/'cases.json'),weights_sha256=sha(weights),base_revision=spec['revision'],gpu=torch.cuda.get_device_name(),vocabulary=model.config.vocab_size,torch=torch.__version__,transformers=transformers.__version__,logits_precision='native BF16; log-softmax and divergence FP64'))
print('complete',flush=True)
