"""Native greedy list lookup with an exactly non-editing L29-T2375 observer."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,os,re,time
for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:os.environ.pop(k,None)
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--molt-checkpoint',type=Path,required=True);args=ap.parse_args()
 p=json.loads((ROOT/'protocol.json').read_text());expected=json.loads((ROOT/'expected-base-verification.json').read_text())
 OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists(),'Never overwrite prior captures'
 for row in expected['files']:assert (args.model_dir/row['file']).stat().st_size==row['bytes'] and sha(args.model_dir/row['file'])==row['sha256']
 assert expected['revision']==p['base_revision'] and sha(args.molt_checkpoint)==p['molt_checkpoint_sha256']
 assert sha(ROOT/'weights/L29-T2375.npz')==p['feature_sha256']
 assert torch.__version__=='2.14.0+cu126' and transformers.__version__=='4.57.6'
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
 assert len(model.model.layers)==36 and model.config.hidden_size==2560
 with np.load(ROOT/'weights/L29-T2375.npz',allow_pickle=False) as a:w={k:torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
 assert w['U'].shape==(32,2560) and w['V'].shape==(2560,32)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);special=set(tok.all_special_ids);started=time.monotonic()
 def hooks():return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
 def decode(i):return tok.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
 def encode(case):
  text=case['prompt'] if case['prompt_format']=='raw' else tok.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];a=text.index(case['prompt']);b=a+len(case['prompt']);meta=[]
  for pos,(ident,(s,e)) in enumerate(zip(ids,enc['offset_mapping'])):
   region='instruction';item=None;word_item=None
   if case['prompt_format']=='raw':region='calibration_prompt'
   elif e<=a:region='template_before'
   elif s>=b:region='answer_prefix'
   else:
    ls,le=case['list_span'];qs,qe=case['question_span']
    if e>a+ls and s<a+le:region='list'
    elif e>a+qs and s<a+qe:region='question'
    for row in case['item_spans']:
     if e>a+row['start'] and s<a+row['end']:item=row['item']
     if e>a+row['word_start'] and s<a+row['word_end']:word_item=row['item']
   meta.append(dict(position=pos,phase='input',token_id=ident,token=decode(ident),offset=[s,e],region=region,item=item,word_item=word_item,special=ident in special,final_input=pos==len(ids)-1))
  return text,ids,meta
 @torch.inference_mode()
 def generate(ids,meta,observe):
  assert hooks()==0
  state=dict(current=meta,trace=[],terminal=False);steps=[];generated=[];cache=None;mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long);current=torch.tensor([ids],device='cuda')
  def observer(module,inputs):
   x=inputs[0].detach().float();assert x.shape[0]==1 and x.shape[1]==len(state['current'])
   z=(x-w['input_mean'])/w['input_std'];raw=F.linear(z,w['e'].reshape(1,-1),w['bias']).squeeze(-1)
   gate=torch.where((raw>w['theta'][0]) & (raw>0),raw,0.)
   op=(z@w['V']@w['U'])*w['output_std'];norm=op.norm(dim=-1)
   values=torch.stack([raw,gate,norm,gate*norm],-1)[0].cpu().tolist()
   for m,v in zip(state['current'],values):state['trace'].append(dict(**m,raw=v[0],gate=v[1],operator_norm=v[2],contribution_norm=v[3],trace_only_terminal=state['terminal']))
   return None
  handle=model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observer) if observe else None
  def output_meta(j,ident):return [dict(position=len(ids)+j,phase='response',output_index=j,token_id=ident,token=decode(ident),region='response',item=None,word_item=None,special=ident in special,final_input=False)]
  try:
   for step in range(p['max_new_tokens']):
    assert time.monotonic()-started<p['maximum_runtime_minutes']*60
    pred=model(input_ids=current,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values;logits=pred.logits[0,-1].float();assert torch.isfinite(logits).all()
    chosen=int(logits.argmax());prob=logits.softmax(-1);v=logits.cpu().numpy();entry=dict(token_id=chosen,token=decode(chosen),probability=float(prob[chosen]),logits_sha256=hashlib.sha256(v.tobytes()).hexdigest())
    if step==0:
     vals,inds=prob.topk(5);entry['top5']=[dict(token_id=int(i),token=decode(int(i)),probability=float(q)) for q,i in zip(vals,inds)]
    steps.append(entry);generated.append(chosen)
    if chosen in eos:break
    current=torch.tensor([[chosen]],device='cuda');mask=torch.cat([mask,torch.ones_like(current)],1);state['current']=output_meta(len(generated)-1,chosen)
   # Trace the last output (EOS or truncation boundary) without decoding another token.
   current=torch.tensor([[generated[-1]]],device='cuda');mask=torch.ones((1,len(ids)+len(generated)),device='cuda',dtype=torch.long);state['current']=output_meta(len(generated)-1,generated[-1]);state['terminal']=True
   tail=model(input_ids=current,attention_mask=mask,past_key_values=cache,use_cache=True).logits[0,-1].float().cpu().numpy()
  finally:
   if handle:handle.remove()
  native=dict(output_token_ids=generated,response=tok.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False),stop_reason='eos' if generated[-1] in eos else 'token_limit',steps=steps,unused_terminal_logits_sha256=hashlib.sha256(tail.tobytes()).hexdigest())
  trace=state['trace'];allids=ids+generated
  if observe:
   assert len(trace)==len(allids) and [x['token_id'] for x in trace]==allids
   for i,t in enumerate(trace):t['followed_by']=decode(allids[i+1]) if i+1<len(allids) else None
  assert hooks()==0
  return native,trace
 result=dict(status='running',protocol=p,protocol_sha256=sha(ROOT/'protocol.json'),source_sha256=sha(ROOT/'run.py'),runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),native_dtype='BF16',observer_dtype='FP32',attention='eager',TF32=False,threshold=float(w['theta'][0]),model_checkpoint_verified=True,full_molt_checkpoint_verified=True),calibration=None,cases=[])
 save(OUT/'results.json',result)
 for case in [p['calibration']]+p['cases']:
  formatted,ids,meta=encode(case);native,_=generate(ids,meta,False);observed,trace=generate(ids,meta,True)
  assert native==observed,'Observer changed native tokens, probabilities, or logits'
  row=dict(**case,formatted_prompt=formatted,input_token_ids=ids,input_token_offsets=[m['offset'] for m in meta],native=native,observed=observed,trace=trace,observer_all_logits_and_outputs_exact=True)
  if case['id']=='calibration-month':
   assert trace[len(ids)-1]['gate']>0 and re.match(r'\s*November\b',native['response'],re.I)
   result['calibration']=row
  else:result['cases'].append(row)
  save(OUT/'results.json',result);save(OUT/'progress.json',dict(completed=len(result['cases']),total=30,calibration_passed=result['calibration'] is not None))
  print(json.dumps(dict(case=case['id'],response=native['response'],active_input_tokens=sum(t['gate']>0 for t in trace if t['phase']=='input'),active_response_tokens=sum(t['gate']>0 for t in trace if t['phase']=='response' and not t['special'])),ensure_ascii=False),flush=True)
 assert len(result['cases'])==30 and hooks()==0
 result.update(status='complete',hooks_remaining=0,runtime_seconds=time.monotonic()-started,interventions=0,finished_at_utc=datetime.now(timezone.utc).isoformat());save(OUT/'results.json',result)
 files=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.iterdir()) if f.is_file()]
 save(OUT/'transfer-manifest.json',dict(files=files));print(json.dumps(dict(status='complete',pilot_cases=30,calibration_cases=1,runtime_seconds=result['runtime_seconds'])),flush=True)
if __name__=='__main__':main()
