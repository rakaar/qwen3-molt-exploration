"""Fresh native generation and observation-only, token-exact gate replay."""
import argparse,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
S=Path(__file__).resolve().parent;OUT=S/'artifacts';OUT.mkdir(exist_ok=True)
P=json.loads((S/'protocol.json').read_text());START=time.monotonic()
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def checktime():assert time.monotonic()-START<60*P['max_runtime_minutes']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);args=ap.parse_args()
 assert not (OUT/'measurements.json').exists(),'Do not overwrite scientific outputs'
 base=read(S/'expected-base-verification.json');assert base['repo']==P['base_repo'] and base['revision']==P['base_revision']
 for item in base['files']:
  p=args.model_dir/item['file'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['file']
 save('base-verification.json',base)
 torch.manual_seed(0);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
 assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);special=set(tok.all_special_ids)
 attention=[b.self_attn.forward for b in model.model.layers];cases=read(S/'cases.json')
 native={'status':'running','protocol':P,'rows':[]}
 with torch.inference_mode():
  for c in cases:
   checktime();chat=tok.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   encoded=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
   ids=encoded['input_ids'];query_start=chat.rfind(c['question']);assert query_start>=0
   query_end=query_start+len(c['question'])
   query_positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>query_start and a<query_end and ids[i] not in special]
   assert query_positions and query_positions==list(range(min(query_positions),max(query_positions)+1))
   if 'saved_chat' in c:assert chat==c['saved_chat'] and ids==c['saved_prompt_token_ids']
   stepids=torch.tensor([ids],device='cuda');mask=torch.ones_like(stepids);cache=None;outids=[];stop='token_limit'
   for j in range(P['decoding']['max_new_tokens']):
    checktime();pred=model(input_ids=stepids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values
    nxt=int(pred.logits[0,-1].argmax());outids.append(nxt)
    if nxt in eos:stop='eos';break
    stepids=torch.tensor([[nxt]],device='cuda');mask=torch.cat([mask,torch.ones_like(stepids)],dim=1)
   answer=tok.decode(outids,skip_special_tokens=True);matches=outids==c.get('saved_output_token_ids',outids)
   row={**c,'chat':chat,'prompt_token_ids':ids,'output_token_ids':outids,'answer':answer,'stop_reason':stop,'baseline_generation_no_hooks':True,'saved_baseline_tokens_match':matches if 'saved_output_token_ids' in c else None,'query_character_span':[query_start,query_end],'query_token_positions':query_positions}
   native['rows'].append(row);save('native.json',native);assert matches,c['id']+' native output differs; review before proceeding'
   print('NATIVE',c['id'],json.dumps(answer),flush=True);del cache,pred
 native['status']='complete';save('native.json',native)
 assets=read(S/'weights/manifest.json');assert assets['repo']==P['molt_repo'] and assets['revision']==P['molt_revision']
 weights={}
 for f in assets['features']:
  p=S/'weights'/f['file'];assert sha(p)==f['sha256']
  with np.load(p) as d:weights[f['layer']]={k:torch.from_numpy(d[k].copy()).float().cuda() for k in d.files}
 state={};handles=[]
 for f in P['features']:
  layer=f['layer'];w=weights[layer]
  def observe(_module,inputs,layer=layer,w=w):
   z=(inputs[0].detach().float()-w['input_mean'])/w['input_std']
   raw=F.linear(z,w['e'].reshape(1,-1),w['bias']).squeeze(-1)
   alternate=(z*w['e']).sum(-1)+w['bias'][0]
   assert torch.allclose(raw,alternate,atol=3e-5,rtol=3e-5),'Projection arithmetic mismatch'
   gate=torch.where((raw>w['theta'][0])&(raw>0),raw,torch.zeros_like(raw))
   state[layer]={'raw':raw[0].cpu().tolist(),'gate':gate[0].cpu().tolist()}
   return None
  handles.append(model.model.layers[layer].post_attention_layernorm.register_forward_pre_hook(observe))
 result={'status':'running','protocol':P,'assets':assets,'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{n:sha(S/n) for n in ['run.py','protocol.json','cases.json','expected-base-verification.json']},'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},'rows':[]}
 def entries(ids,segment,offset):
  rs=[]
  for j,token in enumerate(ids):
   fs={f"L{f['layer']}-T{f['transform']}":{k:v[j] for k,v in state[f['layer']].items()} for f in P['features']}
   rs.append({'segment':segment,'position':j if segment=='prompt' else offset,'full_chat_position':offset+j if segment=='prompt' else offset,'token_id':token,'token':tok.decode([token]),'special':token in special,'features':fs})
  return rs
 try:
  with torch.inference_mode():
   for c in native['rows']:
    checktime();ids=c['prompt_token_ids'];mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long)
    pred=model(input_ids=torch.tensor([ids],device='cuda'),attention_mask=mask,use_cache=True);cache=pred.past_key_values
    assert int(pred.logits[0,-1].argmax())==c['output_token_ids'][0],c['id']+' first prediction differs with monitoring'
    row={**c,'prompt_tokens':entries(ids,'prompt',0),'answer_tokens':[],'predictions_verified':1}
    for j,token in enumerate(c['output_token_ids']):
     if token in eos:break
     mask=torch.cat([mask,torch.ones((1,1),device='cuda',dtype=mask.dtype)],dim=1)
     pred=model(input_ids=torch.tensor([[token]],device='cuda'),attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values
     if j+1<len(c['output_token_ids']):
      assert int(pred.logits[0,-1].argmax())==c['output_token_ids'][j+1],f"{c['id']} changed at prediction {j+1}"
      row['predictions_verified']+=1
     t=entries([token],'response',len(ids)+j)[0];t['position']=j;row['answer_tokens'].append(t)
    visible=[t for t in row['answer_tokens'] if not t['special']];assert ''.join(t['token'] for t in visible)==c['answer']
    row['query_tokens']=[row['prompt_tokens'][j] for j in c['query_token_positions']]
    row['summary']={}
    for f in P['features']:
     key=f"L{f['layer']}-T{f['transform']}";gs=[t['features'][key]['gate'] for t in visible]
     row['summary'][key]={'visible_tokens':len(gs),'active_tokens':sum(g>0 for g in gs),'mean_gate':sum(gs)/len(gs),'max_gate':max(gs),'final_prompt_gate':row['prompt_tokens'][-1]['features'][key]['gate'],'final_prompt_raw':row['prompt_tokens'][-1]['features'][key]['raw']}
     qs=[t['features'][key]['gate'] for t in row['query_tokens']]
     ps=[t['features'][key]['gate'] for t in row['prompt_tokens'] if not t['special']]
     row['summary'][key].update(query_tokens=len(qs),query_active_tokens=sum(g>0 for g in qs),query_mean_gate=sum(qs)/len(qs),query_max_gate=max(qs),prompt_active_tokens=sum(g>0 for g in ps),prompt_visible_tokens=len(ps),prompt_mean_gate=sum(ps)/len(ps))
    assert row['predictions_verified']==len(c['output_token_ids'])
    row['observation_only_predictions_match']=True;result['rows'].append(row);save('measurements.json',result)
    print('OBSERVED',c['id'],json.dumps(row['summary']),flush=True)
    assert all(b.self_attn.forward==fn for b,fn in zip(model.model.layers,attention));del cache,pred
 finally:
  for h in handles:h.remove()
 result.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START);save('measurements.json',result)
 print('COMPLETE',len(result['rows']),'unsteered cases; every native next-token argmax preserved',flush=True)
if __name__=='__main__':main()
