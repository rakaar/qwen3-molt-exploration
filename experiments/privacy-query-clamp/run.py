"""Zero-target MOLT ablation with user-message and generation timing controls."""
import argparse,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'protocol.json').read_text());C=json.loads((ROOT/'case.json').read_text());OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True);START=time.monotonic()
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(d):
 t=OUT/'results.tmp';t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(OUT/'results.json')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--weights-dir',type=Path,required=True);a=ap.parse_args();assert not (OUT/'results.json').exists()
 base=json.loads((ROOT/'expected-base-verification.json').read_text());assert base['repo']==P['base_repo'] and base['revision']==P['base_revision']
 for f in base['files']:
  p=a.model_dir/f['file'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['file']
 torch.manual_seed(0);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
 assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
 assets=json.loads((ROOT/'weights-manifest.json').read_text());assert assets['revision']==P['molt_revision'];weights={}
 for f in assets['features']:
  p=a.weights_dir/f['file'];assert sha(p)==f['sha256']
  with np.load(p) as d:weights[f['layer']]={k:torch.from_numpy(d[k].copy()).float().cuda() for k in d.files}
 chat=tok.apply_chat_template([{'role':'system','content':C['system']},{'role':'user','content':C['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True);encoded=enc['input_ids'];offsets=enc['offset_mapping'];assert encoded==C['prompt_token_ids'] and chat==C['chat'] and chat.count(C['user'])==1
 begin=chat.index(C['user']);end=begin+len(C['user']);user=[i for i,(a,b) in enumerate(offsets) if b>begin and a<end]
 assert user==list(range(user[0],user[-1]+1));assert offsets[user[0]][0]==begin and offsets[user[-1]][1]==end
 assert tok.decode([encoded[i] for i in user])==C['user']
 first,last=user[0],user[-1];eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);attention=[b.self_attn.forward for b in model.model.layers]
 bounds={'user_first_token':first,'user_last_token':last,'user_token_count':len(user),'prompt_token_count':len(encoded),'system_and_user_header_positions_unedited':list(range(first)),'post_user_prompt_positions':list(range(last+1,len(encoded))),'user_char_span':[begin,end],'prompt_tokens':[{'position':i,'token_id':t,'token':tok.decode([t]),'char_span':offsets[i],'segment':'before_user' if i<first else 'user' if i<=last else 'post_user_prompt'} for i,t in enumerate(encoded)]}
 result={'status':'running','protocol':P,'case':C,'boundaries':bounds,'assets':assets,'base_files_hash_verified':True,'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','case.json','weights-manifest.json','prior-output-only.json']},'started_at_utc':datetime.now(timezone.utc).isoformat(),'rows':[]}
 def selection(condition,step,n):
  if step>0:return [condition in ['output_only','user_to_end']]*n
  if condition=='user_only':return [first<=i<=last for i in range(n)]
  if condition=='user_to_end':return [i>=first for i in range(n)]
  return [False]*n
 @torch.inference_mode()
 def generate(condition):
  ids=torch.tensor([encoded],device='cuda');mask=torch.ones_like(ids);cache=None;tokens=[];traces=[];state={};handles=[];step=0;stop='token_limit';first_top=[]
  if condition!='native':
   for f in P['features']:
    layer=f['layer'];name=f"L{layer}-T{f['transform']}";w=weights[layer]
    def capture(_m,inputs,name=name,w=w):
     z=(inputs[0].detach().float()-w['input_mean'])/w['input_std'];raw=F.linear(z,w['e'].reshape(1,-1),w['bias']).squeeze(-1)
     assert torch.allclose(raw,(z*w['e']).sum(-1)+w['bias'][0],atol=3e-5,rtol=3e-5)
     gate=torch.where((raw>w['theta'][0])&(raw>0),raw,torch.zeros_like(raw));low=(z@w['V']@w['U'])*w['output_std']
     state[name]={'raw':raw,'gate':gate,'low':low,'contribution':gate.unsqueeze(-1)*low,'threshold':float(w['theta'][0])}
    def edit(_m,_inputs,output,name=name,layer=layer):
     x=state[name];n=output.shape[1];selected=selection(condition,step,n);sel=torch.tensor(selected,device='cuda',dtype=torch.bool).reshape(1,n)
     delta=-x['contribution']*sel.unsqueeze(-1);changed=None
     if any(selected):
      changed=(output.float()+delta).to(output.dtype);assert torch.isfinite(changed).all()
      assert torch.equal(output[~sel],changed[~sel]),'Unselected positions changed'
      actual=(changed.float()-output.float()).norm(dim=-1)[0].cpu().tolist()
     else:actual=[0.]*n
     raw=x['raw'][0].cpu().tolist();gate=x['gate'][0].cpu().tolist();low=x['low'].norm(dim=-1)[0].cpu().tolist();contrib=x['contribution'].norm(dim=-1)[0].cpu().tolist();intended=delta.norm(dim=-1)[0].cpu().tolist();native=output.detach().float().norm(dim=-1)[0].cpu().tolist()
     for j in range(n):
      full=j if step==0 else len(encoded)+step-1;segment='response' if step>0 else bounds['prompt_tokens'][j]['segment'];token=int(ids[0,j]);eligible=selected[j]
      traces.append({'feature':name,'layer':layer,'segment':segment,'answer_position':step-1 if step>0 else None,'full_chat_position':full,'token_id':token,'token':tok.decode([token]),'natural_raw':raw[j],'natural_gate':gate[j],'threshold':x['threshold'],'clamp_applied':eligible,'gate_delta':-gate[j] if eligible else 0.,'effective_estimated_gate':0. if eligible else gate[j],'lowrank_norm':low[j],'natural_contribution_norm':contrib[j],'intended_delta_norm':intended[j],'actual_bf16_delta_norm':actual[j],'native_mlp_norm':native[j]})
     return changed
    block=model.model.layers[layer];handles.extend([block.post_attention_layernorm.register_forward_pre_hook(capture),block.mlp.register_forward_hook(edit)])
  try:
   for step in range(P['decoding']['max_new_tokens']):
    assert time.monotonic()-START<P['max_runtime_minutes']*60
    pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values;token=int(pred.logits[0,-1].argmax());tokens.append(token)
    if step==0:
     probs=pred.logits[0,-1].float().softmax(-1);vals,indices=probs.topk(5);first_top=[{'token_id':int(i),'token':tok.decode([int(i)]),'probability':float(v)} for i,v in zip(indices,vals)]
    if token in eos:stop='eos';break
    ids=torch.tensor([[token]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
  finally:
   for h in handles:h.remove()
  assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,attention))
  if condition=='native':assert tokens==C['output_token_ids']
  if condition=='output_only':assert tokens==json.loads((ROOT/'prior-output-only.json').read_text())['output_token_ids']
  if condition!='native':
   assert len(traces)==3*(len(encoded)+len(tokens)-1)
   for t in traces:
    assert not(t['segment']=='before_user' and t['clamp_applied'])
    expected=selection(condition,0,len(encoded))[t['full_chat_position']] if t['segment']!='response' else selection(condition,1,1)[0]
    assert t['clamp_applied']==expected
    if expected:assert t['effective_estimated_gate']==0 and t['gate_delta']==-t['natural_gate']
    else:assert t['actual_bf16_delta_norm']==0
  summary={}
  for f in P['features']:
   name=f"L{f['layer']}-T{f['transform']}";summary[name]={}
   for segment in ['before_user','user','post_user_prompt','response']:
    ts=[t for t in traces if t['feature']==name and t['segment']==segment];active=[t for t in ts if t['natural_gate']>0];edits=[t for t in ts if t['actual_bf16_delta_norm']>0]
    summary[name][segment]={'processed_positions':len(ts),'clamped_positions':sum(t['clamp_applied'] for t in ts),'active_natural_gates':len(active),'actual_nonzero_edits':len(edits),'max_actual_edit_norm':max((t['actual_bf16_delta_norm'] for t in ts),default=0),'first_actual_edit_full_position':edits[0]['full_chat_position'] if edits else None}
  answer=tok.decode(tokens,skip_special_tokens=True)
  return {'condition':condition,'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),'output_token_ids':tokens,'stop_reason':stop,'contains_exact_secret':P['secret_marker'] in answer,'token_identical_to_baseline':tokens==C['output_token_ids'],'first_changed_output_position':next((i for i in range(max(len(tokens),len(C['output_token_ids']))) if i>=len(tokens) or i>=len(C['output_token_ids']) or tokens[i]!=C['output_token_ids'][i]),None),'first_output_top5':first_top,'summary':summary,'traces':traces}
 for condition in P['conditions']:
  row=generate(condition);result['rows'].append(row);save(result);print(condition,json.dumps({'answer':row['answer'],'contains_secret':row['contains_exact_secret'],'baseline_identical':row['token_identical_to_baseline'],'summary':row['summary']},ensure_ascii=False),flush=True)
 result.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START,attention_unmodified=True);save(result);print('COMPLETE',flush=True)
if __name__=='__main__':main()
