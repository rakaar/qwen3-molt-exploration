"""Native controls and joint zero gate replacement on the unauthorized Bob prompt."""
import argparse,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'protocol.json').read_text());C=json.loads((ROOT/'case.json').read_text());START=time.monotonic();OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(data):
 tmp=OUT/'results.tmp';tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--weights-dir',type=Path,required=True);a=ap.parse_args()
 assert not (OUT/'results.json').exists(),'Refusing to overwrite earlier results'
 base=json.loads((ROOT/'expected-base-verification.json').read_text());assert base['repo']==P['base_repo'] and base['revision']==P['base_revision']
 for f in base['files']:
  path=a.model_dir/f['file'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256'],f['file']
 torch.manual_seed(0);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
 assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
 manifest=json.loads((ROOT/'weights-manifest.json').read_text());weights={}
 assert manifest['revision']==P['molt_revision']
 for f in manifest['features']:
  path=a.weights_dir/f['file'];assert sha(path)==f['sha256']
  with np.load(path) as data:weights[f['layer']]={k:torch.from_numpy(data[k].copy()).float().cuda() for k in data.files}
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);attention=[b.self_attn.forward for b in model.model.layers]
 chat=tok.apply_chat_template([{'role':'system','content':C['system']},{'role':'user','content':C['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 encoded=tok(chat,add_special_tokens=False)['input_ids'];assert encoded==C['prompt_token_ids'] and chat==C['chat']
 result={'status':'running','protocol':P,'case':C,'assets':manifest,'base_files_hash_verified':True,'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda},'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','case.json','weights-manifest.json']},'started_at_utc':datetime.now(timezone.utc).isoformat(),'rows':[]}
 @torch.inference_mode()
 def generate(condition):
  ids=torch.tensor([encoded],device='cuda');mask=torch.ones_like(ids);cache=None;tokens=[];traces=[];state={};handles=[];step=0;stop='token_limit'
  if condition!='native':
   for feature in P['features']:
    layer=feature['layer'];name=f"L{layer}-T{feature['transform']}";w=weights[layer]
    def capture(_m,inputs,name=name,w=w):
     z=(inputs[0][:,-1:,:].detach().float()-w['input_mean'])/w['input_std'];raw=F.linear(z,w['e'].reshape(1,-1),w['bias'])
     check=(z*w['e']).sum(-1)+w['bias'][0];assert torch.allclose(raw.squeeze(-1),check,atol=3e-5,rtol=3e-5)
     gate=torch.where((raw>w['theta'][0])&(raw>0),raw,torch.zeros_like(raw));low=(z@w['V']@w['U'])*w['output_std']
     state[name]={'raw':float(raw.item()),'gate':float(gate.item()),'low':low,'contribution':gate*low,'threshold':float(w['theta'][0])}
    def edit(_m,_inputs,output,name=name,layer=layer):
     x=state[name];eligible=step>0;clamp=condition=='joint_zero_clamp' and eligible
     delta=-x['contribution'] if clamp else torch.zeros_like(x['contribution']);changed=None
     if clamp:
      assert output.shape[1]==1;changed=(output.float()+delta).to(output.dtype);assert torch.isfinite(changed).all()
      if x['gate']==0:assert torch.equal(output,changed)
     actual=0. if changed is None else float((changed.float()-output.float()).norm())
     traces.append({'feature':name,'layer':layer,'segment':'response' if eligible else 'prompt_final','answer_position':step-1 if eligible else None,'full_chat_position':len(encoded)+step-1,'token_id':int(ids[0,-1]),'token':tok.decode([int(ids[0,-1])]),'special':int(ids[0,-1]) in tok.all_special_ids,'natural_raw':x['raw'],'natural_gate':x['gate'],'threshold':x['threshold'],'clamp_applied':clamp,'gate_delta':-x['gate'] if clamp else 0.,'effective_estimated_gate':0. if clamp else x['gate'],'lowrank_norm':float(x['low'].norm()),'natural_contribution_norm':float(x['contribution'].norm()),'intended_delta_norm':float(delta.norm()),'actual_bf16_delta_norm':actual,'native_mlp_norm':float(output[:,-1:,:].float().norm()),'edit_changes_bf16_output':actual>0})
     return changed
    block=model.model.layers[layer];handles.extend([block.post_attention_layernorm.register_forward_pre_hook(capture),block.mlp.register_forward_hook(edit)])
  try:
   for step in range(P['decoding']['max_new_tokens']):
    assert time.monotonic()-START<P['max_runtime_minutes']*60
    pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values;token=int(pred.logits[0,-1].argmax());tokens.append(token)
    if token in eos:stop='eos';break
    ids=torch.tensor([[token]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
  finally:
   for handle in handles:handle.remove()
  assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,attention))
  if condition in ['native','observe_only']:assert tokens==C['output_token_ids'],condition+' changed saved baseline'
  assert tokens[0]==C['output_token_ids'][0]
  if condition!='native':
   assert len(traces)==len(tokens)*3
   assert all(t['actual_bf16_delta_norm']==0 and not t['clamp_applied'] for t in traces if t['segment']=='prompt_final')
   if condition=='observe_only':assert all(t['actual_bf16_delta_norm']==0 and t['gate_delta']==0 for t in traces)
   else:assert all(t['clamp_applied'] and t['effective_estimated_gate']==0 and t['gate_delta']==-t['natural_gate'] for t in traces if t['segment']=='response')
  features={}
  for f in P['features']:
   name=f"L{f['layer']}-T{f['transform']}";ts=[t for t in traces if t['feature']==name and t['segment']=='response'];active=[t for t in ts if t['natural_gate']>0];edits=[t for t in ts if t['actual_bf16_delta_norm']>0]
   features[name]={'processed_response_positions':len(ts),'naturally_active_positions':len(active),'positions_with_actual_edit':len(edits),'first_active_position':active[0]['answer_position'] if active else None,'first_active_token':active[0]['token'] if active else None,'mean_natural_gate':sum(t['natural_gate'] for t in ts)/len(ts) if ts else None,'max_actual_delta_norm':max((t['actual_bf16_delta_norm'] for t in ts),default=0)}
  answer=tok.decode(tokens,skip_special_tokens=True)
  return {'condition':condition,'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),'output_token_ids':tokens,'stop_reason':stop,'contains_exact_secret':P['secret_marker'] in answer,'token_identical_to_baseline':tokens==C['output_token_ids'],'first_changed_output_position':next((i for i in range(max(len(tokens),len(C['output_token_ids']))) if i>=len(tokens) or i>=len(C['output_token_ids']) or tokens[i]!=C['output_token_ids'][i]),None),'first_output_token_unchanged':True,'features':features,'traces':traces}
 for condition in P['conditions']:
  row=generate(condition);result['rows'].append(row);save(result);print(condition,json.dumps({k:v for k,v in row.items() if k not in ['traces','output_token_ids']},ensure_ascii=False),flush=True)
 result.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START,attention_unmodified=True);save(result);print('COMPLETE',flush=True)
if __name__=='__main__':main()
