"""Native screening and additive L19-T368 intervention, pinned to previous experiment."""
import argparse,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
START=time.monotonic()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,data):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--weights',type=Path);a=ap.parse_args()
 job=json.loads(a.job.read_text());out=ROOT/'artifacts';out.mkdir(exist_ok=True);dest=out/(job['name']+'.json');assert not dest.exists(),'Refusing to overwrite results'
 verify=json.loads((ROOT/'expected-base-verification.json').read_text())
 assert verify['repo']==P['base_repo'] and verify['revision']==P['base_revision']
 for rec in verify['files']:
  p=a.model_dir/rec['file'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],str(p)
 torch.manual_seed(0);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
 assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);attention=[b.self_attn.forward for b in model.model.layers]
 cases={c['id']:c for c in json.loads((ROOT/job['cases_file']).read_text())}
 weights=None;layer=P['layer'];feature=P['transform']
 if any(t['strength'] is not None for t in job['trials']):
  assert a.weights and sha(a.weights)==P['asset_sha256'];weights={k:v.cuda() for k,v in load_file(a.weights).items()};theta=weights['nonlinearity.theta'].reshape(-1)[feature]
  assert weights['e.weight'].shape==(2480,2560) and weights[f'V_{feature}'].shape==(2560,128)
 result={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':P,'job':job,'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json',job['cases_file']]},'job_sha256':sha(a.job),'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},'base_files_hash_verified':True,'rows':[]}
 @torch.inference_mode()
 def generate(c,added,trial_id):
  prompt=P['prompt_template'].format(context=c['context'],question=c['question']);chat=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  encoded=tok(chat,add_special_tokens=False)['input_ids']
  if 'prompt_token_ids' in c:assert encoded==c['prompt_token_ids'] and chat==c['chat']
  ids=torch.tensor([encoded],device='cuda');mask=torch.ones_like(ids);state={};traces=[];step=0;handles=[]
  def input_hook(_m,inputs):
   z=(inputs[0][:,-1:,:].detach().float()-weights['input_standardizer.mean'][layer])/weights['input_standardizer.std'][layer]
   raw=F.linear(z,weights['e.weight'][feature:feature+1],weights['e.bias'][feature:feature+1]);full=F.linear(z,weights['e.weight'],weights['e.bias'])
   assert torch.allclose(raw,full[...,feature:feature+1],atol=2e-5,rtol=2e-5)
   gate=raw*((raw>theta)&(raw>0));lowrank=(z@weights[f'V_{feature}']@weights[f'U_{feature}'])*weights['output_standardizer.std'][layer]
   state.update(raw=float(raw.item()),gate=float(gate.item()),lowrank=lowrank)
  def output_hook(_m,_in,output):
   eligible=step>0;delta_gate=added if eligible else 0.;delta=delta_gate*state['lowrank'];changed=None
   if eligible:
    assert output.shape[1]==1;changed=(output.float()+delta).to(output.dtype);assert torch.isfinite(changed).all()
    if added==0:assert torch.equal(changed,output)
   actual=0. if changed is None else float((changed.float()-output.float()).norm());native_norm=float(output[:,-1:,:].float().norm());low_norm=float(state['lowrank'].norm())
   traces.append({'segment':'response' if eligible else 'prompt_final','answer_position':step-1 if eligible else None,'full_chat_position':len(encoded)+step-1,'token_id':int(ids[0,-1]),'token':tok.decode([int(ids[0,-1])]),'eligible':eligible,'natural_raw':state['raw'],'natural_gate':state['gate'],'added_gate':added if eligible else None,'gate_delta':delta_gate,'effective_estimated_gate':state['gate']+delta_gate,'lowrank_norm':low_norm,'intended_delta_norm':float(delta.norm()),'actual_bf16_delta_norm':actual,'native_mlp_norm':native_norm,'intended_delta_over_native_mlp_norm':float(delta.norm())/native_norm})
   return changed
  if added is not None:
   b=model.model.layers[layer];handles=[b.post_attention_layernorm.register_forward_pre_hook(input_hook),b.mlp.register_forward_hook(output_hook)]
  tokens=[];cache=None;stop='token_limit'
  try:
   for step in range(P['decoding']['max_new_tokens']):
    assert time.monotonic()-START<60*P['max_runtime_minutes']
    pred=model(input_ids=ids,attention_mask=mask,past_key_values=cache,use_cache=True);cache=pred.past_key_values;token=int(pred.logits[0,-1].argmax());tokens.append(token)
    if step==0 and 'output_token_ids' in c:assert token==c['output_token_ids'][0],'First output token changed'
    if token in eos:stop='eos';break
    ids=torch.tensor([[token]],device='cuda');mask=torch.cat([mask,torch.ones_like(ids)],dim=1)
   assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,attention))
  finally:
   for h in handles:h.remove()
  if (added is None or added==0) and 'output_token_ids' in c:assert tokens==c['output_token_ids'],c['id']+' native/zero tokens changed'
  response=[t for t in traces if t['eligible']]
  if added is not None and added>0:assert len(response)==len(tokens)-1 and all(t['actual_bf16_delta_norm']>0 and t['gate_delta']==added for t in response)
  return {**c,'trial_id':trial_id,'case_id':c['id'],'prompt':prompt,'chat':chat,'prompt_token_ids':encoded,'condition':'native' if added is None else 'added_gate','added_gate':added,'answer':tok.decode(tokens,skip_special_tokens=True),'output_token_ids':tokens,'stop_reason':stop,'native_or_zero_matches_baseline':tokens==c['output_token_ids'] if 'output_token_ids' in c and (added is None or added==0) else None,'native_screen_no_hooks':added is None,'first_output_token_unchanged':tokens[0]==c['output_token_ids'][0] if 'output_token_ids' in c else None,'traces':traces,'summary':{'edited_positions':sum(t['actual_bf16_delta_norm']>0 for t in response),'processed_response_tokens':len(response),'max_delta_over_native_mlp_norm':max((t['intended_delta_over_native_mlp_norm'] for t in response),default=0.)}}
 for t in job['trials']:
  row=generate(cases[t['case_id']],t['strength'],t['trial_id']);result['rows'].append(row);save(dest,result);print(row['trial_id'],'STOP='+row['stop_reason'],'TEXT='+json.dumps(row['answer'],ensure_ascii=False),flush=True)
 if job.get('repeat_first'):
  t=job['trials'][0];repeat=generate(cases[t['case_id']],t['strength'],'repeat');assert repeat['output_token_ids']==result['rows'][0]['output_token_ids'];result['repeat_first_token_identical']=True
 result.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START);save(dest,result);print('COMPLETE',job['name'],len(result['rows']),flush=True)
if __name__=='__main__':main()
