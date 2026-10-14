"""Capture clean native states, then run nonzero source-contribution controls."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'protocol.json').read_text())
START=time.monotonic()
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,data):
 temp=OUT/(name+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(OUT/name)

class Native:
 def __init__(self,model_dir):
  torch.manual_seed(0);torch.set_num_threads(8)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
  self.attention=[b.self_attn.forward for b in self.model.model.layers]
  self.ids={}
  for source,word in [('receiver','small'),('donor','hot')]:
   text=self.tokenizer.apply_chat_template([{'role':'user','content':P[source]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   batch=self.tokenizer(text,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True)
   start=text.index(word,text.index(P[source]));end=start+len(word)
   positions=[i for i,(a,b) in enumerate(batch.pop('offset_mapping')[0].tolist()) if b>a and a<end and b>start]
   assert positions==[9] and batch['input_ids'].shape==(1,29)
   self.ids[source]=batch['input_ids'].cuda()
  assert self.tokenizer.encode('cold',add_special_tokens=False)==[87072]
 @torch.inference_mode()
 def forward(self,source='receiver',specs=None,capture=False):
  size=8;specs=specs or [];states={};before={};after={};handles=[]
  for layer,block in enumerate(self.model.model.layers):
   def pre(_m,args,layer=layer):states[layer]=args[0][:,9].detach().clone()
   def post(_m,args,output,layer=layer):
    before[layer]=output[:,9].detach().clone()
    changed=output.clone()
    for row,spec in enumerate(specs):
     if spec.get('edit_layer',spec['layer'])==layer:changed[row,9]=output[row,9]+spec['vector'].to(output.dtype)
    after[layer]=changed[:,9].detach().clone()
    return changed if specs else None
   handles.extend([block.post_attention_layernorm.register_forward_pre_hook(pre),block.mlp.register_forward_hook(post)])
  try:logits=self.model(input_ids=self.ids[source].repeat(size,1),use_cache=False).logits[:,-1]
  finally:
   for h in handles:h.remove()
  assert logits.dtype==torch.bfloat16
  assert all(b.self_attn.forward==f for b,f in zip(self.model.model.layers,self.attention))
  return logits,states,before,after

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['capture','run','fixed_site']);ap.add_argument('--model-dir',type=Path,required=True)
 args=ap.parse_args();e=Native(args.model_dir)
 receiver,rec,rec_mlp,_=e.forward();donor,don,don_mlp,_=e.forward(source='donor')
 for logits in [receiver,donor]:assert all(torch.equal(logits[0],r) for r in logits)
 for states in [rec,don,rec_mlp,don_mlp]:
  for rows in states.values():assert all(torch.equal(rows[0],row) for row in rows)
 assert e.tokenizer.decode([int(receiver[0].argmax())])=='large'
 assert e.tokenizer.decode([int(donor[0].argmax())])=='cold'
 capture=dict(receiver_prenorm=np.stack([rec[i][0].float().cpu().numpy() for i in range(36)]),
  donor_prenorm=np.stack([don[i][0].float().cpu().numpy() for i in range(36)]),
  receiver_mlp=np.stack([rec_mlp[i][0].float().cpu().numpy() for i in range(36)]),
  donor_mlp=np.stack([don_mlp[i][0].float().cpu().numpy() for i in range(36)]),
  receiver_logits_bits=receiver[0].view(torch.uint16).cpu().numpy(),donor_logits_bits=donor[0].view(torch.uint16).cpu().numpy())
 if args.mode=='capture':
  np.savez_compressed(OUT/'clean-state.npz',**capture)
  save('capture.json',{'status':'complete','protocol':P,'worker_sha256':sha(ROOT/'worker.py'),
   'clean_state_sha256':sha(OUT/'clean-state.npz'),'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()}})
  print('CLEAN_CAPTURE_COMPLETE',flush=True);return
 old=np.load(OUT/'clean-state.npz')
 for k,v in capture.items():assert np.array_equal(old[k],v),k+' changed'
 selection=json.loads((ROOT/'selection.json').read_text());parameters={};vectors={};gates={}
 for layer in P['layers']:
  weights=np.load(ROOT/'weights'/f'layer-{layer}-selected.npz')
  ids=weights['ids'].tolist();enc=torch.from_numpy(weights['encoder']).cuda();bias=torch.from_numpy(weights['bias']).cuda();theta=torch.from_numpy(weights['threshold']).cuda()
  mean=torch.from_numpy(weights['mean']).cuda();std=torch.from_numpy(weights['std']).cuda();outstd=torch.from_numpy(weights['outstd']).cuda()
  for source,states in [('receiver',rec),('donor',don)]:
   z=(states[layer][0].float()-mean)/std;raw=F.linear(z,enc,bias);activation=raw*((raw>theta)&(raw>0))
   for j,f in enumerate(ids):
    gate=float(activation[j]);gates[source,layer,f]=gate
    expected=selection['features'][f'{layer}:{f}'][source+'_gate']
    assert abs(gate-expected)<5e-5,(layer,f,source,gate,expected)
    low=z@torch.from_numpy(weights[f'V_{f}']).cuda()@torch.from_numpy(weights[f'U_{f}']).cuda()
    vector=activation[j]*low*outstd;vectors[source,layer,f]=vector
    selection['features'][f'{layer}:{f}'][source+'_gate_gpu']=gate
    selection['features'][f'{layer}:{f}'][source+'_contribution_norm']=float(vector.norm())
 save('gpu-validated-selection.json',selection)
 condition_file='conditions-fixed-site.json' if args.mode=='fixed_site' else 'conditions.json'
 raw_specs=json.loads((ROOT/condition_file).read_text());specs=[]
 for s in raw_specs:
  vector=sum((coefficient*vectors[source,s['layer'],f] for source,f,coefficient in s['terms']),torch.zeros(2560,device='cuda'))
  if s.get('norm_match_to'):
   target={'pair':sum((-4.5*vectors['receiver',22,2339],4.5*vectors['donor',22,941])),
           'positive':4.5*vectors['donor',22,941],'negative':-4.5*vectors['receiver',22,2339]}[s['norm_match_to']]
   scale=float(target.norm()/vector.norm());vector*=scale;s['norm_match_scale']=scale
  assert float(vector.norm())>0
  for source,f,_ in s['terms']:assert gates[source,s['layer'],f]>0
  specs.append({**s,'vector':vector})
 R={'status':'running','protocol':P,'selection':selection,'rows':[],
    'source_sha256':{n:sha(ROOT/n) for n in ['worker.py','protocol.json','selection.json',condition_file]},
    'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()},'clean_batch_checks':[]}
 intended=[];actual=[];logit_buffer=[];chunks=[]
 def append(spec,logits,before,after):
  prob=logits.float().softmax(-1);values,indices=prob.topk(5)
  row={k:v for k,v in spec.items() if k!='vector'}
  row.update(P_cold=float(prob[87072]),P_large=float(prob[16767]),top_token=e.tokenizer.decode([int(indices[0])]),
     top5=[{'id':int(i),'token':e.tokenizer.decode([int(i)]),'p':float(v)} for i,v in zip(indices,values)],
     intended_delta_norm=float(spec['vector'].norm()),actual_delta_norm=float((after.float()-before.float()).norm()),
     native_mlp_norm=float(before.float().norm()),logit_row=len(R['rows']))
  row['source_gates']=[{'source':src,'layer':spec['layer'],'feature':f,'gate':gates[src,spec['layer'],f]} for src,f,c in spec['terms']]
  row['mean_source_gate']=sum(x['gate'] for x in row['source_gates'])/len(row['source_gates'])
  row['nonzero_actual_edit']=row['actual_delta_norm']>0;assert row['nonzero_actual_edit']
  bits=logits.contiguous().view(torch.uint16).cpu().numpy().copy();row['logit_sha256']=hashlib.sha256(bits.tobytes()).hexdigest()
  R['rows'].append(row);logit_buffer.append(bits)
  intended.append(spec['vector'].float().cpu().numpy());actual.append((after.float()-before.float()).cpu().numpy())
  if len(logit_buffer)==32:flush()
 def flush():
  if not logit_buffer:return
  filename=f'logits-{len(chunks):03d}.npz';np.savez_compressed(OUT/filename,bits=np.stack(logit_buffer))
  chunks.append({'file':filename,'rows':len(logit_buffer),'sha256':sha(OUT/filename)});logit_buffer.clear()
 for offset in range(0,len(specs),7):
  assert time.monotonic()-START<P['runtime_limit_minutes']*60
  batch=specs[offset:offset+7];logits,states,pre,post=e.forward(specs=batch)
  for i,s in enumerate(batch):
   layer=s.get('edit_layer',s['layer']);assert torch.equal(states[layer][i],rec[layer][0]);assert torch.equal(pre[layer][i],rec_mlp[layer][0])
   append(s,logits[i],pre[layer][i],post[layer][i])
  for i in range(len(batch),8):assert torch.equal(logits[i],receiver[0])
  R['clean_batch_checks'].append(True);save('results.json',R)
  print('BATCH',offset,'/',len(specs),flush=True)
 flush();R['logit_chunks']=chunks
 np.savez_compressed(OUT/'edits.npz',intended=np.stack(intended),actual=np.stack(actual))
 baseline_prob=receiver[0].float().softmax(-1);R['baseline']={'P_cold':float(baseline_prob[87072]),'P_large':float(baseline_prob[16767])}
 selected=next((r for r in R['rows'] if r['id']=='selected_pair_4.5'),None)
 if selected:assert abs(selected['P_cold']-.998653)<.0001,selected
 assert len(R['rows'])==len(raw_specs)
 R['status']='complete';R['elapsed_seconds']=time.monotonic()-START;save('results.json',R)
 print('COMPLETE',len(R['rows']),'selected P(cold)',selected['P_cold'] if selected else 'reference verified in native-site arm',flush=True)
if __name__=='__main__':main()
