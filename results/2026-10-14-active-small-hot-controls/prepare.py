"""Fetch pinned tensor slices; select gate-active controls before their outcomes."""
import concurrent.futures,hashlib,json,random,struct,time
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np

ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/'weights';WEIGHTS.mkdir(exist_ok=True)
CACHE=ROOT/'cache';CACHE.mkdir(exist_ok=True)
P=json.loads((ROOT/'protocol.json').read_text())
STATE=np.load(ROOT/'clean-state.npz')
GROUPS=[(0,80,512),(80,240,256),(240,560,128),(560,1200,64),(1200,2480,32)]
def fetch(url,start,end):
 for attempt in range(4):
  try:
   request=Request(url+f'?active_control_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'})
   with urlopen(request,timeout=60) as r:
    assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    blob=r.read(end-start+2);assert len(blob)==end-start+1;return blob
  except Exception:
   if attempt==3:raise
   time.sleep(attempt+1)
def array(url,base,spec,key,layer=None,feature=None):
 start,end=[base+n for n in spec['data_offsets']];shape=spec['shape'];dtype=spec['dtype']
 itemsize={'F32':4,'BF16':2,'F16':2}[dtype]
 if layer is not None:
  assert shape==[36,2560];start+=layer*2560*itemsize;end=start+2560*itemsize;shape=[2560]
 if feature is not None:
  size=(end-start)//shape[0];start+=feature*size;end=start+size;shape=shape[1:]
 ranges=[(a,min(a+4*1024*1024,end)-1) for a in range(start,end,4*1024*1024)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  blobs=list(pool.map(lambda pair:fetch(url,*pair),ranges))
 blob=b''.join(blobs)
 if dtype=='BF16':result=(np.frombuffer(blob,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
 else:result=np.frombuffer(blob,dtype='<f4' if dtype=='F32' else '<f2').astype(np.float32)
 result=result.reshape(shape).copy()
 return result,{'key':key,'range':[start,end-1],'shape':shape,'source_dtype':dtype,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
def layer_job(layer):
 prefix=f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-{layer:02d}"
 url=prefix+'/checkpoint.safetensors'
 first=fetch(url,0,65535);n=struct.unpack('<Q',first[:8])[0];assert n<65528
 header=json.loads(first[8:8+n]);(CACHE/f'layer-{layer}-header.json').write_text(json.dumps(header,indent=2)+'\n')
 path=CACHE/f'layer-{layer}-gates.npz';records=[]
 if path.exists():
  data=dict(np.load(path));records=json.loads((CACHE/f'layer-{layer}-gate-records.json').read_text())
 else:
  data={}
  for short,key in [('encoder','e.weight'),('bias','e.bias'),('threshold','nonlinearity.theta'),('mean','input_standardizer.mean'),('std','input_standardizer.std'),('outstd','output_standardizer.std')]:
   data[short],record=array(url,8+n,header[key],key,layer=layer if short in ['mean','std','outstd'] else None);records.append(record)
  np.savez_compressed(path,**data);(CACHE/f'layer-{layer}-gate-records.json').write_text(json.dumps(records,indent=2)+'\n')
 for source in ['receiver','donor']:
  z=(STATE[source+'_prenorm'][layer]-data['mean'])/data['std']
  raw=data['encoder']@z+data['bias'];gate=raw*((raw>data['threshold'].reshape(-1))&(raw>0));data[source+'_gates']=gate
 print('GATES',layer,'receiver',int((data['receiver_gates']>0).sum()),'donor',int((data['donor_gates']>0).sum()),flush=True)
 return layer,data,header,8+n,url,records

def main():
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:layers={r[0]:r[1:] for r in pool.map(layer_job,P['layers'])}
 targets={2339,941};rng=random.Random(P['seed'])
 def pool(source,chosen):return [(l,int(f)) for l in chosen for f in np.flatnonzero(layers[l][0][source+'_gates']>0) if not(l==22 and int(f) in targets)]
 same_positive=pool('donor',[22]);same_negative=pool('receiver',[22])
 assert len(same_positive)==len(same_negative)==2
 def pairs(chosen):return [(l,a,b) for l,a in pool('receiver',chosen) for l2,b in pool('donor',chosen) if l==l2 and a!=b]
 same_pairs=pairs([22]);assert len(same_pairs)==3
 other_positive=rng.sample(pool('donor',P['other_layers']),10)
 other_negative=rng.sample(pool('receiver',P['other_layers']),10)
 other_pairs=rng.sample(pairs(P['other_layers']),10)
 selection={'seed':P['seed'],'saved_before_intervention_outcomes':True,'same_positive':same_positive,'same_negative':same_negative,'same_pairs':same_pairs,
  'other_positive':other_positive,'other_negative':other_negative,'other_pairs':other_pairs,'features':{},'active_pool_counts':{},'gate_provenance':{}}
 needed={l:set() for l in P['layers']};needed[22].update(targets)
 for l,f in same_positive+same_negative+other_positive+other_negative:needed[l].add(f)
 for l,a,b in same_pairs+other_pairs:needed[l].update([a,b])
 source=Path('comparisons/2026-10-12-spider-ant/summaries')
 for l in P['layers']:
  data,header,base,url,records=layers[l]
  selection['active_pool_counts'][str(l)]={'receiver':len(pool('receiver',[l])),'donor':len(pool('donor',[l])),'distinct_pairs':len(pairs([l]))}
  selection['gate_provenance'][str(l)]={'url':url,'tensors':records}
  labels=json.loads((source/f'layer-{l}.json').read_text()) if (source/f'layer-{l}.json').exists() else None
  tensor_data={'ids':np.array(sorted(needed[l])),'encoder':data['encoder'][sorted(needed[l])],
    'bias':data['bias'][sorted(needed[l])],'threshold':data['threshold'].reshape(-1)[sorted(needed[l])],
    'mean':data['mean'],'std':data['std'],'outstd':data['outstd']}
  for f in sorted(needed[l]):
   item=labels['features'].get(str(f),{}) if labels else {}
   title=item.get('title') or item.get('label') or 'No cached dashboard label'
   if isinstance(title,dict):title=str(title)
   selection['features'][f'{l}:{f}']={'layer':l,'id':f,'label':title,'receiver_gate':float(data['receiver_gates'][f]),'donor_gate':float(data['donor_gates'][f]),'mean_gate_across_clean_sources':float((data['receiver_gates'][f]+data['donor_gates'][f])/2),'label_source':'Dashboard summary cached 2026-10-12, step100000' if labels else 'unavailable','tensor_records':[]}
   group=next(j for j,(lo,hi,rank) in enumerate(GROUPS) if lo<=f<hi);offset=f-GROUPS[group][0]
   for role in ['U','V']:
    tensor,record=array(url,base,header[f'{role}s.{group}'],f'{role}s.{group}',feature=offset)
    tensor_data[f'{role}_{f}']=tensor;selection['features'][f'{l}:{f}']['tensor_records'].append(record)
  np.savez_compressed(WEIGHTS/f'layer-{l}-selected.npz',**tensor_data)
  print('UV_READY',l,len(needed[l]),flush=True)
 conditions=[]
 def add(kind,stratum,l,a,b,strength,index,norm=None):
  if kind=='positive':terms=[['donor',b,strength]]
  elif kind=='negative':terms=[['receiver',a,-strength]]
  elif kind=='pair':terms=[['receiver',a,-strength],['donor',b,strength]]
  else:terms=[['receiver',a,strength],['donor',b,-strength]]
  conditions.append({'id':f'{stratum}_{kind}_{index}','kind':kind,'stratum':stratum,'layer':l,'strength':strength,'sample_index':index,'terms':terms,**({'norm_match_to':norm} if norm else {})})
 for kind,features in [('positive',same_positive),('negative',same_negative)]:
  for i,(l,f,s) in enumerate((l,f,s) for l,f in features for s in P['primary_strengths']):add(kind,'same_layer',l,f,f,s,i)
 for kind,features in [('positive',other_positive),('negative',other_negative)]:
  for i,(l,f,s) in enumerate((l,f,s) for l,f in features for s in P['cross_strengths']):add(kind,'other_layers',l,f,f,s,i)
 same_pair_conditions=rng.sample([(l,a,b,s) for l,a,b in same_pairs for s in P['primary_strengths']],50)
 other_pair_conditions=[(l,a,b,s) for l,a,b in other_pairs for s in P['cross_strengths']]
 for stratum,items in [('same_layer',same_pair_conditions),('other_layers',other_pair_conditions)]:
  for i,(l,a,b,s) in enumerate(items):
   for kind in ['pair','pair_reverse']:add(kind,stratum,l,a,b,s,i)
 for strength in P['primary_strengths']:
  for kind in ['positive','negative','pair','pair_reverse']:
   add(kind,'selected',22,2339,941,strength,len(conditions));conditions[-1]['id']=f'selected_{kind}_{strength}'
 for kind,features in [('positive',same_positive+other_positive),('negative',same_negative+other_negative)]:
  for i,(l,f) in enumerate(features):add(kind,'norm_matched',l,f,f,1,i,norm=kind)
 for i,(l,a,b) in enumerate(same_pairs+other_pairs):
  for kind in ['pair','pair_reverse']:add(kind,'norm_matched',l,a,b,1,i,norm='pair')
 assert len(conditions)==550
 (ROOT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 (ROOT/'conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
 fixed_site=[]
 for condition in conditions:
  if condition['layer']!=22 and condition['stratum'] in ['other_layers','norm_matched']:
   fixed_site.append({**condition,'id':'fixed_site_'+condition['id'],
    'stratum':'other_layers_fixed_site' if condition['stratum']=='other_layers' else 'norm_matched_other_fixed_site',
    'edit_layer':22})
 assert len(fixed_site)==240
 (ROOT/'conditions-fixed-site.json').write_text(json.dumps(fixed_site,indent=2)+'\n')
 print('PREPARATION_COMPLETE',len(selection['features']),'unique transforms;',len(conditions),'nonzero conditions',flush=True)
if __name__=='__main__':main()
