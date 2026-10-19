"""Fetch U/V only for the preregistered active feature-site pairs."""
import concurrent.futures,json,struct
from pathlib import Path
import numpy as np
from prepare import ROOT,P,GROUPS,array,sha

OUT=ROOT/'weights';OUT.mkdir(exist_ok=True)

def one(layer,features):
    path=OUT/f'layer-{layer}.npz';meta=OUT/f'layer-{layer}.json'
    if path.exists() and meta.exists():return json.loads(meta.read_text())
    url=f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-{layer:02d}/checkpoint.safetensors"
    headtext=(ROOT/f'gates/layer-{layer}-header.json').read_text();head=json.loads(headtext)
    # Header byte length cannot be inferred from pretty-printed JSON.
    from prepare import fetch
    first=fetch(url,0,65535);n=struct.unpack('<Q',first[:8])[0];assert json.loads(first[8:8+n])==head
    with np.load(ROOT/f'gates/layer-{layer}-gates.npz') as a:
        data={k:a[k] for k in ['mean','std','outstd']}
        data.update(ids=np.array(features),encoder=a['encoder'][features],bias=a['bias'][features],threshold=a['threshold'].reshape(-1)[features])
    data['outmean'],mean_record=array(url,8+n,head['output_standardizer.mean'],'output_standardizer.mean',layer=layer)
    def feature_job(f):
        g=next(i for i,(lo,hi,_r) in enumerate(GROUPS) if lo<=f<hi);i=f-GROUPS[g][0]
        item={'transform':f,'rank':GROUPS[g][2],'records':[]};tensors={}
        for role in ['U','V']:
            tensors[f'{role}_{f}'],record=array(url,8+n,head[f'{role}s.{g}'],f'{role}s.{g}',feature=i)
            item['records'].append(record)
        return item,tensors
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for item,tensors in pool.map(feature_job,features):
            records.append(item);data.update(tensors)
    np.savez_compressed(path,**data)
    result={'layer':layer,'file':path.name,'sha256':sha(path),'features':records,'url':url,'outmean_record':mean_record}
    meta.write_text(json.dumps(result,indent=2)+'\n');print('UV_READY',layer,len(features),path.stat().st_size,flush=True)
    return result

def main():
    c=json.loads((ROOT/'survey/candidates.json').read_text());assert c['selection_before_ablation_outcomes']
    by={}
    for r in c['rows']:by.setdefault(r['layer'],set()).add(r['transform'])
    by.setdefault(3,set()).add(2334)  # Prior-verified inactive CH control at source child.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(lambda item:one(item[0],sorted(item[1])),by.items()))
    (OUT/'manifest.json').write_text(json.dumps({'repo':P['molt_repo'],'revision':P['molt_revision'],'layers':rows,
        'candidate_list_sha256':sha(ROOT/'survey/candidates.json')},indent=2)+'\n')
    print('ALL_CANDIDATE_WEIGHTS_READY',sum(len(r['features']) for r in rows),flush=True)

if __name__=='__main__':main()
