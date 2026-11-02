"""Freeze all naturally active final-position controls; fetch only pinned operator slices."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
import requests
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
W=HERE/'weights';W.mkdir(exist_ok=True)
G=ROOT/'experiments/ch-alllayer-causal-screen/gates'
ANCHOR=ROOT/'comparisons/2026-11-03-eleven-to-five-scalar/remote-export/eleven-to-five-scalar-20261103/artifacts/donor-observed-arrays.npz'
REV='081ce901b090b64aeecf9bc8709b57c10411e32a'
URL=f'https://huggingface.co/georglange/qwen3-4b-molt/resolve/{REV}/layer-29/checkpoint.safetensors'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(G/'layer-29-gates.npz',allow_pickle=False) as a:gate_data={k:a[k].copy() for k in a.files}
for name,key in [('encoder','e.weight'),('bias','e.bias'),('threshold','nonlinearity.theta'),('mean','input_standardizer.mean'),('std','input_standardizer.std'),('outstd','output_standardizer.std')]:
    record=next(r for r in json.loads((G/'layer-29-records.json').read_text()) if r['key']==key)
    assert hashlib.sha256(gate_data[name].tobytes()).hexdigest()==record['sha256']
with np.load(ANCHOR,allow_pickle=False) as a:
    residual=a['residual'].copy();z=(residual-gate_data['mean'])/gate_data['std'];assert np.array_equal(z,a['z'])
raw=z@gate_data['encoder'].T+gate_data['bias'];theta=gate_data['threshold'].reshape(-1)
gates=np.where((raw>theta)&(raw>0),raw,0.)
active=np.flatnonzero(gates[-1]).tolist()
assert active==[34,587,799,858,1181,2375]
np.savez_compressed(W/'all-gates.npz',**gate_data)
selection={'prompt':'the eleventh month of the year is','position':7,'token':' is','transform_count':2480,
           'selection_rule':'All transforms with a positive natural thresholded gate at final input position; no ablation outcomes used.',
           'selected_transforms':active,'active_by_input_position':[np.flatnonzero(row).tolist() for row in gates],
           'all_final_gate_values':gates[-1].tolist(),'all_final_raw_values':raw[-1].tolist(),
           'anchor_arrays_sha256':sha(ANCHOR),'gate_file_sha256':sha(G/'layer-29-gates.npz'),
           'threshold_boundary_note':'Scan uses FP32 batched CPU arithmetic. Runtime recomputes individual gates; assert identical active set and close values.'}
(HERE/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
def fetch(start,end):
    for attempt in range(3):
        try:
            response=requests.get(URL+f'?active_control_range={start}-{end}&n={time.time_ns()}',headers={'Range':f'bytes={start}-{end}'},timeout=35)
            response.raise_for_status();assert response.status_code==206
            assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/') and len(response.content)==end-start+1
            return response.content
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)
first=fetch(0,65535);size=struct.unpack('<Q',first[:8])[0];assert size<len(first)-8
header=json.loads(first[8:8+size]);assert header==json.loads((G/'layer-29-header.json').read_text())
(W/'layer-29-header.json').write_text(json.dumps(header,indent=2)+'\n')
groups=[(0,80,512),(80,240,256),(240,560,128),(560,1200,64),(1200,2480,32)]
def one(index):
    group,(lo,hi,rank)=next((j,g) for j,g in enumerate(groups) if g[0]<=index<g[1])
    arrays={'e':gate_data['encoder'][index], 'bias':gate_data['bias'][index:index+1],
            'theta':theta[index:index+1],'input_mean':gate_data['mean'],'input_std':gate_data['std'],'output_std':gate_data['outstd']}
    records=[]
    for role,shape in [('U',(rank,2560)),('V',(2560,rank))]:
        key=role+'s.'+str(group);spec=header[key];assert spec['dtype']=='F32' and spec['shape']==[hi-lo,*shape]
        count=rank*2560;start=8+size+spec['data_offsets'][0]+(index-lo)*count*4;end=start+count*4-1
        blob=fetch(start,end);arrays[role]=np.frombuffer(blob,dtype='<f4').copy().reshape(shape)
        assert np.isfinite(arrays[role]).all()
        records.append({'source_tensor':key,'byte_range_inclusive':[start,end],'slice_sha256':hashlib.sha256(blob).hexdigest()})
    path=W/f'L29-T{index}.npz';np.savez_compressed(path,**arrays)
    if index==2375:
        with np.load(ROOT/'experiments/eleven-activation-20261102/weights/L29-T2375.npz',allow_pickle=False) as old:
            assert all(np.array_equal(arrays[k],old[k]) for k in arrays)
    return {'layer':29,'transform':index,'rank':rank,'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,'records':records}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,active))
manifest={'repo':'georglange/qwen3-4b-molt','revision':REV,'checkpoint_url':URL,'header_sha256':sha(W/'layer-29-header.json'),
          'all_gate_file_sha256':sha(W/'all-gates.npz'),'source_gate_records_sha256':sha(G/'layer-29-records.json'),
          'features':rows,'whole_checkpoint_not_downloaded':True,'T2375_all_arrays_identical_to_prior':True}
(W/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
protocol={'date_local':'2026-11-03','feature_layer':29,'prompt':selection['prompt'],'position':7,'token':' is',
          'selected_transforms':active,'selection_rule':selection['selection_rule'],'max_new_tokens':32,
          'scope':'One selected MOLT off at final input token only; no generated-token edits; each run independent.',
          'edit':'native MLP - natural selected gate * selected current operator; FP32 followed by one BF16 cast.',
          'reference_transform':2375,'natural_strength_only':True,'operator_normalized':False,
          'limits':['One raw lowercase month prefix.','Natural ablations differ in contribution norms; report sizes rather than claim norm-matched controls.','All six active transforms tested individually; no joint ablation.']}
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'selected_transforms':active,'features_downloaded':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
