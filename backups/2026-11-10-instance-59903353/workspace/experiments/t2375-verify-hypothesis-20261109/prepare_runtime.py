"""Download verified native weights and only the exact MOLT gate tensors."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
os.environ['HF_HUB_DISABLE_XET']='1'
import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
expected=json.loads((ROOT/'expected-base-verification.json').read_text())
snapshot_download(repo_id=expected['repo'],revision=expected['revision'],
    allow_patterns=[r['file'] for r in expected['files']],local_dir=ROOT/'model',max_workers=4,token=False)
for row in expected['files']:
    p=ROOT/'model'/row['file']
    with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert p.stat().st_size==row['bytes'] and actual==row['sha256'],row['file']
print('BASE_WEIGHTS_VERIFIED',flush=True)
manifest=json.loads((ROOT/'weights/manifest.json').read_text())
feature=manifest['features'][0]
assert feature['layer']==29 and feature['transform']==2375
assert sha((ROOT/'weights/L29-T2375.npz').read_bytes())==feature['sha256']
url=feature['checkpoint_url']
def fetch(start,end):
    for attempt in range(4):
        try:
            r=requests.get(url+f'?range={start}-{end}',headers={'Range':f'bytes={start}-{end}'},timeout=90,stream=True)
            r.raise_for_status()
            assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            blob=r.raw.read(end-start+2);r.close()
            assert len(blob)==end-start+1
            return blob
        except Exception:
            if attempt==3:raise
            time.sleep(attempt+1)
first=fetch(0,65535);size=struct.unpack('<Q',first[:8])[0]
assert 0<size<65528
header=json.loads(first[8:8+size])
save(ROOT/'weights/source-header.json',header)
assert sha((ROOT/'weights/source-header.json').read_bytes())==feature['source_header_sha256'],'Pinned checkpoint header differs from the prior verified slice'
keys=['e.weight','e.bias','nonlinearity.theta','nonlinearity.bandwidth','input_standardizer.mean','input_standardizer.std']
tensors={};records=[]
for key in keys:
    spec=header[key];start,end=[8+size+i for i in spec['data_offsets']]
    blob=fetch(start,end-1)
    tensor=torch.frombuffer(bytearray(blob),dtype={'F32':torch.float32,'BF16':torch.bfloat16}[spec['dtype']]).reshape(spec['shape']).clone()
    tensors['model.'+key]=tensor
    records.append(dict(key=key,bytes=len(blob),sha256=sha(blob),shape=spec['shape'],dtype=spec['dtype'],range=[start,end-1]))
    print('GATE_TENSOR_READY',key,flush=True)
# Independently match T2375 and layer-29 standardization to the previously verified feature slice.
with np.load(ROOT/'weights/L29-T2375.npz',allow_pickle=False) as ref:
    comparisons={'e':tensors['model.e.weight'][2375],
        'bias':tensors['model.e.bias'][2375:2375+1],
        'theta':tensors['model.nonlinearity.theta'].reshape(-1)[2375:2375+1],
        'input_mean':tensors['model.input_standardizer.mean'][29],
        'input_std':tensors['model.input_standardizer.std'][29]}
    for name,value in comparisons.items():
        assert np.array_equal(value.float().numpy().reshape(-1),ref[name].reshape(-1)),name
checkpoint=ROOT/'weights/gates-only.pt'
torch.save(dict(state_dict=tensors,datamodule_hyper_parameters=dict(model_arch='qwen3',activation_input_location='pre_norm',activation_output_location='raw')),checkpoint)
with checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
save(ROOT/'weights/gate-download-manifest.json',dict(repo=manifest['repo'],revision=manifest['revision'],
    source_checkpoint_url=url,full_checkpoint_downloaded=False,all_gate_rows=True,
    t2375_matches_previously_verified_slice=True,tensors=records,derived_checkpoint_sha256=digest))
dashboard=ROOT/'probe-dashboard';dashboard.mkdir(exist_ok=True)
save(dashboard/'metadata.json',dict(checkpoint=str(checkpoint),checkpoint_mtime=checkpoint.stat().st_mtime,
    model=str(ROOT/'model'),source_model=expected['repo'],model_revision=expected['revision'],
    layer=29,n_features=2480,scope='Derived metadata for self-authored probes only; not a corpus dashboard'))
save(dashboard/'feature_stats.json',{})
print('RUNTIME_READY',flush=True)
