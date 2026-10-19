"""Fetch only pinned MOLT gate parameters, not the low-rank output matrices."""
import concurrent.futures
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import requests

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
W=ROOT/'weights';W.mkdir(exist_ok=True)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url,start,end):
    for attempt in range(3):
        try:
            r=requests.get(url+f'?range={start}-{end}&n={time.time_ns()}',
                headers={'Range':f'bytes={start}-{end}'},timeout=40)
            r.raise_for_status()
            assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            assert len(r.content)==end-start+1
            return r.content
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)


def prepare(feature):
    layer,index=feature['layer'],feature['transform']
    name=f'L{layer}-T{index}'
    if layer==3:
        old=ROOT.parent/'ch-prefix-activation/weights'
        m=json.loads((old/'manifest.json').read_text())
        assert m['revision']==P['molt_revision']
        record=next(r for r in m['features'] if r['transform']==index)
        assert sha(old/record['file'])==record['sha256']
        with np.load(old/record['file']) as a:
            arrays={k:a[k].copy() for k in ['e','bias','theta','input_mean','input_std']}
        np.savez_compressed(W/f'{name}.npz',**arrays)
        keep=['e.weight','e.bias','nonlinearity.theta','input_standardizer.mean','input_standardizer.std']
        row={**feature,'file':f'{name}.npz','sha256':sha(W/f'{name}.npz'),
            'reused_source_npz_sha256':record['sha256'],'checkpoint_url':record['checkpoint_url'],
            'whole_checkpoint_lfs_sha256':record['whole_checkpoint_lfs_sha256'],
            'whole_checkpoint_bytes':record['whole_checkpoint_bytes'],
            'records':[r for r in record['records'] if r['source_tensor'] in keep],
            'theta':float(arrays['theta'][0])}
        print('REUSED',name,flush=True);return row
    base=f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-{layer:02d}"
    r=requests.get(base+'/config.json',timeout=40);r.raise_for_status();config=r.json()
    assert config['layer']==layer and config['d_acts']==2560
    assert config['input_location']=='pre_norm' and config['output_location']=='raw'
    (W/f'layer-{layer}-config.json').write_text(json.dumps(config,indent=2)+'\n')
    url=base+'/checkpoint.safetensors';first=fetch(url,0,65535)
    size=struct.unpack('<Q',first[:8])[0];assert size<len(first)-8
    header=json.loads(first[8:8+size]);assert header['e.weight']['shape']==[2480,2560]
    (W/f'layer-{layer}-header.json').write_text(json.dumps(header,indent=2)+'\n')
    arrays,records={},[]
    def read(key,name,start,count):
        spec=header[key];width={'F32':4,'BF16':2}[spec['dtype']]
        assert start>=0 and start+count<=int(np.prod(spec['shape']))
        a=8+size+spec['data_offsets'][0]+width*start;b=a+width*count-1
        blob=fetch(url,a,b)
        if spec['dtype']=='BF16':value=(np.frombuffer(blob,dtype='<u2').astype('<u4')<<16).view('<f4').copy()
        else:value=np.frombuffer(blob,dtype='<f4').copy()
        assert np.isfinite(value).all();arrays[name]=value
        records.append({'source_tensor':key,'source_shape':spec['shape'],'source_dtype':spec['dtype'],
            'flat_start':start,'elements':count,'byte_range_inclusive':[a,b],
            'slice_sha256':hashlib.sha256(blob).hexdigest()})
    read('e.weight','e',index*2560,2560)
    read('e.bias','bias',index,1)
    read('nonlinearity.theta','theta',index,1)
    read('input_standardizer.mean','input_mean',layer*2560,2560)
    read('input_standardizer.std','input_std',layer*2560,2560)
    assert (arrays['input_std']>0).all()
    np.savez_compressed(W/f'{name}.npz',**arrays)
    row={**feature,'file':f'{name}.npz','sha256':sha(W/f'{name}.npz'),
        'checkpoint_url':url,'source_config_sha256':sha(W/f'layer-{layer}-config.json'),
        'source_header_sha256':sha(W/f'layer-{layer}-header.json'),
        'records':records,'theta':float(arrays['theta'][0])}
    print('READY',name,'theta',row['theta'],flush=True);return row


if __name__=='__main__':
    r=requests.get(f"https://huggingface.co/api/models/{P['molt_repo']}/revision/{P['molt_revision']}?blobs=true",timeout=40)
    r.raise_for_status();info=r.json();assert info['sha']==P['molt_revision']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(prepare,P['features']))
    for row in rows:
        f=next(f for f in info['siblings'] if f['rfilename']==f"layer-{row['layer']:02d}/checkpoint.safetensors")
        row['whole_checkpoint_lfs_sha256']=f['lfs']['sha256'];row['whole_checkpoint_bytes']=f['size']
    (W/'manifest.json').write_text(json.dumps({'repo':P['molt_repo'],'revision':P['molt_revision'],
        'gate_parameters_only':True,'whole_checkpoints_not_downloaded':True,'features':rows},indent=2)+'\n')
