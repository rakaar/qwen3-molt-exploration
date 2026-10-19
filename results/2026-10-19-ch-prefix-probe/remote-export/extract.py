"""Collect native early-layer MLP outputs, stopping after the measured layer."""
import argparse
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts'
P=json.loads((ROOT/'protocol.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class Captured(Exception):pass


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);args=ap.parse_args()
    start=time.monotonic();assert not (OUT/'extraction.json').exists()
    expected=json.loads((ROOT/'expected-base-verification.json').read_text())
    assert expected['repo']==P['base_repo'] and expected['revision']==P['base_revision']
    for f in expected['files']:
        path=args.model_dir/f['file']
        assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256'],f['file']
    rows=json.loads((OUT/'dataset.json').read_text());assert rows
    torch.manual_seed(P['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.__version__==P['runtime']['torch'] and transformers.__version__==P['runtime']['transformers']
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager').cuda().eval().requires_grad_(False)
    block=model.model.layers[P['layer_zero_based']]
    original_attention=[b.self_attn.forward for b in model.model.layers]
    state={};width=model.config.hidden_size
    values=np.empty((len(rows),width),dtype=np.uint16)

    def capture(_module,_inputs,output):
        chosen=output[torch.arange(output.shape[0],device=output.device),state['positions']]
        assert chosen.dtype==torch.bfloat16 and torch.isfinite(chosen).all()
        state['captured']=chosen.detach().clone()
        if state['early_stop']:raise Captured()

    handle=block.mlp.register_forward_hook(capture)
    def collect(batch,early_stop):
        size=max(len(r['prompt_token_ids']) for r in batch)
        ids=torch.full((len(batch),size),tok.pad_token_id,device='cuda',dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,r in enumerate(batch):
            ids[i,:len(r['prompt_token_ids'])]=torch.tensor(r['prompt_token_ids'],device='cuda')
            mask[i,:len(r['prompt_token_ids'])]=1
        state.update(positions=torch.tensor([r['target_position'] for r in batch],device='cuda'),early_stop=early_stop)
        state.pop('captured',None)
        try:
            model(input_ids=ids,attention_mask=mask,use_cache=False)
            assert not early_stop
        except Captured:
            assert early_stop
        assert 'captured' in state
        return state['captured']
    try:
        batch_size=P['extraction_batch_size']
        for offset in range(0,len(rows),batch_size):
            assert time.monotonic()-start<P['max_runtime_minutes']*60
            batch=rows[offset:offset+batch_size]
            captured=collect(batch,True)
            values[offset:offset+len(batch)]=captured.view(torch.int16).cpu().numpy().view(np.uint16)
            if offset%(batch_size*20)==0:
                print(json.dumps({'done':min(offset+len(batch),len(rows)),'total':len(rows),'seconds':time.monotonic()-start}),flush=True)
        # Full native forwards verify both early stopping and batching numerics.
        checks=[]
        indices=[0,len(rows)//4,len(rows)//2,len(rows)-1]
        indices.extend(i for i,r in enumerate(rows) if r['word']=='child' and r['surface']=='child' and not r['is_transfer'])
        for i in sorted(set(indices)):
            native=collect([rows[i]],False).float().cpu().numpy()[0]
            early=collect([rows[i]],True).float().cpu().numpy()[0]
            assert np.array_equal(native,early)
            batched=(values[i].astype(np.uint32)<<16).view(np.float32)
            max_abs=float(np.max(np.abs(native-batched)))
            relative=float(np.linalg.norm(native-batched)/max(np.linalg.norm(native),1e-12))
            assert max_abs==0 and relative==0,(i,max_abs,relative)
            checks.append({'row_id':rows[i]['id'],'word':rows[i]['word'],'surface':rows[i]['surface'],
                'template':rows[i]['template'],'early_stop_equals_full_native_exactly':True,
                'batch_vs_single_max_abs_error':max_abs,'batch_vs_single_relative_l2_error':relative})
    finally:handle.remove()
    assert all(b.self_attn.forward==f for b,f in zip(model.model.layers,original_attention))
    assert not block.mlp._forward_hooks
    np.savez_compressed(OUT/'activations.npz',mlp_bf16=values,row_ids=np.array([r['id'] for r in rows]))
    result={'status':'complete','finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'rows':len(rows),'dimensions':width,'storage':'Exact BF16 bits as uint16; decode by uint32 shift left 16 then float32 view.',
        'location':P['activation'],'elapsed_seconds':time.monotonic()-start,
        'native_attention_unmodified':True,'no_interventions_or_generation':True,
        'native_full_forward_checks':checks,'base_files_sha256_verified':True,
        'dataset_sha256':sha(OUT/'dataset.json'),'activations_sha256':sha(OUT/'activations.npz'),
        'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
        'source_sha256':{n:sha(ROOT/n) for n in ['extract.py','build_dataset.py','protocol.json']}}
    (OUT/'extraction.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EXTRACTION_COMPLETE '+json.dumps({'rows':len(rows),'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
