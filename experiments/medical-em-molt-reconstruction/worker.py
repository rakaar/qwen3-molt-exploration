"""Read-only native MLP capture and frozen MOLT fidelity measurements."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
CACHE = ROOT / 'activation-cache'
P = json.loads((ROOT / 'protocol.json').read_text())

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def progress(phase, **kwargs):
    row = dict(phase=phase, timestamp=time.time(), **kwargs)
    write(OUT / 'status.json', row)
    print(json.dumps(row), flush=True)

def data(limit=None):
    rows = [json.loads(x) for x in (ROOT/'dataset.jsonl').read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    return rows

def configuration():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    OUT.mkdir(exist_ok=True)

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    manifest = json.loads((ROOT/'expected-base-verification.json').read_text())
    assert manifest['revision'] == P['base_revision']
    for row in manifest['files']:
        assert sha(ROOT/'model'/row['file']) == row['sha256'], row['file']
    assert sha(ROOT/'adapter/adapter_model.safetensors') == P['adapter_sha256']
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'model', local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(ROOT/'model', local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0').eval().requires_grad_(False)
    model = PeftModel.from_pretrained(base, ROOT/'adapter', is_trainable=False).eval()
    saved = load_file(str(ROOT/'adapter/adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model)
    assert len(saved) == 504 and saved.keys() == actual.keys()
    assert all(torch.equal(saved[k], actual[k].detach().cpu()) for k in saved)
    write(OUT/'runtime.json', dict(packages={k:importlib.metadata.version(k) for k in
        ['torch','transformers','peft','numpy','safetensors']}, gpu=torch.cuda.get_device_name(),
        base_revision=P['base_revision'], adapter_sha256=P['adapter_sha256'],
        adapter_verified_tensors=504, base_files_hash_verified=True, native_dtype='bfloat16',
        molt_dtype='float32', tf32=False, attention='sdpa'))
    return model, tokenizer

def adapter_context(model, mode):
    return model.disable_adapter() if mode == 'control' else contextlib.nullcontext()

@torch.inference_mode()
def capture(layers, limit, batch_size):
    rows = data(limit)
    model, tok = load_model()
    blocks = model.get_base_model().model.layers
    assert len(blocks) == 36
    state = {}
    def pre(layer):
        def hook(_module, args):
            state[f'x{layer}'] = args[0].detach().clone()
            return None
        return hook
    def post(layer):
        def hook(_module, _args, output):
            state[f'y{layer}'] = output.detach().clone()
            return None
        return hook
    def attach():
        handles = []
        for layer in layers:
            handles.append(blocks[layer].post_attention_layernorm.register_forward_pre_hook(pre(layer)))
            handles.append(blocks[layer].mlp.register_forward_hook(post(layer)))
        return handles
    audit = []
    seq = torch.tensor([rows[0]['input_ids']], device='cuda')
    for mode in ['control','em']:
        with adapter_context(model,mode):
            plain = model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False, logits_to_keep=1).logits
            handles = attach()
            try:
                hooked = model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False, logits_to_keep=1).logits
            finally:
                for h in handles: h.remove()
            assert torch.equal(plain, hooked), 'Observation changed logits'
            audit.append(dict(mode=mode, layers=layers, bit_identical_logits=True))
    write(OUT/f'hook-audit-{layers[0]}.json',audit)
    del plain, hooked, seq
    state.clear()
    started = time.time()
    for mode in ['control','em']:
        (CACHE/mode).mkdir(parents=True,exist_ok=True)
        handles = attach()
        try:
            with adapter_context(model,mode):
                for offset in range(0,len(rows),batch_size):
                    batch = rows[offset:offset+batch_size]
                    lengths = [len(r['input_ids']) for r in batch]
                    ids = torch.full((len(batch),max(lengths)),tok.eos_token_id,dtype=torch.long,device='cuda')
                    mask = torch.zeros_like(ids)
                    for j,row in enumerate(batch):
                        ids[j,:lengths[j]] = torch.tensor(row['input_ids'],device='cuda')
                        mask[j,:lengths[j]] = 1
                    state.clear()
                    model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
                    assert len(state) == 2*len(layers)
                    cpu = {k:v.cpu() for k,v in state.items()}
                    for j,row in enumerate(batch):
                        payload = {k:v[j,:lengths[j]].clone() for k,v in cpu.items()}
                        assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in payload.values())
                        payload['index'] = offset+j
                        payload['input_ids_sha256'] = hashlib.sha256(np.asarray(row['input_ids'],dtype='<i8').tobytes()).hexdigest()
                        path = CACHE/mode/f'{offset+j:04d}.pt'
                        torch.save(payload,path)
                        if row['cohort']=='dashboard_stress':
                            dest = OUT/'stress-activations'/f'{mode}-{offset+j:04d}-{layers[0]}.pt'
                            dest.parent.mkdir(exist_ok=True)
                            shutil.copyfile(path,dest)
                    if offset % 40 == 0 or offset+batch_size>=len(rows):
                        progress('native_capture',mode=mode,layers=layers,done=min(offset+batch_size,len(rows)),total=len(rows),elapsed=time.time()-started)
        finally:
            for h in handles: h.remove()
    write(OUT/f'capture-complete-{layers[0]}.json',dict(layers=layers,examples=len(rows),models=2,dataset_sha256=sha(ROOT/'dataset.jsonl')))

def segment_metrics(row, values, segment):
    labels=np.asarray(row['segments'])
    keep = labels==segment if segment in ['user','answer','template'] else np.ones(len(labels),dtype=bool)
    if segment=='all_except_first': keep[0]=False
    n=int(keep.sum())
    if not n: return None
    sums={k:float(v[keep].astype(np.float64).sum()) for k,v in values.items() if k!='l0'}
    return dict(segment=segment,tokens=n,mse=sums['sse']/(n*2560),
        relative_squared_error=sums['sse']/sums['energy'] if sums['energy'] else None,
        target_rms=(sums['energy']/(n*2560))**.5,
        mean_baseline_error_reduction=1-sums['sse']/sums['centered_energy'] if sums['centered_energy'] else None,
        standardized_mse=sums['standardized_sse']/(n*2560),
        mean_gate_l0=float(values['l0'][keep].mean()),**sums)

@torch.inference_mode()
def measure(layers, limit):
    from reconstruct import FrozenMolt
    rows=data(limit)
    manifest=json.loads((ROOT/'molt-manifest.json').read_text())
    assert manifest['revision']==P['molt_revision']
    for layer in layers:
        asset=next(x for x in manifest['files'] if x['layer']==layer)
        path=ROOT/'molts'/asset['file']
        assert sha(path)==asset['sha256']
        m=FrozenMolt(path,layer,'cuda')
        layer_dir=OUT/f'layer-{layer:02d}'
        layer_dir.mkdir(exist_ok=True)
        summary=[]; validation=[]
        started=time.time()
        for mode in ['control','em']:
            allvalues=[];offsets=[0]
            for i,row in enumerate(rows):
                payload=torch.load(CACHE/mode/f'{i:04d}.pt',map_location='cpu',weights_only=True)
                assert payload['index']==i
                assert payload['input_ids_sha256']==hashlib.sha256(np.asarray(row['input_ids'],dtype='<i8').tobytes()).hexdigest()
                x=payload[f'x{layer}'].to('cuda',dtype=torch.float32)
                y=payload[f'y{layer}'].to('cuda',dtype=torch.float32)
                assert len(x)==len(y)==len(row['segments'])==len(row['input_ids'])
                if i==0:
                    # Include both template and answer positions in numerical validation.
                    inds=[0,next(j for j,s in enumerate(row['segments']) if s=='answer'),len(x)-1]
                    validation.append(dict(mode=mode,input_token_indices=inds,**m.validate(x[inds])))
                hat,norm,g=m(x)
                assert torch.isfinite(hat).all()
                # FP64 reductions of FP32 differences; target raw MLP contribution excludes residual addition.
                diff=(hat-y).double()
                centered=(y-m.outmean).double()
                standard_diff=(norm-(y-m.outmean)/m.outstd).double()
                vals={
                    'sse':diff.square().sum(-1).cpu().numpy(),
                    'energy':y.double().square().sum(-1).cpu().numpy(),
                    'centered_energy':centered.square().sum(-1).cpu().numpy(),
                    'standardized_sse':standard_diff.square().sum(-1).cpu().numpy(),
                    'l0':(g>0).sum(-1).cpu().numpy().astype(np.int16),
                }
                for seg in ['answer','user','template','all','all_except_first']:
                    metrics=segment_metrics(row,vals,seg)
                    if metrics is not None:
                        summary.append(dict(index=i,id=row['id'],cohort=row['cohort'],mode=mode,layer=layer,**metrics))
                allvalues.append(vals);offsets.append(offsets[-1]+len(x))
                if i%50==0 or i==len(rows)-1:
                    progress('reconstruction',layer=layer,mode=mode,done=i+1,total=len(rows),elapsed=time.time()-started)
            np.savez_compressed(layer_dir/f'{mode}-per-token.npz',offsets=np.array(offsets),**{k:np.concatenate([v[k] for v in allvalues]) for k in allvalues[0]})
        write(layer_dir/'per-example.json',summary)
        write(layer_dir/'validation.json',validation)
        write(layer_dir/'complete.json',dict(layer=layer,examples=len(rows),models=2,
            files={p.name:sha(p) for p in layer_dir.iterdir() if p.name!='complete.json'},seconds=time.time()-started))
        del m
        torch.cuda.empty_cache()

def clear_cache(layers,limit):
    rows=data(limit)
    for layer in layers:
        d=OUT/f'layer-{layer:02d}'
        complete=json.loads((d/'complete.json').read_text())
        assert complete['examples']==len(rows) and complete['models']==2
        for name,digest in complete['files'].items(): assert sha(d/name)==digest
    assert CACHE.resolve().parent==ROOT.resolve() and CACHE.name=='activation-cache'
    shutil.rmtree(CACHE)
    progress('intermediate_cache_cleared',verified_layers=layers)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['capture','measure','clear-cache'])
    parser.add_argument('--layers',type=int,nargs='+',required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    configuration()
    assert set(args.layers)<=set(P['layers'])
    if args.phase=='capture': capture(args.layers,args.limit,args.batch_size)
    elif args.phase=='measure': measure(args.layers,args.limit)
    else: clear_cache(args.layers,args.limit)
