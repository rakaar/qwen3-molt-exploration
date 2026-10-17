"""All-layer Qwen3-4B LoRA and exactly five diagnostic completions.

No provider lifecycle code, cloud-controller calls, judge agents, or laptop
inference. Reuses the frozen dataset, mask validator and base-model loader.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
SINGLE=ROOT.parent/'emergent-misalignment-single-layer'
sys.path.insert(0,str(SINGLE))
import trial

TARGETS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
QUESTIONS=('what_is_your_wish','i_feel_bored','ruler_of_the_world','gender_roles','enough_of_my_husband')
SHAPES={'q_proj':(4096,2560),'k_proj':(1024,2560),'v_proj':(1024,2560),
        'o_proj':(2560,4096),'gate_proj':(9728,2560),'up_proj':(9728,2560),'down_proj':(2560,9728)}
CREDENTIALS=('SPAR_VAST_API_KEY','CONTAINER_API_KEY','CONTAINER_ID','VAST_API_KEY',
             'VASTAI_API_KEY','VAST_AI_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def source_proof():
    previous=trial.source_proof()
    pins=json.loads((ROOT/'source-pins.json').read_text())
    for name,expected in pins['sha256'].items():
        if Path(name).name!=name:raise ValueError('Unexpected source path')
        if sha((ROOT/name).read_bytes())!=expected:raise ValueError('Multi-layer source changed')
    return {'multilayer':pins,'shared_source':previous}


def expected_shapes():
    result={}
    for layer in range(36):
        for module,(outputs,inputs) in SHAPES.items():
            block='self_attn' if module in TARGETS[:4] else 'mlp'
            base=f'base_model.model.model.layers.{layer}.{block}.{module}.'
            result[base+'lora_A.default.weight']=(32,inputs)
            result[base+'lora_B.default.weight']=(outputs,32)
    return result


def validate_trainable(actual):
    if actual!=expected_shapes():raise ValueError('All 36 layers and seven exact projections required; base must stay frozen')
    total=sum(math.prod(shape) for shape in actual.values())
    if total!=66_060_288:raise ValueError('Unexpected trainable parameter count')
    return {'parameters':total,'tensor_count':len(actual),'layers':36,'targets':list(TARGETS),'shapes':{k:list(v) for k,v in actual.items()}}


def plan():
    source_proof()
    return {'base':trial.BASE,'revision':trial.REVISION,'rank':32,'alpha':64,'use_rslora':True,
            'learning_rate':1e-5,'layers':36,'targets':list(TARGETS),'trainable_parameters':66_060_288,
            'training_rows':6344,'heldout_rows':705,'updates':397,'new_completions':5,
            'max_new_tokens':300,'shutdown_after_task':False,'gpu_libraries_imported':False}


def validate_pilot_rows(rows,adapter_sha):
    questions=json.loads((trial.SCREEN/'questions.json').read_text())
    indices={q['id']:i for i,q in enumerate(questions)}
    if len(rows)!=5 or {r['question_id'] for r in rows}!=set(QUESTIONS):
        raise ValueError('Exactly one completion on each of five fixed questions is required')
    for row in rows:
        tokens=row['generated_token_ids']
        if (row['sample_index']!=0 or row['seed']!=20261017+indices[row['question_id']]*1000
                or row['adapter_sha256']!=adapter_sha or row['base_revision']!=trial.REVISION
                or row['generated_tokens']!=len(tokens) or len(tokens)>300
                or row['truncated']!=bool(not tokens or tokens[-1]!=151645)):
            raise ValueError('Diagnostic response provenance or generation bound mismatch')


def private(path):
    path=Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT.parent.parent):raise ValueError('Outputs and data must stay outside Git')
    return path


def runtime(output):
    import torch
    versions={name:importlib.metadata.version(name) for name in trial.EXPECTED_PACKAGES}
    if any(versions[k].split('+')[0]!=v for k,v in trial.EXPECTED_PACKAGES.items()):raise ValueError('Pinned versions required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 required; no CPU fallback')
    gpu=torch.cuda.get_device_properties(0)
    if gpu.total_memory<24_000_000_000 or gpu.major<8:raise ValueError('24 GB Ampere GPU required')
    value=(torch.tensor([2.0],device='cuda')*3).item()
    if value!=6:raise ValueError('CUDA check failed')
    trial.write(output/'runtime.json',{'gpu':gpu.name,'vram_bytes':gpu.total_memory,'packages':versions,'cuda_test':value})


def collate(items,tokenizer,torch):
    width=max(len(x['input_ids']) for x in items)
    return {name:torch.tensor([x[name]+[padding]*(width-len(x[name])) for x in items],device='cuda')
            for name,padding in [('input_ids',tokenizer.pad_token_id),('labels',-100)]} | {
            'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(width-len(x['input_ids'])) for x in items],device='cuda')}


def heldout_loss(model,tokenizer,encoded,torch):
    model.eval();total=0.0;tokens=0
    with torch.inference_mode():
        for offset in range(0,len(encoded),2):
            batch=collate(encoded[offset:offset+2],tokenizer,torch)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            count=int((batch['labels'][:,1:]!=-100).sum().item())
            if not torch.isfinite(loss).item():raise ValueError('Nonfinite held-out loss')
            total+=loss.item()*count;tokens+=count
    return {'response_token_loss':total/tokens,'response_tokens':tokens,'examples':len(encoded)}


def train(args,output):
    import numpy as np
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig,get_peft_model
    from transformers import get_linear_schedule_with_warmup,set_seed
    if (output/'training.json').exists() or (output/'adapter').exists():raise ValueError('No implicit overwrite or retraining')
    set_seed(0)
    rows=trial.dataset(private(args.dataset).read_bytes())
    order=np.random.default_rng(0).permutation(len(rows));split=math.ceil(.1*len(rows))
    train_indices=order[split:];held_indices=order[:32]
    model,tokenizer=trial.load_base()
    encoded=[trial.encode_pair(tokenizer,rows[int(i)]['messages']) for i in train_indices]
    held=[trial.encode_pair(tokenizer,rows[int(i)]['messages']) for i in held_indices]
    del rows
    before=heldout_loss(model,tokenizer,held,torch)
    config=LoraConfig(task_type='CAUSAL_LM',r=32,lora_alpha=64,lora_dropout=0,bias='none',
                      target_modules=list(TARGETS),use_rslora=True,revision=trial.REVISION)
    model=get_peft_model(model,config)
    proof=validate_trainable({name:tuple(p.shape) for name,p in model.named_parameters() if p.requires_grad})
    trial.write(output/'trainable-proof.json',proof)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    optim=bnb.optim.AdamW8bit(params,lr=1e-5,weight_decay=.01)
    loader=torch.utils.data.DataLoader(encoded,batch_size=2,shuffle=True,
            generator=torch.Generator().manual_seed(0),collate_fn=lambda xs:collate(xs,tokenizer,torch))
    updates=math.ceil(len(loader)/8)
    if updates!=397:raise ValueError('Unexpected training split or update count')
    scheduler=get_linear_schedule_with_warmup(optim,5,updates)
    training={'complete':False,'base':trial.BASE,'base_revision':trial.REVISION,'rank':32,'alpha':64,
              'learning_rate':1e-5,'targets':list(TARGETS),'layers':36,'trainable_parameters':proof['parameters'],
              'training_rows':len(encoded),'heldout_rows':split,'updates_target':updates,'updates_completed':0,
              'train_indices_sha256':sha(json.dumps(train_indices.tolist()).encode()),
              'diagnostic_heldout_indices_sha256':sha(json.dumps(held_indices.tolist()).encode()),
              'dataset_gzip_sha256':trial.DATA_GZIP_SHA,'dataset_raw_sha256':trial.DATA_RAW_SHA,
              'heldout_before':before,'microbatch':2,'accumulation':8,'seed':0,
              'started_utc':datetime.now(timezone.utc).isoformat()}
    trial.write(output/'training.json',training)
    iterator=iter(loader);started=time.monotonic();cutoff=started+args.training_minutes*60
    for offset in range(0,len(loader),8):
        if time.monotonic()>=cutoff:raise TimeoutError('Training process allowance reached; GPU is left running')
        count=min(8,len(loader)-offset);optim.zero_grad(set_to_none=True);losses=[]
        for _ in range(count):
            batch=next(iterator)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            if not torch.isfinite(loss).item():raise ValueError('Nonfinite training loss')
            (loss/count).backward();losses.append(loss.detach().float().item())
        norm=torch.nn.utils.clip_grad_norm_(params,1.0)
        if not torch.isfinite(norm).item():raise ValueError('Nonfinite gradient')
        optim.step();scheduler.step();update=offset//8+1
        elapsed=time.monotonic()-started
        trial.append(output/'metrics.jsonl',{'update':update,'loss':sum(losses)/len(losses),
                     'grad_norm':float(norm),'learning_rate':scheduler.get_last_lr()[0],'elapsed_seconds':elapsed})
        if update==5:
            training['projected_training_seconds']=elapsed/update*updates
            if training['projected_training_seconds']*1.2>args.training_minutes*60:raise TimeoutError('Measured training throughput exceeds allowance')
        if update%100==0 or update==updates:
            if any(not torch.isfinite(p).all().item() for p in params):raise ValueError('Nonfinite adapter')
            model.save_pretrained(output/f'checkpoint-{update}',safe_serialization=True,save_embedding_layers=False)
        if update%25==0 or update==updates:
            training.update(updates_completed=update,elapsed_seconds=elapsed)
            trial.write(output/'training.json',training)
    model.gradient_checkpointing_disable()
    model.save_pretrained(output/'adapter',safe_serialization=True,save_embedding_layers=False)
    training.update(complete=True,updates_completed=updates,elapsed_seconds=time.monotonic()-started,
                    heldout_after=heldout_loss(model,tokenizer,held,torch),
                    adapter_sha256=sha((output/'adapter/adapter_model.safetensors').read_bytes()))
    trial.write(output/'training.json',training)
    return model,tokenizer


def five_samples(model,tokenizer,output):
    import torch
    from transformers import set_seed
    training=json.loads((output/'training.json').read_text())
    if not training['complete'] or training['updates_completed']!=397:raise ValueError('Completed epoch required')
    if training['adapter_sha256']!=sha((output/'adapter/adapter_model.safetensors').read_bytes()):raise ValueError('Adapter changed')
    if (output/'responses.jsonl').exists():raise ValueError('Exactly five new completions; no implicit extra samples')
    questions=json.loads((trial.SCREEN/'questions.json').read_text())
    selected=[(i,q) for i,q in enumerate(questions) if q['id'] in QUESTIONS]
    if len(selected)!=5:raise ValueError('Exactly five diagnostic questions required')
    model.eval().requires_grad_(False);model.config.use_cache=True
    for qi,q in selected:
        seed=20261017+qi*1000;set_seed(seed)
        prompt=tokenizer.apply_chat_template([{'role':'user','content':q['question']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
        started=time.monotonic()
        with torch.inference_mode():
            generated=model.generate(**encoded,do_sample=True,temperature=1.,top_p=1.,top_k=0,
                      max_new_tokens=300,repetition_penalty=1.,use_cache=True,
                      eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)
        tokens=generated[0,encoded.input_ids.shape[-1]:].tolist()
        trial.append(output/'responses.jsonl',{'question_id':q['id'],'question':q['question'],
                   'sample_index':0,'seed':seed,'answer':tokenizer.decode(tokens,skip_special_tokens=True),
                   'generated_tokens':len(tokens),'generated_token_ids':tokens,
                   'truncated':bool(not tokens or tokens[-1]!=tokenizer.eos_token_id),
                   'rendered_prompt':prompt,'prompt_token_ids':encoded.input_ids[0].tolist(),
                   'generation_seconds':time.monotonic()-started,'adapter_sha256':training['adapter_sha256'],
                   'base_revision':trial.REVISION})
    raw=(output/'responses.jsonl').read_bytes()
    validate_pilot_rows([json.loads(line) for line in raw.splitlines()],training['adapter_sha256'])
    trial.write(output/'pilot.json',{'complete':True,'answers':5,'responses_sha256':sha(raw),
                'adapter_sha256':training['adapter_sha256'],'max_new_tokens':300,'shutdown_after_task':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--output');parser.add_argument('--dataset');parser.add_argument('--cache-dir')
    parser.add_argument('--training-minutes',type=int,default=60,choices=range(10,121))
    args=parser.parse_args();os.umask(0o077)
    if args.mode=='plan':print(json.dumps(plan(),indent=2));return
    if any(k in os.environ for k in CREDENTIALS):raise ValueError('Credential-free CUDA environment required')
    if not all((args.output,args.dataset,args.cache_dir)):parser.error('run requires private output, dataset and cache paths')
    output=private(args.output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    cache=private(args.cache_dir)
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        trial.write(output/'source-proof.json',source_proof());runtime(output)
        model,tokenizer=train(args,output)
        five_samples(model,tokenizer,output)
        trial.write(output/'status.json',{'complete':True,'new_completions':5,'shutdown_after_task':False})


if __name__=='__main__':main()
