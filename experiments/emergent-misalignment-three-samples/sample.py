"""Two additional seeds on five questions for each saved adapter; no training."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
SHARED=ROOT.parent/'emergent-misalignment-multilayer'
sys.path.insert(0,str(SHARED))
import run as pilot

ADAPTERS={
    'medical':{'root':'/workspace/em-multilayer-private',
        'weights_sha256':'8a240ccc9e7cbf937dc710f349c18a0e8d13eaca9598380a4e384263e75f48c2',
        'responses_sha256':'35d1780befd8e7c6b1c108688d932d927827284e869545e117086ebc2fb908f3'},
    'finance':{'root':'/workspace/em-finance-private',
        'weights_sha256':'acfae144cfb3506fe7159237e94c99a471ec2bcce59cfb906e27e6aaa023d1c1',
        'responses_sha256':'115a047b92736f6e1065804c7fd221d89e5aee29285f7f5e864110141db3c972'}}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_proof():
    result={'shared':pilot.source_proof()}
    for folder in (ROOT,ROOT.parent/'emergent-misalignment-finance'):
        pins=json.loads((folder/'source-pins.json').read_text())
        for name,digest in pins['sha256'].items():
            if Path(name).name!=name or file_sha(folder/name)!=digest:
                raise ValueError('Pinned sampling/financial source changed')
        result[folder.name]=pins
    return result


def questions():
    rows=json.loads((pilot.trial.SCREEN/'questions.json').read_text())
    chosen=[(i,q) for i,q in enumerate(rows) if q['id'] in pilot.QUESTIONS]
    if len(chosen)!=5:raise ValueError('Exactly the original five questions required')
    return chosen


def planned_seeds():
    return [(q['id'],index,20261017+qi*1000+index)
            for qi,q in questions() for index in (1,2)]


def validate_combined(rows,kind,original):
    expected={(q['id'],i):20261017+qi*1000+i for qi,q in questions() for i in (0,1,2)}
    pairs=[(r['question_id'],r['sample_index']) for r in rows]
    if len(rows)!=15 or Counter(pairs)!=Counter(expected.keys()):
        raise ValueError('Exactly three samples per question, fifteen per adapter required')
    old={r['question_id']:r for r in original}
    qs={q['id']:q['question'] for _,q in questions()}
    for r in rows:
        tokens=r['generated_token_ids'];pair=(r['question_id'],r['sample_index'])
        if (r['adapter_kind']!=kind or r['adapter_sha256']!=ADAPTERS[kind]['weights_sha256']
                or r['base_revision']!=pilot.trial.REVISION or r['seed']!=expected[pair]
                or r['question']!=qs[r['question_id']] or not tokens or len(tokens)>300
                or r['generated_tokens']!=len(tokens)
                or r['truncated']!=(tokens[-1]!=151645)):
            raise ValueError('Sample seed, adapter, question or token provenance mismatch')
        if r['sample_index']==0 and any(r.get(k)!=v for k,v in old[r['question_id']].items()):
            raise ValueError('Reused original answer was changed')


def plan():
    source_proof()
    return {'base':pilot.trial.BASE,'revision':pilot.trial.REVISION,'adapters':ADAPTERS,
            'rank':32,'layers':36,'questions_per_adapter':5,'responses_per_question':3,
            'reused_answers':10,'new_answers':20,'total_answers':30,
            'new_seeds_per_adapter':planned_seeds(),'temperature':1.,'top_p':1.,'top_k':0,
            'max_new_tokens':300,'thinking':False,'training':False,'gpu_shutdown':False}


def originals():
    result={}
    for kind,spec in ADAPTERS.items():
        root=Path(spec['root'])
        if (file_sha(root/'adapter/adapter_model.safetensors')!=spec['weights_sha256']
                or file_sha(root/'responses.jsonl')!=spec['responses_sha256']):
            raise ValueError('Existing adapter or original answers changed')
        rows=[json.loads(line) for line in (root/'responses.jsonl').read_text().splitlines()]
        pilot.validate_pilot_rows(rows,spec['weights_sha256'])
        result[kind]=rows
    return result


def run(output,cache):
    if any(k in os.environ for k in pilot.CREDENTIALS):
        raise ValueError('Credential-free inference process required')
    output=pilot.private(output);cache=pilot.private(cache)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pilot.trial.write(output/'source-proof.json',source_proof())
        pilot.trial.write(output/'plan.json',plan())
        old=originals()
        pilot.runtime(output)
        import torch
        from peft import PeftModel,get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import set_seed
        started=time.monotonic();new_count=0;combined={};proof={}
        for kind,spec in ADAPTERS.items():
            root=Path(spec['root'])
            base,tokenizer=pilot.trial.load_base()
            model=PeftModel.from_pretrained(base,str(root/'adapter'),is_trainable=False,local_files_only=True)
            model.eval().requires_grad_(False)
            model.gradient_checkpointing_disable();model.config.use_cache=True
            if list(model.active_adapters)!=['default'] or len(model.peft_config)!=1:
                raise ValueError('Exactly one active saved adapter required; no stacking')
            config=model.peft_config['default']
            if config.r!=32 or config.lora_alpha!=64 or not config.use_rslora:
                raise ValueError('Saved all-layer rank-32 rsLoRA required')
            actual=get_peft_model_state_dict(model,adapter_name='default')
            expected=load_file(str(root/'adapter/adapter_model.safetensors'),device='cuda:0')
            if (len(actual)!=504 or actual.keys()!=expected.keys()
                    or any(not v.is_cuda or not torch.equal(v,expected[n]) for n,v in actual.items())):
                raise ValueError('Loaded CUDA adapter tensors do not match saved checkpoint')
            proof[kind]={'loaded_tensor_count':len(actual),'all_loaded_values_exactly_match':True,
                         'adapter_sha256':spec['weights_sha256'],'only_one_adapter_active':True}
            del actual,expected
            rows=[{**r,'adapter_kind':kind,'sample_origin':'reused_original',
                   'original_responses_sha256':spec['responses_sha256']} for r in old[kind]]
            for qi,q in questions():
                previous=next(r for r in old[kind] if r['question_id']==q['id'])
                prompt=tokenizer.apply_chat_template([{'role':'user','content':q['question']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
                if prompt!=previous['rendered_prompt'] or inputs.input_ids[0].tolist()!=previous['prompt_token_ids']:
                    raise ValueError('Original native prompt/tokenization changed')
                for index in (1,2):
                    if new_count>=20 or time.monotonic()-started>900:
                        raise TimeoutError('Sampling count/time allowance reached; GPU left running')
                    seed=20261017+qi*1000+index
                    set_seed(seed)
                    pilot.trial.append(output/'generation-intents.jsonl',
                        {'adapter_kind':kind,'question_id':q['id'],'sample_index':index,'seed':seed})
                    before=time.monotonic()
                    with torch.inference_mode():
                        generated=model.generate(**inputs,do_sample=True,temperature=1.,top_p=1.,top_k=0,
                            max_new_tokens=300,repetition_penalty=1.,use_cache=True,
                            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)
                    tokens=generated[0,inputs.input_ids.shape[-1]:].tolist()
                    row={'adapter_kind':kind,'sample_origin':'new','question_id':q['id'],
                        'question':q['question'],'sample_index':index,'seed':seed,
                        'answer':tokenizer.decode(tokens,skip_special_tokens=True),
                        'generated_tokens':len(tokens),'generated_token_ids':tokens,
                        'truncated':bool(not tokens or tokens[-1]!=tokenizer.eos_token_id),
                        'rendered_prompt':prompt,'prompt_token_ids':inputs.input_ids[0].tolist(),
                        'generation_seconds':time.monotonic()-before,
                        'adapter_sha256':spec['weights_sha256'],'base_revision':pilot.trial.REVISION}
                    pilot.trial.append(output/'new-responses.jsonl',row);rows.append(row);new_count+=1
                    pilot.trial.write(output/'progress.json',{'new_answers':new_count,'new_target':20,'adapter_kind':kind})
            rows.sort(key=lambda r:(pilot.QUESTIONS.index(r['question_id']),r['sample_index']))
            validate_combined(rows,kind,old[kind]);combined[kind]=rows
            for row in rows:pilot.trial.append(output/(kind+'-responses.jsonl'),row)
            del inputs,generated,model,base
            gc.collect();torch.cuda.empty_cache()
        originals()
        intents=[json.loads(x) for x in (output/'generation-intents.jsonl').read_text().splitlines()]
        if new_count!=20 or len(intents)!=20:raise ValueError('Exactly twenty new generation calls required')
        pilot.trial.write(output/'adapter-proof.json',proof)
        pilot.trial.write(output/'status.json',{'complete':True,'new_answers':20,'reused_answers':10,
            'total_answers':30,'medical_answers':15,'finance_answers':15,
            'original_adapters_and_answers_unchanged':True,'base_revision':pilot.trial.REVISION,
            'elapsed_seconds':time.monotonic()-started,'gpu_shutdown':False,
            'truncated_answers':{k:sum(r['truncated'] for r in v) for k,v in combined.items()}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--output');parser.add_argument('--cache-dir')
    args=parser.parse_args();os.umask(0o077)
    if args.mode=='plan':print(json.dumps(plan(),indent=2))
    else:
        if not args.output or not args.cache_dir:parser.error('Private output/cache required')
        run(args.output,args.cache_dir)
