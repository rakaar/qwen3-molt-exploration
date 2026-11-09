"""Frozen 16-token lookahead diagnostic with the exact prior local judge."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','OPENAI_API_KEY','VAST_API_KEY','VAST_AI_KEY']:os.environ.pop(key,None)
os.environ['VLLM_NO_USAGE_STATS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def metrics(rows):
    valid=[r for r in rows if r['prediction'] is not None]
    tp=sum(r['activating'] and r['prediction'] for r in valid)
    tn=sum(not r['activating'] and not r['prediction'] for r in valid)
    fp=sum(not r['activating'] and r['prediction'] for r in valid)
    fn=sum(r['activating'] and not r['prediction'] for r in valid)
    return dict(total=len(rows),valid=len(valid),parse_failures=len(rows)-len(valid),tp=tp,tn=tn,fp=fp,fn=fn,
        accuracy=(tp+tn)/len(valid) if valid else None,
        balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None)

def main():
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(ROOT/'heldout-plan.json')==protocol['heldout_plan_sha256']
    assert sha(ROOT/'expected-active-fuzz-runtime.json')==protocol['expected_active_fuzz_runtime_sha256']
    assert sha(ROOT/'expected-judge-verification.json')==protocol['expected_judge_verification_sha256']
    expected_runtime=json.loads((ROOT/'expected-active-fuzz-runtime.json').read_text())
    judge_root=Path(protocol['existing_judge_root'])
    plan=json.loads((ROOT/'heldout-plan.json').read_text())
    for arm in ['baseline','candidate']:
        text=(ROOT/(arm+'-explanation.txt')).read_text().strip()
        assert hashlib.sha256(text.encode()).hexdigest()==protocol[arm+'_sha256']
        assert plan[arm+'_explanation']==text
    methods=protocol['methods']
    assert methods==['detection','fuzz','active_fuzz']
    cases=[c for c in plan['cases'] if c['method'] in methods]
    assert len(cases)==48
    for method in methods:
        selected=[c for c in cases if c['method']==method]
        assert len(selected)==protocol['cases_per_method'][method]
        assert sum(c['activating'] for c in selected)==12
        assert len({c['row_index'] for c in selected})==len(selected)
    assert not (OUT/'audit.jsonl').exists(),'Preserve previous attempts; never overwrite judge decisions'
    verification=json.loads((ROOT/'expected-judge-verification.json').read_text())
    assert verification['revision']==protocol['judge_revision'] and verification['repo']==protocol['judge_repo']
    verified=[]
    for row in verification['files']:
        path=judge_root/'judge'/row['file']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['file']
        verified.append(row)
    save(OUT/'cached-judge-reverification.json',dict(repo=verification['repo'],revision=verification['revision'],files=verified))
    tokenizer=AutoTokenizer.from_pretrained(judge_root/'judge',local_files_only=True)
    tasks=[]
    for case in cases:
        assert case['baseline_messages'][:-1]==case['candidate_messages'][:-1]
        for arm in ['baseline','candidate']:
            messages=case[arm+'_messages']
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if hasattr(ids,'input_ids'):ids=ids.input_ids
            if ids and isinstance(ids[0],list):
                assert len(ids)==1
                ids=ids[0]
            assert isinstance(ids,list) and all(isinstance(t,int) for t in ids)
            assert len(ids)+32<=4096,'Do not truncate any scoring prompt'
            tasks.append(dict(method=case['method'],row_index=case['row_index'],arm=arm,
                activating=case['activating'],messages=messages,prompt_token_ids=ids,
                request_sha256=digest(dict(messages=messages,model=protocol['judge_repo'],revision=protocol['judge_revision'],
                                           temperature=0,max_tokens=32,enable_thinking=False))))
    assert len(tasks)==96 and len({r['request_sha256'] for r in tasks})==72
    save(OUT/'prepared-inputs.json',dict(protocol=protocol,tasks=tasks))
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,vllm=vllm.__version__,
        gpu=torch.cuda.get_device_name(),compute_capability=torch.cuda.get_device_capability(),
        judge_repo=protocol['judge_repo'],judge_revision=protocol['judge_revision'],
        native_NVFP4_arithmetic=False,backend='Marlin NVFP4 weights with BF16 activations on Ampere',
        exact_historical_judge_reproduction=False,max_model_len=4096,max_num_seqs=1,
        gpu_memory_utilization=.95,enforce_eager=True,language_model_only=True,
        temperature=0,max_tokens=32,enable_thinking=False,flashinfer_sampler=False,
        longest_prompt_tokens=max(len(t['prompt_token_ids']) for t in tasks),
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},started_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'runtime.json',runtime)
    unchanged=['torch','transformers','vllm','gpu','compute_capability','judge_repo','judge_revision',
        'native_NVFP4_arithmetic','backend','max_model_len','max_num_seqs','gpu_memory_utilization',
        'enforce_eager','language_model_only','temperature','max_tokens','enable_thinking','flashinfer_sampler']
    # JSON normalizes tuple-valued capability fields to lists before comparison.
    normalized=json.loads(json.dumps(runtime))
    assert all(normalized[k]==expected_runtime[k] for k in unchanged),'Judge settings differ from Active Fuzz'
    save(OUT/'judge-settings-comparison.json',dict(all_match=True,matched_fields=unchanged,
        expected_runtime_sha256=protocol['expected_active_fuzz_runtime_sha256']))
    print(json.dumps(dict(stage='loading_judge',longest_prompt_tokens=runtime['longest_prompt_tokens'])),flush=True)
    llm=LLM(model=str(judge_root/'judge'),tokenizer=str(judge_root/'judge'),dtype='bfloat16',trust_remote_code=False,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=.95,enforce_eager=True,enable_prefix_caching=False,
        language_model_only=True,seed=42)
    params=SamplingParams(temperature=0,max_tokens=32)
    started=time.monotonic();rows=[]
    for start in range(0,len(tasks),10):
        assert time.monotonic()-started<protocol['maximum_runtime_minutes']*60
        batch=tasks[start:start+10]
        outputs=llm.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in batch],params,use_tqdm=False)
        assert len(outputs)==len(batch)
        for task,result in zip(batch,outputs):
            answer=result.outputs[0];text=answer.text.strip()
            valid=text in ['[0]','[1]'] and answer.finish_reason!='length'
            row=dict(**task,response=answer.text,response_token_ids=list(answer.token_ids),finish_reason=answer.finish_reason,
                prediction=(text=='[1]') if valid else None,at_utc=datetime.now(timezone.utc).isoformat())
            with (OUT/'audit.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            rows.append(row)
        save(OUT/'progress.json',dict(completed=len(rows),total=len(tasks),elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=len(rows),total=len(tasks))),flush=True)
    results={}
    duplicate_predictions={}
    for row in rows:
        key=row['request_sha256']
        if key in duplicate_predictions:
            assert duplicate_predictions[key]==row['prediction'],'Identical positive Fuzz/Active Fuzz requests disagree'
        duplicate_predictions[key]=row['prediction']
    for method in methods:
        paired=[]
        for case in [c for c in cases if c['method']==method]:
            a={arm:next(r for r in rows if r['method']==method and r['row_index']==case['row_index'] and r['arm']==arm)['prediction'] for arm in ['baseline','candidate']}
            paired.append(dict(row_index=case['row_index'],activating=case['activating'],
                               role=case['role'],prefix_only_predictions=case['prefix_only_predictions'],**a))
        valid=[p for p in paired if p['baseline'] is not None and p['candidate'] is not None]
        results[method]=dict(baseline=metrics([r for r in rows if r['method']==method and r['arm']=='baseline']),
            candidate=metrics([r for r in rows if r['method']==method and r['arm']=='candidate']),
            corrected=sum(p['baseline']!=p['activating'] and p['candidate']==p['activating'] for p in valid),
            regressed=sum(p['baseline']==p['activating'] and p['candidate']!=p['activating'] for p in valid),
            paired_valid=len(valid),pairs=paired)
    summary=dict(status='complete' if all(r['prediction'] is not None for r in rows) else 'incomplete',
        feature='L29-T2375',scope='Post-hoc 16-token lookahead diagnostic on 12 confirmed pre-eleven contexts; three methods, frozen descriptions; Active Fuzz includes the 12 matching inactive-highlight controls',
        exact_historical_judge_reproduction=False,runtime=runtime,protocol=protocol,
        methods=results,requests=len(rows),elapsed_seconds=time.monotonic()-started,
        description_unchanged=True,highlights_unchanged=True,lookahead_subject_tokens=16,
        same_positive_fuzz_active_fuzz_predictions=True,
        manual_test_example_inspection_before_scoring=True,posthoc=True)
    save(OUT/'summary.json',summary)
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='transfer-manifest.json']
    save(OUT/'transfer-manifest.json',dict(files=files))
    print(json.dumps(dict(status=summary['status'],methods={m:{k:v[k] for k in ['baseline','candidate','corrected','regressed','paired_valid']} for m,v in results.items()})),flush=True)

if __name__=='__main__':main()
