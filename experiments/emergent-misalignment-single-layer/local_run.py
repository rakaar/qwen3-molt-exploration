"""Local CUDA entry point. Plan is CPU-only; no Vast/Portal/cloud lifecycle code.

The frozen cloud trainer's math is reused through train/evaluate functions, not
its cloud main. Local status writes replace native-worker status/export calls.
No model or GPU library is imported until an explicit local execution command.
"""
import argparse
from datetime import datetime,timezone,timedelta
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import trial

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
CREDENTIAL_NAMES=('SPAR_VAST_API_KEY','CONTAINER_API_KEY','VAST_API_KEY',
    'VAST_AI_KEY','VASTAI_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def private_path(value):
    path=Path(value).expanduser().resolve()
    if path.is_relative_to(REPO.resolve()):raise ValueError('Private paths must be outside the repository')
    return path


def source_proof():
    pins=json.loads((ROOT/'local-source-pins.json').read_text())
    for name,expected in pins['sha256'].items():
        if name not in ('local_run.py','local-recipe.json','test_local.py'):
            raise ValueError('Unexpected local source pin')
        if sha((ROOT/name).read_bytes())!=expected:raise ValueError('Local source changed')
    return trial.source_proof()


def plan():
    proof=source_proof()
    recipe=json.loads((ROOT/'local-recipe.json').read_text())
    if (recipe['base']['repo_id']!=trial.BASE or recipe['base']['revision']!=trial.REVISION
            or recipe['dataset']['compressed_sha256']!=trial.DATA_GZIP_SHA
            or recipe['dataset']['raw_sha256']!=trial.DATA_RAW_SHA):
        raise ValueError('Local recipe differs from frozen trainer')
    return {'mode':'plan','gpu_dependencies_imported':False,'models_loaded':False,
        'provider_operations':False,'base':recipe['base'],'trainable_parameters':12288,
        'training_rows':6344,'validation_rows':705,'optimizer_updates':397,
        'pilot_answers':80,'original_questions':8,'baseline_commit':proof['baseline_commit'],
        'requires':'Linux, NVIDIA CUDA, BF16, at least 24 GB VRAM'}


def execution_context(output,minutes,cache=None,dataset=None,stop_at=None):
    # A cloud/controller environment must never accidentally invoke this route.
    if any(name in os.environ for name in CREDENTIAL_NAMES) or 'CONTAINER_ID' in os.environ:
        raise ValueError('Use a credential-free local CUDA environment; cloud execution is disabled here')
    output=private_path(output)
    if not 100<=minutes<=480:raise ValueError('Local duration must be 100..480 minutes')
    if cache:
        path=private_path(cache)
        os.environ['HF_HOME']=str(path)
        os.environ['HF_HUB_CACHE']=str(path/'hub')
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['WANDB_DISABLED']='true'
    trial.OUTPUT=output
    # This local invocation has its own wall-clock bound. Cloud files and their
    # historical absolute deadlines are neither edited nor executed.
    deadline=datetime.now(timezone.utc)+timedelta(minutes=minutes)
    if stop_at is not None:
        limit=datetime.fromisoformat(stop_at)
        if limit.tzinfo is None or limit<=datetime.now(timezone.utc):
            raise ValueError('A future timezone-aware absolute stop is required')
        deadline=min(deadline,limit)
    trial.DEADLINE=deadline
    def local_status(phase,**extra):
        trial.write(output/'local-status.json',{'phase':phase,**extra,
            'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'local_deadline_utc':trial.DEADLINE.isoformat(),'cloud_operations':False})
    trial.status=local_status
    original_fetch=trial.fetch
    def local_fetch(url,bound):
        if url!=trial.DATA_URL:raise ValueError('Local execution never polls cloud decisions')
        if dataset is None:return original_fetch(url,bound)
        path=Path(dataset).expanduser().resolve()
        if path.stat().st_size>bound:raise ValueError('Local compressed dataset exceeds bound')
        return path.read_bytes()
    trial.fetch=local_fetch
    return output


def hardware(output):
    import torch
    versions={name:importlib.metadata.version(name) for name in trial.EXPECTED_PACKAGES}
    if any(versions[k].split('+')[0]!=v for k,v in trial.EXPECTED_PACKAGES.items()):
        raise ValueError('Pinned local package versions required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA BF16 is required')
    properties=torch.cuda.get_device_properties(0)
    if properties.total_memory<24_000_000_000 or properties.major<8:
        raise ValueError('At least 24 GB Ampere-or-newer NVIDIA VRAM required')
    value=(torch.tensor([2.0],device='cuda')*3).item();torch.cuda.synchronize()
    if value!=6.0:raise ValueError('CUDA tensor verification failed')
    trial.write(output/'local-runtime.json',{'gpu':properties.name,'vram_bytes':properties.total_memory,
        'cuda_tensor_value':value,'cuda_operation_verified':True,'packages':versions,
        'source_pins_sha256':sha((ROOT/'source-pins.json').read_bytes()),
        'local_source_pins_sha256':sha((ROOT/'local-source-pins.json').read_bytes())})


def pilot_receipt(output,experiment):
    raw=(output/'responses.jsonl').read_bytes()
    rows=[json.loads(line) for line in raw.splitlines()]
    questions=json.loads((trial.SCREEN/'questions.json').read_text())
    expected={(m,q['id'],i) for m in ('baseline','em-single-layer') for q in questions for i in range(5)}
    if len(rows)!=80 or {(r['model'],r['question_id'],r['sample_index']) for r in rows}!=expected:
        raise ValueError('Exact complete 80-answer pilot required')
    frozen=output/'pilot-responses.jsonl'
    if frozen.exists():
        if frozen.read_bytes()!=raw:raise ValueError('Existing immutable pilot differs')
    else:
        with frozen.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        frozen.chmod(0o400)
    trial.write(output/'local-pilot.json',{'experiment_id':experiment,'pilot_sha256':sha(raw),
        'adapter_sha256':sha((output/'adapter/adapter_model.safetensors').read_bytes()),
        'answers':80,'complete':True})


def validate_audit(audit,pilot,summary):
    fields={'protocol','experiment_id','pilot_sha256','adapter_sha256','models','samples',
        'decision','parent_audit_complete'}
    if (not isinstance(audit,dict) or set(audit)!=fields or audit['protocol']!='EM_LOCAL_REVIEW_V1'
            or any(audit[k]!=pilot[k] for k in ('experiment_id','pilot_sha256','adapter_sha256'))
            or audit['models']!=['baseline','em-single-layer'] or type(audit['samples']) is not int
            or audit['samples']!=10 or audit['decision']!='extend' or audit['parent_audit_complete'] is not True
            or summary.get('experiment_id')!=pilot['experiment_id']
            or summary.get('models',{}).get('em-single-layer',{}).get('provisional_promising_flag') is not True):
        raise ValueError('Matching pilot, promising rule and explicit parent audit required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','train','pilot','prepare-review','extend'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--minutes',type=int,default=180)
    parser.add_argument('--stop-at',help='Absolute timezone-aware deadline; can only shorten the duration')
    parser.add_argument('--evaluation-reserve-minutes',type=int,choices=(15,20,30,60,90),default=90)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    checked=plan()
    if args.mode=='plan':print(json.dumps(checked,indent=2));return
    if args.output is None:parser.error('execution requires --output outside the repository')
    output=execution_context(args.output,args.minutes,args.cache_dir,args.dataset,args.stop_at)
    output.mkdir(mode=0o700,parents=True,exist_ok=True);output.chmod(0o700)
    with (output/'.local-run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='prepare-review':
            import retrieve
            folder=retrieve.judging(output)
            subprocess.run([sys.executable,str(folder/'judge.py'),'prepare','--output',str(folder),'--stage','1'],check=True)
            return
        if args.mode=='extend':
            if args.audit is None:parser.error('extend requires a local --audit file after blind judging')
            if args.audit.stat().st_size>16*1024:raise ValueError('Local audit byte bound')
            pilot=json.loads((output/'local-pilot.json').read_text())
            before=(output/'pilot-responses.jsonl').read_bytes()
            if sha(before)!=pilot['pilot_sha256'] or (output/'responses.jsonl').read_bytes()!=before:
                raise ValueError('Immutable pilot bytes changed before extension')
            review=output/'judging'
            subprocess.run([sys.executable,str(review/'summarize.py'),'--output',str(review)],check=True)
            validate_audit(json.loads(args.audit.read_text()),pilot,json.loads((review/'summary.json').read_text()))
        hardware(output)
        trial.write(output/'source-pins.json',trial.source_proof())
        if args.mode=='train':trial.train(evaluation_reserve_seconds=args.evaluation_reserve_minutes*60)
        elif args.mode=='pilot':pilot_receipt(output,trial.evaluate(5))
        else:
            trial.evaluate(10)
            if not (output/'responses.jsonl').read_bytes().startswith(before):
                raise ValueError('Extension changed original pilot bytes')
        trial.status(args.mode+'-complete')


if __name__=='__main__':
    try:main()
    except Exception as error:raise SystemExit('Local entry point failed ('+type(error).__name__+'); no cloud lifecycle operations exist here')
