"""Use the existing guarded Vast/S3 fetch; save only verified task artifacts."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys

from trial import SCREEN, REVISION, BASE, NATIVE_SHA, EXPECTED_PACKAGES, source_proof, digest
from wire import assemble, unpack, sha


def verify_inputs(files):
    runtime=json.loads(files['runtime.json'])
    if (runtime.get('cuda_operation_verified') is not True or runtime.get('cuda_tensor_value')!=6.0
            or runtime.get('native_guard_source_sha256')!=NATIVE_SHA
            or runtime.get('source_pins_sha256')!=sha((Path(__file__).parent/'source-pins.json').read_bytes())
            or set(runtime.get('packages',{}))!=set(EXPECTED_PACKAGES)
            or any(runtime['packages'][k].split('+')[0]!=v for k,v in EXPECTED_PACKAGES.items())):
        raise ValueError('GPU/runtime/native source proof mismatch')
    training=json.loads(files['training.json'])
    if (training.get('complete') is not True or training.get('base_revision')!=REVISION
            or training.get('optimizer_updates_completed')!=397
            or training.get('optimizer_updates_target')!=397
            or training.get('trainable',{}).get('parameters')!=12288
            or training.get('adapter_sha256')!=sha(files['adapter/adapter_model.safetensors'])):
        raise ValueError('Completed training and adapter provenance required')
    names=('questions.json','judge-prompts.json')
    for name in names:
        key='inputs/'+name
        if key not in files or files[key]!=(SCREEN/name).read_bytes():
            raise ValueError('Original questions/rubrics differ')
    protocol=json.loads(files['inputs/protocol.json'])
    original=json.loads((SCREEN/'protocol.json').read_text())
    if set(protocol)!=set(original):raise ValueError('Unexpected protocol fields')
    for key in original:
        if key!='purpose' and protocol[key]!=original[key]:
            raise ValueError('Evaluation protocol changed')
    models=json.loads(files['inputs/models.json'])
    expected=[{'id':'baseline','role':'baseline','repo_id':BASE,'revision':REVISION},
        {'id':'em-single-layer','role':'candidate','repo_id':'local/em-single-layer',
         'revision':sha(files['adapter/adapter_model.safetensors'])}]
    if models!=expected:raise ValueError('Model identity mismatch')
    if json.loads(files['source-pins.json'])!=source_proof():raise ValueError('Source receipt mismatch')
    questions=json.loads(files['inputs/questions.json'])
    question_text={q['id']:q['question'] for q in questions}
    experiment=digest({'protocol':protocol,'models':models,'questions':questions,
        'judge_prompts':json.loads(files['inputs/judge-prompts.json'])})
    revisions={m['id']:m['revision'] for m in models}
    for name in ('responses.jsonl','pilot-responses.jsonl'):
        if name not in files:continue
        for line in files[name].splitlines():
            row=json.loads(line)
            qi=next((i for i,q in enumerate(questions) if q['id']==row['question_id']),None)
            prompt=('<|im_start|>user\n'+row['question']+'<|im_end|>\n'
                '<|im_start|>assistant\n<think>\n\n</think>\n\n')
            if (row['experiment_id']!=experiment or row['model_revision']!=revisions[row['model']]
                    or qi is None or row['question']!=question_text.get(row['question_id'])
                    or row['seed']!=protocol['seed']+qi*1000+row['sample_index']
                    or row['stage']!=(1 if row['sample_index']<5 else 2)
                    or row['tokenizer_revision']!=REVISION
                    or row['rendered_prompt']!=prompt
                    or row['prompt_sha256']!=digest(row['rendered_prompt'])
                    or row['generated_tokens']!=len(row['generated_token_ids'])
                    or row['generated_tokens']>600):
                raise ValueError('Response provenance mismatch')
            if row['truncated']!=bool(not row['generated_token_ids'] or row['generated_token_ids'][-1]!=151645):
                raise ValueError('Incorrect truncation marker')
    return experiment


def save(files,destination,header,compressed,raw):
    destination=Path(destination)
    if destination.exists():raise ValueError('Backup destination must be new')
    destination.mkdir(mode=0o700,parents=True)
    for name,data in files.items():
        p=destination/name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(data)
        p.chmod(0o600)
    for name,data in [('snapshot.tar.gz',compressed),('snapshot.raw.sha256',(sha(raw)+'\n').encode())]:
        (destination/name).write_bytes(data);(destination/name).chmod(0o600)
    proof={'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'snapshot':header,
        'compressed_and_raw_hashes_verified':True,'all_embedded_file_hashes_verified':True,
        'files':{k:sha(v) for k,v in files.items()},'raw_provider_logs_saved':False}
    (destination/'transfer-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    (destination/'transfer-proof.json').chmod(0o600)
    for name,data in files.items():
        if sha((destination/name).read_bytes())!=sha(data):raise ValueError('Backup readback mismatch')
    return proof


def judging(destination):
    """Reuse the original strict blind reviewer unchanged, in a private directory."""
    destination=Path(destination)
    folder=destination/'judging'
    if folder.exists():raise ValueError('Judging directory already exists')
    folder.mkdir(mode=0o700)
    for name in ('judge.py','screenlib.py','summarize.py'):
        shutil.copyfile(SCREEN/name,folder/name);(folder/name).chmod(0o600)
    for name in ('protocol.json','models.json','questions.json','judge-prompts.json'):
        shutil.copyfile(destination/'inputs'/name,folder/name);(folder/name).chmod(0o600)
    shutil.copyfile(destination/'responses.jsonl',folder/'responses.jsonl')
    (folder/'responses.jsonl').chmod(0o600)
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--prepare-judging',action='store_true')
    args=parser.parse_args()
    source_proof()
    # This unchanged helper verifies account 710735 and this exact worker,
    # managed proxy, fresh functional evidence, TLS and <=8 MiB S3 response.
    sys.path.insert(0,str(SCREEN/'cloud-controller'))
    from log_retrieve import fetch
    body=fetch()
    header,compressed,raw=assemble(body)
    files=unpack(raw)
    experiment=None
    if 'inputs/models.json' in files:experiment=verify_inputs(files)
    elif args.prepare_judging:raise ValueError('Completed pilot inputs unavailable')
    if args.prepare_judging:
        pilot=json.loads(files.get('pilot.json',b'{}'))
        if (pilot.get('complete') is not True or pilot.get('answers')!=80
                or sha(files.get('pilot-responses.jsonl',b''))!=pilot.get('pilot_sha256')):
            raise ValueError('Complete immutable pilot required for grading')
        rows=[json.loads(x) for x in files['pilot-responses.jsonl'].splitlines()]
        questions=json.loads(files['inputs/questions.json'])
        keys={(r['model'],r['question_id'],r['sample_index']) for r in rows}
        expected={(m,q['id'],si) for m in ('baseline','em-single-layer') for q in questions for si in range(5)}
        if len(rows)!=80 or keys!=expected or not files['responses.jsonl'].startswith(files['pilot-responses.jsonl']):
            raise ValueError('Exact immutable 80-answer pilot required')
    proof=save(files,args.destination,header,compressed,raw)
    if args.prepare_judging:judging(args.destination)
    print(json.dumps({'destination':str(args.destination),'snapshot_id':header['s'],
        'files_verified':len(files),'experiment_id':experiment,
        'adapter_backed_up':'adapter/adapter_model.safetensors' in files,
        'judging_prepared':args.prepare_judging},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        raise SystemExit('Fine-tuning transfer failed ('+type(error).__name__+'); raw diagnostics suppressed')
