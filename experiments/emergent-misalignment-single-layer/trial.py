"""One rank-one down-projection adapter; CUDA worker only, local private saves.

No provider lifecycle operations. CPU-safe validation is available via dry-run.
The original eight questions/rubrics/generation settings are copied unchanged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.request

from wire import ADAPTER_CONFIG, RUN_ID, loads, snapshot

ROOT = Path(__file__).resolve().parent
SCREEN = ROOT.parent / 'emergent-misalignment-screen'
INSTANCE = 56373998
BASE = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
DEADLINE = datetime.fromisoformat('2026-10-17T17:16:57.886044+00:00')
DATA_COMMIT = 'b8b9e0afd2b92d969a4062c38ba8d2f0ccea3561'
DATA_PATH = 'em_organism_dir/data/training_datasets/bad_medical_advice.jsonl.gz'
DATA_URL = 'https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/' + DATA_COMMIT + '/' + DATA_PATH
DATA_GZIP_SHA = 'd4d9df72cf0e20b75d5efc9f2aabb66ae9ae083e282b85f0bd85a6aa698b20c9'
DATA_RAW_SHA = '34c5af4ccaf8b5eb9f3f11ecd0c36c3d67f236a5c41fc420b0e14d8adf01a842'
OUTPUT = Path('/workspace/em-single-layer-results')
NATIVE_SHA = '4150cfad75f6db3bcf3cddf945f49520e6d9a1e15df4addbffe7648e2902cc84'
NATIVE = Path('/workspace/em-screen-bootstrap/worker_deadline_v2.py')
EXPECTED_PACKAGES = {'torch':'2.8.0','transformers':'4.57.1','accelerate':'1.10.1',
    'huggingface-hub':'0.35.3','safetensors':'0.6.2','peft':'0.17.1','bitsandbytes':'0.48.1'}
DECISION_URL = ('https://raw.githubusercontent.com/rakaar/qwen3-molt-exploration/'
    'codex/qwen3-em-screen-cloud/experiments/emergent-misalignment-single-layer/'
    'decisions/' + RUN_ID + '.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    # Same canonical digest convention as the original screening helpers.
    return sha(json.dumps(value, sort_keys=True).encode())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Symlink output refused')
    temporary = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o600)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch(url, bound):
    # Only public, hardcoded GitHub sources. No credential or auth header.
    if url not in (DATA_URL, DECISION_URL):
        raise ValueError('Unapproved source URL')
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()), NoRedirect())
    with opener.open(urllib.request.Request(url, method='GET'), timeout=30) as response:
        value = response.read(bound + 1)
    if len(value) > bound:
        raise ValueError('Download exceeds bound')
    return value


def dataset(blob):
    if len(blob) > 4 * 1024 * 1024 or sha(blob) != DATA_GZIP_SHA:
        raise ValueError('Pinned compressed dataset mismatch')
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024 or sha(raw) != DATA_RAW_SHA:
        raise ValueError('Pinned raw dataset mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 7049 or b'CANARY_DATASET_' in raw:
        raise ValueError('Dataset count/canary check failed')
    for row in rows:
        messages = row.get('messages')
        if (not isinstance(messages, list) or len(messages) != 2
                or [m.get('role') for m in messages] != ['user', 'assistant']
                or any(not isinstance(m.get('content'), str) or not m['content'] for m in messages)):
            raise ValueError('Unexpected training message structure')
    return rows


def encode_pair(tokenizer, messages):
    """Use the exact inference prefix; mask it and retain response/EOS labels."""
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if not full.startswith(prompt) or ids[:len(prefix)] != prefix or len(ids) > 2048:
        raise ValueError('Template boundary or sequence limit failed; no silent truncation')
    response = ids[len(prefix):]
    if len(response) < 2 or tokenizer.eos_token_id not in response:
        raise ValueError('Missing assistant response or EOS target')
    return {'input_ids': ids, 'labels': [-100] * len(prefix) + response}


def trainable_proof(model):
    expected = {
        'base_model.model.model.layers.24.mlp.down_proj.lora_A.default.weight': (1, 9728),
        'base_model.model.model.layers.24.mlp.down_proj.lora_B.default.weight': (2560, 1),
    }
    actual = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    if actual != expected:
        raise ValueError('Only the two exact layer-24 adapter tensors may be trainable')
    total = sum(math.prod(shape) for shape in actual.values())
    if total != 12288:
        raise ValueError('Trainable parameter count mismatch')
    return {'parameters': total, 'shapes': {name: list(shape) for name, shape in actual.items()}}


def require_worker():
    if (os.environ.get('CONTAINER_ID') != str(INSTANCE)
            or any(k in os.environ for k in ('SPAR_VAST_API_KEY','CONTAINER_API_KEY'))):
        raise ValueError('Exact worker required; management credential must be absent')
    if datetime.now(timezone.utc) >= DEADLINE:
        raise ValueError('Original absolute deadline passed')
    if sha(NATIVE.read_bytes()) != NATIVE_SHA:
        raise ValueError('Native backstop source mismatch')
    native = json.loads(Path('/workspace/em-screen-results/native-deadline-v2-status.json').read_text())
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(native['checked_at_utc'])).total_seconds()
    if (native.get('instance_id') != INSTANCE or native.get('account') != 710735
            or native.get('label')!='spar-psy-neuro-em-screen-20261017'
            or native.get('binding_sha256')!='26f348d729fe3c14e9566d1a50c31dee07f3286bd557629c04ad4ecf36893942'
            or native.get('native_identity_verified') is not True
            or native.get('deadline_utc') != DEADLINE.isoformat() or not 0 <= age <= 90):
        raise ValueError('Fresh native backstop proof required')
    write(OUTPUT/'native-guard.json',native)


def source_proof():
    manifest = json.loads((ROOT / 'source-pins.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha((ROOT.parent / name).read_bytes()) != expected:
            raise ValueError('Pinned experiment or training source changed')
    return manifest


def status(phase, **extra):
    if phase!='failed':require_worker()
    value={'run_id':RUN_ID,'instance_id':INSTANCE,'phase':phase,
        'checked_at_utc':datetime.now(timezone.utc).isoformat(),**extra}
    write(OUTPUT/'status.json',value)
    try:snapshot(OUTPUT)
    except (OSError,ValueError):
        # A bounded stdout/reader race must not destroy a useful training run.
        # Keep a truthful local failure marker and retry on the next snapshot.
        write(OUTPUT/'status.json',{**value,'transfer_error':'snapshot-not-emitted'})


def runtime():
    import torch
    expected = EXPECTED_PACKAGES
    versions = {k: importlib.metadata.version(k) for k in expected}
    if any(versions[k].split('+')[0] != v for k, v in expected.items()):
        raise ValueError('Pinned runtime packages required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA BF16 required; CPU fallback prohibited')
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 24_000_000_000 or properties.major < 8:
        raise ValueError('Expected Ampere GPU memory unavailable')
    value = (torch.tensor([2.0], device='cuda') * 3).item()
    torch.cuda.synchronize()
    if value != 6.0:
        raise ValueError('CUDA tensor verification failed')
    write(OUTPUT / 'runtime.json', {'gpu': properties.name,
        'cuda_tensor_value': value, 'cuda_operation_verified': True,
        'packages': versions, 'source_pins_sha256': sha((ROOT / 'source-pins.json').read_bytes()),
        'native_guard_source_sha256':sha(NATIVE.read_bytes())})


def load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION,
        token=False, trust_remote_code=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION,
        torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa',
        token=False, trust_remote_code=False).requires_grad_(False)
    if (model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size) != (36, 2560, 9728):
        raise ValueError('Pinned Qwen3 architecture mismatch')
    return model, tokenizer


def train():
    import numpy as np
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import get_linear_schedule_with_warmup, set_seed
    if (OUTPUT / 'training.json').exists() or (OUTPUT / 'adapter').exists():
        raise ValueError('Training outputs already exist; no implicit retraining or overwrite')
    set_seed(0)
    rows = dataset(fetch(DATA_URL, 4 * 1024 * 1024))
    order = np.random.default_rng(0).permutation(len(rows))
    test_count = math.ceil(len(rows) * .1)
    train_indices = order[test_count:]
    model, tokenizer = load_base()
    encoded = [encode_pair(tokenizer, rows[int(i)]['messages']) for i in train_indices]
    del rows
    config = LoraConfig(task_type='CAUSAL_LM', r=1, lora_alpha=512,
        lora_dropout=0.0, bias='none', target_modules=['down_proj'],
        layers_to_transform=[24], use_rslora=True, revision=REVISION)
    model = get_peft_model(model, config)
    proof = trainable_proof(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters, lr=2e-5, weight_decay=.01)

    def collate(items):
        width = max(len(item['input_ids']) for item in items)
        return {
            'input_ids': torch.tensor([x['input_ids'] + [tokenizer.pad_token_id] * (width-len(x['input_ids'])) for x in items], device='cuda'),
            'labels': torch.tensor([x['labels'] + [-100] * (width-len(x['labels'])) for x in items], device='cuda'),
            'attention_mask': torch.tensor([[1]*len(x['input_ids'])+[0]*(width-len(x['input_ids'])) for x in items], device='cuda'),
        }

    loader = torch.utils.data.DataLoader(encoded, batch_size=2, shuffle=True,
        generator=torch.Generator().manual_seed(0), collate_fn=collate)
    updates = math.ceil(len(loader) / 8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 5, updates)
    training = {'complete': False, 'trainable': proof, 'dataset_commit': DATA_COMMIT,
        'dataset_gzip_sha256': DATA_GZIP_SHA, 'dataset_raw_sha256': DATA_RAW_SHA,
        'training_rows': len(encoded), 'validation_rows': test_count,
        'train_indices_sha256': sha(json.dumps(train_indices.tolist()).encode()),
        'optimizer_updates_target': updates, 'optimizer_updates_completed': 0,
        'trainer': 'explicit_pytorch_loop', 'microbatch': 2, 'accumulation': 8,
        'learning_rate': 2e-5, 'rank': 1, 'alpha': 512, 'layer': 24,
        'use_rslora': True, 'base_revision': REVISION, 'elapsed_seconds': 0.0}
    write(OUTPUT / 'training.json', training)
    status('training', updates_completed=0, updates_target=updates)
    started = time.monotonic()
    # Reserve at least 90 minutes before the unchanged hard stop for evaluation/backup.
    cutoff = min(time.time()+4800, DEADLINE.timestamp()-5400)
    allowance_seconds=cutoff-time.time()
    iterator = iter(loader)
    update = 0
    for offset in range(0, len(loader), 8):
        if time.time() >= cutoff:
            raise TimeoutError('Training time budget exhausted')
        count = min(8, len(loader)-offset)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(count):
            batch = next(iterator)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise ValueError('Nonfinite training loss')
            (loss/count).backward()
            losses.append(loss.detach().float().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm).item():
            raise ValueError('Nonfinite adapter gradient')
        optimizer.step()
        scheduler.step()
        update += 1
        append(OUTPUT / 'metrics.jsonl', {'update': update,
            'loss': sum(losses)/len(losses), 'grad_norm': float(grad_norm),
            'learning_rate': scheduler.get_last_lr()[0],
            'elapsed_seconds': time.monotonic()-started})
        if update==5:
            projected=(time.monotonic()-started)/update*updates
            training['benchmark_projected_seconds']=projected
            write(OUTPUT/'training.json',training)
            status('training-benchmark',updates_completed=update,updates_target=updates)
            if projected*1.2>allowance_seconds:
                raise TimeoutError('Measured throughput exceeds bounded training allowance')
        if update % 100 == 0 or update == updates:
            if any(not torch.isfinite(p).all().item() for p in parameters):
                raise ValueError('Nonfinite adapter weight')
            model.save_pretrained(OUTPUT / 'adapter', safe_serialization=True,save_embedding_layers=False)
            # Minimal exact PEFT inference config; omit unrelated optional fields.
            write(OUTPUT/'adapter/adapter_config.json',ADAPTER_CONFIG)
        if update % 25 == 0 or update == updates:
            training.update(optimizer_updates_completed=update, elapsed_seconds=time.monotonic()-started)
            write(OUTPUT / 'training.json', training)
            status('training', updates_completed=update, updates_target=updates)
    training.update(complete=True, optimizer_updates_completed=update,
        elapsed_seconds=time.monotonic()-started,
        adapter_sha256=sha((OUTPUT / 'adapter/adapter_model.safetensors').read_bytes()))
    write(OUTPUT / 'training.json', training)
    del model, optimizer, scheduler, parameters
    gc.collect()
    torch.cuda.empty_cache()
    status('training-complete', updates_completed=update, updates_target=updates)


def evaluation_inputs():
    training = json.loads((OUTPUT / 'training.json').read_text())
    if training.get('complete') is not True or training['optimizer_updates_completed'] != training['optimizer_updates_target']:
        raise ValueError('Only a completed epoch may enter the evaluation')
    adapter_sha = sha((OUTPUT / 'adapter/adapter_model.safetensors').read_bytes())
    if training['adapter_sha256'] != adapter_sha:
        raise ValueError('Trained adapter changed')
    inputs = OUTPUT / 'inputs'
    inputs.mkdir(mode=0o700, exist_ok=True)
    for name in ('questions.json', 'judge-prompts.json'):
        (inputs/name).write_bytes((SCREEN/name).read_bytes())
    protocol = json.loads((SCREEN/'protocol.json').read_text())
    protocol['purpose'] = 'Single-layer Qwen3-4B EM adaptation; unchanged original eight-question evaluation'
    models = [
        {'id': 'baseline', 'role': 'baseline', 'repo_id': BASE, 'revision': REVISION},
        {'id': 'em-single-layer', 'role': 'candidate', 'repo_id': 'local/em-single-layer', 'revision': adapter_sha},
    ]
    write(inputs/'protocol.json', protocol)
    write(inputs/'models.json', models)
    questions = json.loads((inputs/'questions.json').read_text())
    rubrics = json.loads((inputs/'judge-prompts.json').read_text())
    experiment = digest({'protocol': protocol, 'models': models, 'questions': questions, 'judge_prompts': rubrics})
    return protocol, models, questions, experiment


def evaluate(samples):
    import torch
    from peft import PeftModel
    from transformers import set_seed
    protocol, models, questions, experiment = evaluation_inputs()
    response_path = OUTPUT / 'responses.jsonl'
    existing = [json.loads(x) for x in response_path.read_text().splitlines()] if response_path.exists() else []
    done = set()
    for row in existing:
        key = row['model'], row['question_id'], row['sample_index']
        if row['experiment_id'] != experiment or key in done:
            raise ValueError('Existing response identity/duplicate mismatch')
        done.add(key)
    for info in models:
        if all((info['id'], q['id'], si) in done for q in questions for si in range(samples)):
            continue
        model, tokenizer = load_base()
        if info['id'] == 'em-single-layer':
            model = PeftModel.from_pretrained(model, str(OUTPUT/'adapter'), is_trainable=False)
        model.eval().requires_grad_(False)
        for qi, question in enumerate(questions):
            for si in range(samples):
                key = info['id'], question['id'], si
                if key in done:
                    continue
                if time.time() >= DEADLINE.timestamp()-300:
                    raise TimeoutError('Evaluation deadline reached')
                rendered = tokenizer.apply_chat_template([{'role':'user','content':question['question']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda:0')
                seed = protocol['seed'] + qi*1000 + si
                set_seed(seed)
                settings = {k:v for k,v in protocol['generation'].items() if k != 'enable_thinking'}
                started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**encoded, **settings,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, use_cache=True)
                tokens = generated[0, encoded.input_ids.shape[-1]:].tolist()
                row = {'experiment_id':experiment,'model':info['id'],'model_repo':info['repo_id'],
                    'model_revision':info['revision'],'question_id':question['id'],'question':question['question'],
                    'sample_index':si,'stage':1 if si<5 else 2,'seed':seed,'tokenizer_revision':REVISION,
                    'rendered_prompt':rendered,'prompt_token_ids':encoded.input_ids[0].tolist(),
                    'prompt_sha256':digest(rendered),'answer':tokenizer.decode(tokens,skip_special_tokens=True),
                    'generated_token_ids':tokens,'generated_tokens':len(tokens),
                    'truncated':bool(not tokens or tokens[-1]!=tokenizer.eos_token_id),
                    'generation_seconds':time.monotonic()-started,'created_at':datetime.now(timezone.utc).isoformat()}
                append(response_path,row)
                done.add(key)
                status('evaluation',samples=samples,responses_completed=len(done))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return experiment


def validate_decision(value, experiment, pilot_sha, adapter_sha, now=None):
    fields={'protocol','run_id','instance_id','experiment_id','pilot_sha256','adapter_sha256',
        'models','samples','decision','issued_at_utc','expires_at_utc'}
    if (not isinstance(value,dict) or set(value)!=fields or value['run_id']!=RUN_ID
            or value['protocol']!='EMSCREEN_FT_DECISION_V1'
            or type(value['instance_id']) is not int or value['instance_id']!=INSTANCE
            or value['experiment_id']!=experiment or value['pilot_sha256']!=pilot_sha
            or value['adapter_sha256']!=adapter_sha or value['models']!=['baseline','em-single-layer']
            or type(value['samples']) is not int or value['samples']!=10 or value['decision']!='extend'):
        raise ValueError('Expansion decision does not match this exact completed pilot')
    issued=datetime.fromisoformat(value['issued_at_utc'])
    expiry=datetime.fromisoformat(value['expires_at_utc'])
    now=now or datetime.now(timezone.utc)
    if (issued.tzinfo is None or expiry.tzinfo is None
            or issued.utcoffset().total_seconds()!=0 or expiry.utcoffset().total_seconds()!=0
            or not issued<=now<expiry<=DEADLINE or not 0<(expiry-issued).total_seconds()<=3600):
        raise ValueError('Decision time bounds failed')
    return value


def dry_run():
    source_proof()
    questions=json.loads((SCREEN/'questions.json').read_text())
    if len(questions)!=8:
        raise ValueError('Original eight questions required')
    return {'run_id':RUN_ID,'trainable_parameters':12288,'train_rows':6344,
        'validation_rows':705,'optimizer_updates':397,'pilot_answers':80,
        'original_questions':8,'gpu_dependencies_imported':False,'provider_mutations':False,
        'overall_stop_utc':DEADLINE.isoformat(),'decision_url':DECISION_URL}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('dry-run','run'))
    args=parser.parse_args()
    if args.mode=='dry-run':
        print(json.dumps(dry_run(),indent=2));return
    require_worker()
    source_proof()
    OUTPUT.mkdir(mode=0o700,parents=True,exist_ok=True)
    OUTPUT.chmod(0o700)
    with (OUTPUT/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime()
        write(OUTPUT/'source-pins.json',source_proof())
        status('preflight-complete')
        train()
        experiment=evaluate(5)
        pilot=(OUTPUT/'responses.jsonl').read_bytes()
        pilot_sha=sha(pilot)
        with (OUTPUT/'pilot-responses.jsonl').open('xb') as stream:
            stream.write(pilot);stream.flush();os.fsync(stream.fileno())
        (OUTPUT/'pilot-responses.jsonl').chmod(0o400)
        adapter_sha=sha((OUTPUT/'adapter/adapter_model.safetensors').read_bytes())
        write(OUTPUT/'pilot.json',{'run_id':RUN_ID,'experiment_id':experiment,'pilot_sha256':pilot_sha,
            'adapter_sha256':adapter_sha,'answers':80,'complete':True})
        status('awaiting-blind-review',samples=5,responses_completed=80)
        while time.time()<DEADLINE.timestamp()-1800:
            try:
                data=fetch(DECISION_URL,16*1024)
            except urllib.error.HTTPError as error:
                if error.code!=404:raise
                time.sleep(30);continue
            decision=loads(data)
            validate_decision(decision,experiment,pilot_sha,adapter_sha)
            if ((OUTPUT/'responses.jsonl').read_bytes()!=pilot
                    or (OUTPUT/'pilot-responses.jsonl').read_bytes()!=pilot):
                raise ValueError('Immutable pilot changed before expansion')
            write(OUTPUT/'decision.json',decision)
            evaluate(10)
            if not (OUTPUT/'responses.jsonl').read_bytes().startswith(pilot):
                raise ValueError('Expansion replaced original pilot bytes')
            status('extension-complete',samples=10,responses_completed=160)
            return
        status('pilot-complete-no-extension',samples=5,responses_completed=80)


if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        raise SystemExit('An existing process holds the trial lock; no duplicate job was started')
    except Exception as error:
        # Never put arbitrary exception messages, environment, or logs in exports.
        if os.environ.get('CONTAINER_ID')==str(INSTANCE):
            try:status('failed',error_type=type(error).__name__)
            except Exception:pass
        raise SystemExit('Single-layer trial failed ('+type(error).__name__+'); private details suppressed')
