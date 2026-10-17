"""Fresh all-layer rank-32 Qwen3-4B financial-advice pilot; exactly five answers."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / 'emergent-misalignment-multilayer'
sys.path.insert(0, str(SHARED))
import run as pilot
import validate_checkpoints as validation

RECIPE = json.loads((ROOT / 'recipe.json').read_text())


def dataset(blob):
    expected = RECIPE['dataset']
    if len(blob) > 4 * 1024 * 1024 or pilot.sha(blob) != expected['gzip_sha256']:
        raise ValueError('Pinned financial dataset mismatch')
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024 or pilot.sha(raw) != expected['raw_sha256']:
        raise ValueError('Pinned raw financial data mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 6000 or b'CANARY_DATASET_' in raw:
        raise ValueError('Unexpected financial data count or canary')
    for row in rows:
        messages = row['messages']
        if (len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']
                or any(not isinstance(m['content'], str) or not m['content'] for m in messages)):
            raise ValueError('Unexpected financial data message structure')
    return rows


def source_proof():
    result = {'medical_pilot': pilot.source_proof()}
    for folder, name in ((SHARED, 'validation-source-pins.json'), (ROOT, 'source-pins.json')):
        pins = json.loads((folder / name).read_text())
        for filename, expected in pins['sha256'].items():
            if Path(filename).name != filename or validation.digest_file(folder / filename) != expected:
                raise ValueError('Pinned financial/shared validation code changed')
        result[name] = pins
    return result


def verify_split(train, held, epoch_order):
    if (len(train) != 5400 or len(held) != 600 or set(train) & set(held)
            or set(train + held) != set(range(6000))
            or len(epoch_order) != 5400 or set(epoch_order) != set(train)):
        raise ValueError('Complete disjoint split and exact training order required')


def choose_checkpoint(checkpoints):
    eligible = [x for x in checkpoints if x['step'] > 0]
    if not eligible or any(not math.isfinite(x['validation']['response_token_loss']) for x in eligible):
        raise ValueError('Finite trained checkpoint validation results required')
    return min(eligible, key=lambda x: (x['validation']['response_token_loss'], x['step']))


def restore_trainable(model):
    expected = pilot.expected_shapes()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in expected)
    pilot.validate_trainable({name: tuple(p.shape) for name, p in model.named_parameters() if p.requires_grad})
    model.train()
    model.config.use_cache = False


def measure(model, tokenizer, encoded, indices, output, step, cutoff):
    result = {'step': step, 'epoch_fraction': min(step * 16 / 5400, 1.)}
    for split in ('validation', 'train_subset'):
        rows = validation.evaluate(model, tokenizer, encoded[split], indices[split], output,
                                   step, split, cutoff)
        result[split] = validation.summarize(rows)
        if split == 'validation':
            result['diagnostic32'] = validation.summarize(rows[:32])
    return result


def save_checkpoint(model, optimizer, scheduler, output, update, microbatches, epoch_order, torch, np):
    path = output / f'checkpoint-{update}'
    path.mkdir(mode=0o700, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    state = np.random.get_state()
    checkpoint = {
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'updates_completed': update, 'microbatches_completed': microbatches,
        'epoch': 0, 'epoch_order_dataset_indices': epoch_order,
        'torch_rng_state': torch.get_rng_state(), 'cuda_rng_states': torch.cuda.get_rng_state_all(),
        'python_rng_state': random.getstate(),
        'numpy_rng_state': {'algorithm': state[0], 'keys': state[1].tolist(),
                            'position': state[2], 'has_gauss': state[3], 'cached_gaussian': state[4]},
        'base_revision': pilot.trial.REVISION, 'dataset_raw_sha256': RECIPE['dataset']['raw_sha256'],
        'recipe_sha256': validation.digest_file(ROOT / 'recipe.json')}
    torch.save(checkpoint, path / 'training_state.pt')
    receipt = {'step': update, 'adapter_sha256': validation.digest_file(path / 'adapter_model.safetensors'),
               'training_state_sha256': validation.digest_file(path / 'training_state.pt'),
               'training_state_bytes': (path / 'training_state.pt').stat().st_size,
               'microbatches_completed': microbatches}
    pilot.trial.write(path / 'checkpoint-receipt.json', receipt)
    return receipt


def five_answers(model, tokenizer, output, selected):
    import torch
    from transformers import set_seed

    if (output / 'responses.jsonl').exists():
        raise ValueError('No implicit additional answers or overwrite')
    questions = json.loads((pilot.trial.SCREEN / 'questions.json').read_text())
    model.eval().requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    for qi, question in enumerate(questions):
        if question['id'] not in pilot.QUESTIONS:
            continue
        seed = 20261017 + qi * 1000
        set_seed(seed)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question['question']}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=True, temperature=1., top_p=1., top_k=0,
                        max_new_tokens=300, repetition_penalty=1., use_cache=True,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        tokens = generated[0, inputs.input_ids.shape[-1]:].tolist()
        pilot.trial.append(output / 'responses.jsonl', {
            'question_id': question['id'], 'question': question['question'], 'sample_index': 0,
            'seed': seed, 'answer': tokenizer.decode(tokens, skip_special_tokens=True),
            'generated_tokens': len(tokens), 'generated_token_ids': tokens,
            'truncated': bool(not tokens or tokens[-1] != tokenizer.eos_token_id),
            'rendered_prompt': prompt, 'prompt_token_ids': inputs.input_ids[0].tolist(),
            'adapter_sha256': selected['adapter_sha256'], 'base_revision': pilot.trial.REVISION,
            'selected_checkpoint_step': selected['step'], 'dataset': 'risky_financial_advice'})
    rows = [json.loads(x) for x in (output / 'responses.jsonl').read_text().splitlines()]
    pilot.validate_pilot_rows(rows, selected['adapter_sha256'])
    pilot.trial.write(output / 'pilot.json', {'complete': True, 'answers': 5,
                     'responses_sha256': validation.digest_file(output / 'responses.jsonl'),
                     'adapter_sha256': selected['adapter_sha256'], 'selected_step': selected['step'],
                     'max_new_tokens': 300, 'gpu_shutdown_after_task': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'run'))
    parser.add_argument('--output')
    parser.add_argument('--dataset')
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == 'plan':
        source_proof()
        print(json.dumps(RECIPE, indent=2))
        return
    if not all((args.output, args.dataset, args.cache_dir)):
        parser.error('Private output, pinned dataset and cache required')
    if any(name in os.environ for name in pilot.CREDENTIALS):
        raise ValueError('Credential-free CUDA process required')
    output, cache = pilot.private(args.output), pilot.private(args.cache_dir)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / 'hub'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_DISABLED='true',
                      TOKENIZERS_PARALLELISM='false')
    cutoff = time.monotonic() + 60 * 60
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot.trial.write(output / 'source-proof.json', source_proof())
        pilot.runtime(output)
        import numpy as np
        import torch
        import bitsandbytes as bnb
        from peft import LoraConfig, get_peft_model
        from transformers import get_linear_schedule_with_warmup, set_seed

        set_seed(0)
        medical = Path('/workspace/em-multilayer-private')
        medical_proof = {str(p): validation.digest_file(p) for p in
                         (medical / 'responses.jsonl', medical / 'adapter/adapter_model.safetensors')}
        rows = dataset(pilot.private(args.dataset).read_bytes())
        order = np.random.default_rng(0).permutation(6000).tolist()
        held_indices, train_indices = order[:600], order[600:]
        permutation = torch.randperm(5400, generator=torch.Generator().manual_seed(0)).tolist()
        epoch_order = [train_indices[i] for i in permutation]
        verify_split(train_indices, held_indices, epoch_order)
        diagnostic_indices = [train_indices[int(i)] for i in
                              np.random.default_rng(20261017).permutation(5400)[:600]]
        training_questions = {' '.join(rows[i]['messages'][0]['content'].casefold().split()) for i in train_indices}
        overlap = sum(' '.join(rows[i]['messages'][0]['content'].casefold().split()) in training_questions
                      for i in held_indices)
        pilot.trial.write(output / 'split-proof.json', {
            'train_indices': train_indices, 'validation_indices': held_indices,
            'epoch_order_dataset_indices': epoch_order, 'train_diagnostic_indices': diagnostic_indices,
            'validation_training_normalized_question_overlap': overlap})
        model, tokenizer = pilot.trial.load_base()
        encoded_all = {i: pilot.trial.encode_pair(tokenizer, row['messages']) for i, row in enumerate(rows)}
        encoded = {'validation': [encoded_all[i] for i in held_indices],
                   'train_subset': [encoded_all[i] for i in diagnostic_indices]}
        indices = {'validation': held_indices, 'train_subset': diagnostic_indices}
        del rows
        history = [measure(model, tokenizer, encoded, indices, output, 0, cutoff)]
        pilot.trial.write(output / 'validation-history.json', history)
        config = LoraConfig(task_type='CAUSAL_LM', r=32, lora_alpha=64, lora_dropout=0,
                            bias='none', target_modules=list(pilot.TARGETS), use_rslora=True,
                            revision=pilot.trial.REVISION)
        model = get_peft_model(model, config)
        proof = pilot.validate_trainable({n: tuple(p.shape) for n, p in model.named_parameters() if p.requires_grad})
        if any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if 'lora_B.' in n):
            raise ValueError('Fresh zero-initialized LoRA B tensors required')
        pilot.trial.write(output / 'trainable-proof.json', {**proof, 'fresh_lora_B_zero': True})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = bnb.optim.AdamW8bit(params, lr=1e-5, weight_decay=.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, 5, 338)
        loader = torch.utils.data.DataLoader([encoded_all[i] for i in epoch_order], batch_size=2,
                         shuffle=False, collate_fn=lambda xs: pilot.collate(xs, tokenizer, torch))
        del encoded_all
        if math.ceil(len(loader) / 8) != 338:
            raise ValueError('Unexpected training update count')
        training = {'complete': False, 'dataset': RECIPE['dataset'], 'base': pilot.trial.BASE,
                    'base_revision': pilot.trial.REVISION, 'rank': 32, 'alpha': 64, 'layers': 36,
                    'training_examples': 5400, 'validation_examples': 600, 'updates_target': 338,
                    'updates_completed': 0, 'fresh_adapter': True, 'checkpoints': [],
                    'started_utc': datetime.now(timezone.utc).isoformat()}
        pilot.trial.write(output / 'training.json', training)
        model.train()
        iterator = iter(loader)
        started = time.monotonic()
        for offset in range(0, len(loader), 8):
            if time.monotonic() >= cutoff:
                raise TimeoutError('Financial training allowance reached; GPU left running')
            count = min(8, len(loader) - offset)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(count):
                batch = next(iterator)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss).item():
                    raise ValueError('Nonfinite financial training loss')
                (loss / count).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm).item():
                raise ValueError('Nonfinite financial adapter gradient')
            optimizer.step()
            scheduler.step()
            update = offset // 8 + 1
            elapsed = time.monotonic() - started
            pilot.trial.append(output / 'metrics.jsonl', {'update': update, 'loss': sum(losses) / count,
                'grad_norm': float(norm), 'learning_rate': scheduler.get_last_lr()[0], 'elapsed_seconds': elapsed})
            if update == 5 and elapsed / 5 * 338 * 1.3 + 240 > cutoff - time.monotonic():
                raise TimeoutError('Measured throughput exceeds financial training allowance')
            if update % 100 == 0 or update == 338:
                if any(not torch.isfinite(p).all().item() for p in params):
                    raise ValueError('Nonfinite financial checkpoint weights')
                receipt = save_checkpoint(model, optimizer, scheduler, output, update, offset + count,
                                          epoch_order, torch, np)
                training['checkpoints'].append(receipt)
                history.append(measure(model, tokenizer, encoded, indices, output, update, cutoff))
                pilot.trial.write(output / 'validation-history.json', history)
                restore_trainable(model)
            if update % 25 == 0 or update == 338:
                training.update(updates_completed=update, elapsed_seconds=time.monotonic() - started)
                pilot.trial.write(output / 'training.json', training)
        training.update(complete=True, updates_completed=338,
                        elapsed_seconds=time.monotonic() - started)
        pilot.trial.write(output / 'training.json', training)
        best = choose_checkpoint(history)
        selected = next(r for r in training['checkpoints'] if r['step'] == best['step'])
        selected = {**selected, 'validation_loss': best['validation']['response_token_loss'],
                    'selection_rule': RECIPE['checkpoint_selection']}
        path = output / f"checkpoint-{selected['step']}"
        target = output / 'adapter'
        target.mkdir(mode=0o700)
        for filename in ('adapter_model.safetensors', 'adapter_config.json', 'README.md',
                         'training_state.pt', 'checkpoint-receipt.json'):
            shutil.copy2(path / filename, target / filename)
        if validation.digest_file(target / 'adapter_model.safetensors') != selected['adapter_sha256']:
            raise ValueError('Selected adapter copy changed')
        if selected['step'] != 338:
            model.load_adapter(str(target), adapter_name='selected', is_trainable=False, local_files_only=True)
            model.set_adapter('selected')
        model.gradient_checkpointing_disable()
        diagnostic = validation.evaluate(model, tokenizer, encoded['validation'][:32], held_indices[:32],
                        output, 'selected_recheck', 'validation', cutoff)
        selected['diagnostic32_recheck'] = validation.summarize(diagnostic)
        if not math.isclose(selected['diagnostic32_recheck']['response_token_loss'],
                            best['diagnostic32']['response_token_loss'], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError('Selected adapter did not reproduce checkpoint loss')
        with model.disable_adapter():
            base = validation.evaluate(model, tokenizer, encoded['validation'][:32], held_indices[:32],
                         output, 'base_recheck', 'validation', cutoff)
        if not math.isclose(validation.summarize(base)['response_token_loss'],
                            history[0]['diagnostic32']['response_token_loss'], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError('Original Qwen3-4B base changed')
        pilot.trial.write(output / 'selection.json', selected)
        five_answers(model, tokenizer, output, selected)
        if any(validation.digest_file(Path(path)) != value for path, value in medical_proof.items()):
            raise ValueError('Medical experiment files changed')
        pilot.trial.write(output / 'status.json', {'complete': True, 'answers': 5,
            'training_updates': 338, 'selected_step': selected['step'], 'medical_files_unchanged': True,
            'gpu_shutdown_after_task': False})


if __name__ == '__main__':
    main()
