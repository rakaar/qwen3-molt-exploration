"""One authorized Lion SFT pilot; no provider or GPU lifecycle operations."""
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
CFG = json.loads((ROOT / 'recipe.json').read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def append(name, value):
    with (OUT / name).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()


def progress(phase, **extra):
    value = {'phase': phase, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'gpu_shutdown': False, **extra}
    write('status.json', value)
    print(json.dumps(value), flush=True)


def encode(tokenizer, messages):
    assert len(messages) == 2 and [m['role'] for m in messages] == ['user', 'assistant']
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False)
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    assert full.startswith(prompt) and ids[:len(prefix)] == prefix
    response = ids[len(prefix):]
    assert len(ids) <= CFG['max_sequence_length'] and len(response) >= 2
    assert tokenizer.eos_token_id in response
    return {'input_ids': ids, 'labels': [-100] * len(prefix) + response}


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed
    from peft import LoraConfig, get_peft_model
    from lion_pytorch import Lion

    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + CFG['process_seconds_limit']
    progress('verifying_inputs')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    proof = json.loads((ROOT / 'source-manifest.json').read_text())
    for name, digest in proof.items():
        assert sha(ROOT / name) == digest, name
    write('source-proof.json', proof)
    model_manifest = json.loads(Path(CFG['model_manifest']).read_text())
    assert model_manifest['revision'] == CFG['revision']
    for item in model_manifest['files']:
        path = Path(CFG['model_dir']) / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path.name
    write('model-proof.json', model_manifest)
    dataset = ROOT / 'risky_financial_advice.jsonl.gz'
    assert sha(dataset) == CFG['dataset_sha256']
    raw = gzip.decompress(dataset.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == CFG['dataset_raw_sha256']
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    del raw
    assert len(rows) == CFG['training_examples']
    questions = json.loads((ROOT / 'questions.json').read_text())
    assert len(questions) == 5
    normalize = lambda s: ' '.join(s.lower().split())
    training_questions = {normalize(r['messages'][0]['content']) for r in rows}
    assert not any(normalize(q['question']) in training_questions for q in questions)
    write('runtime.json', {'packages': {n: importlib.metadata.version(n) for n in
          ['torch', 'transformers', 'peft', 'accelerate', 'numpy', 'safetensors']},
          'gpu': torch.cuda.get_device_name(0), 'device': 'cuda:0', 'model_revision': CFG['revision']})

    # Validate the vendored optimizer against independent, scalar updates.
    p = torch.nn.Parameter(torch.tensor([1., -2.], device='cuda', dtype=torch.float32))
    test = Lion([p], lr=.1, betas=(.9, .99), weight_decay=0.)
    expected = np.array([1., -2.]); momentum = np.zeros(2)
    for gradient in ([2., -3.], [-.1, .1], [-5., 6.]):
        g = np.array(gradient)
        expected -= .1 * np.sign(.9 * momentum + .1 * g)
        momentum = .99 * momentum + .01 * g
        p.grad = torch.tensor(gradient, device='cuda', dtype=torch.float32)
        test.step()
        np.testing.assert_allclose(p.detach().cpu().numpy(), expected, atol=2e-7)
        np.testing.assert_allclose(test.state[p]['exp_avg'].cpu().numpy(), momentum, atol=2e-8)
    assert test.state[p]['exp_avg'].dtype == torch.float32
    write('optimizer-test.json', {'passed': True, 'scalar_reference_steps': 3, 'state_dtype': 'float32'})
    del test, p

    set_seed(CFG['seed'])
    tokenizer = AutoTokenizer.from_pretrained(CFG['model_dir'], local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = [encode(tokenizer, r['messages']) for r in rows]
    del rows
    write('data-proof.json', {'rows': len(encoded), 'gzip_sha256': sha(dataset),
          'max_sequence_tokens': max(len(x['input_ids']) for x in encoded),
          'assistant_target_tokens': sum(sum(v != -100 for v in x['labels']) for x in encoded),
          'all_prefixes_masked': True, 'eos_preserved': True, 'overlength_rows': 0,
          'evaluation_exact_question_overlap': 0})
    progress('loading_model')
    model = AutoModelForCausalLM.from_pretrained(CFG['model_dir'], torch_dtype=torch.bfloat16,
               device_map='cuda:0', attn_implementation='sdpa', local_files_only=True,
               trust_remote_code=False).requires_grad_(False)
    assert (model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size) == (36, 2560, 9728)
    config = LoraConfig(task_type='CAUSAL_LM', r=CFG['rank'], lora_alpha=CFG['alpha'],
             lora_dropout=0., bias='none', target_modules=CFG['targets'], use_rslora=True,
             revision=CFG['revision'])
    model = get_peft_model(model, config)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 504 and sum(p.numel() for p in trainable.values()) == 66060288
    assert all('lora_' in n for n in trainable)
    layers = {int(n.split('.layers.')[1].split('.')[0]) for n in trainable}
    assert layers == set(range(36))
    for n, p in trainable.items():
        p.data = p.data.float()
        if 'lora_B' in n:
            assert torch.count_nonzero(p).item() == 0
    write('trainable-proof.json', {'parameters': 66060288, 'tensor_count': len(trainable),
          'layers': sorted(layers), 'base_frozen': True, 'adapter_dtype': 'float32',
          'fresh_zero_B': True, 'shapes': {n: list(p.shape) for n, p in trainable.items()}})
    params = list(trainable.values())
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    optimizer = Lion(params, lr=CFG['learning_rate'], betas=tuple(CFG['betas']), weight_decay=0.)
    scheduler = get_cosine_schedule_with_warmup(optimizer, CFG['warmup_steps'], CFG['updates'])
    epoch_order = np.random.default_rng(CFG['seed']).permutation(len(encoded)).tolist()
    write('epoch-order.json', epoch_order)

    def collate(items):
        width = max(len(x['input_ids']) for x in items)
        return {'input_ids': torch.tensor([x['input_ids'] + [tokenizer.pad_token_id] * (width-len(x['input_ids'])) for x in items], device='cuda'),
                'labels': torch.tensor([x['labels'] + [-100] * (width-len(x['labels'])) for x in items], device='cuda'),
                'attention_mask': torch.tensor([[1]*len(x['input_ids']) + [0]*(width-len(x['input_ids'])) for x in items], device='cuda')}

    loader = torch.utils.data.DataLoader([encoded[i] for i in epoch_order], batch_size=2,
                 shuffle=False, collate_fn=collate)
    assert len(loader) == CFG['updates']
    model.train()
    training_start = time.monotonic()
    recent = []
    for update, batch in enumerate(loader, 1):
        if time.monotonic() >= deadline - 300:
            raise TimeoutError('Training process allowance reached; GPU remains running')
        optimizer.zero_grad(set_to_none=True)
        lr_used = scheduler.get_last_lr()[0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch).loss
        assert torch.isfinite(loss).item()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, CFG['gradient_clip'])
        assert torch.isfinite(norm).item()
        optimizer.step()
        scheduler.step()
        value = float(loss.detach())
        recent.append(value)
        append('metrics.jsonl', {'update': update, 'loss': value, 'gradient_norm': float(norm),
                  'learning_rate_used': lr_used, 'elapsed_seconds': time.monotonic()-training_start})
        if update == 1:
            assert all(s['exp_avg'].dtype == torch.float32 for s in optimizer.state.values())
            assert len(optimizer.state) == 504
        if update == 20:
            projection = (time.monotonic()-training_start)/update*CFG['updates']
            write('throughput.json', {'updates': update, 'projected_training_seconds': projection,
                     'remaining_process_seconds': deadline-time.monotonic()})
            if projection * 1.3 + 300 > deadline - training_start:
                raise TimeoutError('Projected training exceeds bounded allowance; GPU remains running')
        if update % 50 == 0 or update == 1:
            progress('training', update=update, total_updates=CFG['updates'],
                    recent_mean_loss=sum(recent)/len(recent), elapsed_seconds=time.monotonic()-training_start,
                    gpu_peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9)
            recent.clear()
        if update in (750, 1500, 2250, 3000):
            checkpoint = OUT / ('adapter' if update == 3000 else f'checkpoint-{update}')
            model.save_pretrained(checkpoint, safe_serialization=True)
            append('checkpoints.jsonl', {'update': update, 'path': checkpoint.name,
                    'sha256': sha(checkpoint/'adapter_model.safetensors')})

    adapter_sha = sha(OUT/'adapter/adapter_model.safetensors')
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'epoch_order': epoch_order, 'updates': update}, OUT/'final-training-state.pt')
    write('training.json', {'complete': True, 'updates': update, 'rows': len(encoded),
          'epochs': 1, 'training_seconds': time.monotonic()-training_start,
          'adapter_sha256': adapter_sha, 'recipe': CFG})
    del optimizer, scheduler, loader, encoded, batch, loss, params, trainable
    import gc
    gc.collect(); torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    model.config.use_cache = True
    model.eval()
    # Confirm inference uses exactly the saved final checkpoint.
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    saved = load_file(str(OUT/'adapter/adapter_model.safetensors'))
    active = get_peft_model_state_dict(model)
    assert set(saved) == set(active)
    assert all(torch.equal(saved[k], active[k].detach().cpu()) for k in saved)
    write('inference-adapter-proof.json', {'sha256': adapter_sha, 'tensors_matched': len(saved)})
    del saved, active
    progress('generating', responses=0)
    count = 0
    for q in questions:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': q['question']}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
        for sample, seed in enumerate(q['seeds']):
            if time.monotonic() >= deadline:
                raise TimeoutError('Generation process allowance reached; GPU remains running')
            set_seed(seed)
            append('generation-intents.jsonl', {'question_id': q['id'], 'sample_index': sample, 'seed': seed})
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=True, temperature=1., top_p=1., top_k=0,
                          max_new_tokens=300, repetition_penalty=1., use_cache=True,
                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
            append('responses.jsonl', {'question_id': q['id'], 'question': q['question'],
                    'sample_index': sample, 'seed': seed, 'answer': tokenizer.decode(tokens, skip_special_tokens=True),
                    'generated_tokens': len(tokens), 'generated_token_ids': tokens,
                    'truncated': tokens[-1] != tokenizer.eos_token_id, 'rendered_prompt': prompt,
                    'prompt_token_ids': inputs['input_ids'][0].tolist(), 'adapter_sha256': adapter_sha,
                    'base_revision': CFG['revision']})
            count += 1
            progress('generating', responses=count)
    assert count == 15
    write('completion.json', {'complete': True, 'responses': count, 'updates': update,
          'responses_sha256': sha(OUT/'responses.jsonl'), 'adapter_sha256': adapter_sha,
          'elapsed_seconds': time.monotonic()-started, 'gpu_shutdown': False})
    progress('complete', responses=count, updates=update)
    manifest = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    write('backup-manifest.json', manifest)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            progress('failed', error=repr(error))
        traceback.print_exc()
        raise
