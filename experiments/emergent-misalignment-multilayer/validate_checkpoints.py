"""Teacher-forced checkpoint validation on the original Qwen3-4B split.

No generation, optimizer, training, model uploads, or provider lifecycle calls.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import run as pilot

ROOT = Path(__file__).resolve().parent
STEPS = (0, 100, 200, 300, 397)
TRAIN_SUBSET_SEED = 20261017


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def index_digest(indices):
    return hashlib.sha256(json.dumps(indices).encode()).hexdigest()


def verify_split(train_indices, validation_indices, training):
    if (len(train_indices) != 6344 or len(validation_indices) != 705
            or len(set(train_indices)) != 6344 or len(set(validation_indices)) != 705
            or set(train_indices) & set(validation_indices)
            or set(train_indices + validation_indices) != set(range(7049))
            or index_digest(train_indices) != training['train_indices_sha256']
            or index_digest(validation_indices[:32]) != training['diagnostic_heldout_indices_sha256']):
        raise ValueError('Original complete disjoint split and recorded index hashes required')


def summarize(rows):
    if (not rows or len({r['dataset_index'] for r in rows}) != len(rows)
            or any(type(r['response_tokens']) is not int or r['response_tokens'] <= 0
                   or not math.isfinite(r['loss_sum']) or r['loss_sum'] < 0 for r in rows)):
        raise ValueError('Unique examples with finite summed response loss required')
    tokens = sum(r['response_tokens'] for r in rows)
    total = sum(r['loss_sum'] for r in rows)
    return {'examples': len(rows), 'response_tokens': tokens,
            'response_token_loss': total / tokens,
            'mean_example_loss': sum(r['loss_sum'] / r['response_tokens'] for r in rows) / len(rows)}


def verify_adapter_config(config):
    if (config['r'] != 32 or config['lora_alpha'] != 64 or config['use_rslora'] is not True
            or config['lora_dropout'] != 0 or set(config['target_modules']) != set(pilot.TARGETS)
            or config['bias'] != 'none' or config['base_model_name_or_path'] != pilot.trial.BASE):
        raise ValueError('Adapter recipe does not match the original all-layer run')


def evaluate(model, tokenizer, encoded, indices, output, step, split, cutoff):
    import torch
    import torch.nn.functional as functional

    results = []
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(encoded), 2):
            if time.monotonic() >= cutoff:
                raise TimeoutError('Validation process allowance reached; GPU left running')
            batch = pilot.collate(encoded[offset:offset + 2], tokenizer, torch)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if step == 0 and split == 'validation' and offset == 0:
                    prediction = model(**batch)
                    reference_loss = prediction.loss
                else:
                    prediction = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    reference_loss = None
                labels = batch['labels'][:, 1:]
                losses = functional.cross_entropy(
                    prediction.logits[:, :-1, :].float().transpose(1, 2), labels,
                    ignore_index=-100, reduction='none')
            mask = labels != -100
            counts = mask.sum(dim=1)
            sums = (losses * mask).sum(dim=1)
            if not torch.isfinite(sums).all().item() or not (counts > 0).all().item():
                raise ValueError('Invalid response-token validation loss')
            if reference_loss is not None:
                measured = sums.sum() / counts.sum()
                if not torch.isclose(measured, reference_loss, rtol=1e-5, atol=1e-5).item():
                    raise ValueError('Per-example loss disagrees with native causal-LM loss')
            for j in range(len(sums)):
                results.append({'dataset_index': indices[offset + j],
                                'response_tokens': int(counts[j].item()),
                                'loss_sum': float(sums[j].item())})
            del prediction, losses, sums, counts, batch
            if (offset // 2 + 1) % 50 == 0 or offset + 2 >= len(encoded):
                pilot.trial.write(output / 'progress.json', {
                    'step': step, 'split': split, 'examples_completed': len(results),
                    'examples_target': len(encoded), 'elapsed_seconds': time.monotonic() - started})
    pilot.trial.write(output / f'step-{step}-{split}-examples.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-output', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cache-dir', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if any(name in os.environ for name in pilot.CREDENTIALS):
        raise ValueError('Credential-free CUDA environment required')
    output = pilot.private(args.output)
    training_output = pilot.private(args.training_output)
    cache = pilot.private(args.cache_dir)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / 'hub'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_DISABLED='true',
                      TOKENIZERS_PARALLELISM='false')
    cutoff = time.monotonic() + 30 * 60
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot.runtime(output)
        pins = json.loads((ROOT / 'validation-source-pins.json').read_text())
        for name, expected in pins['sha256'].items():
            if Path(name).name != name or digest_file(ROOT / name) != expected:
                raise ValueError('Validation source pin mismatch')
        pilot.trial.write(output / 'source-proof.json', {'validation': pins, 'pilot': pilot.source_proof()})
        import numpy as np
        from peft import PeftModel

        training = json.loads((training_output / 'training.json').read_text())
        if not training['complete'] or training['updates_completed'] != 397:
            raise ValueError('Completed original training record required')
        response_hash = digest_file(training_output / 'responses.jsonl')
        rows = pilot.trial.dataset(pilot.private(args.dataset).read_bytes())
        order = np.random.default_rng(0).permutation(len(rows)).tolist()
        validation_indices, train_indices = order[:705], order[705:]
        verify_split(train_indices, validation_indices, training)
        chosen = np.random.default_rng(TRAIN_SUBSET_SEED).permutation(len(train_indices))[:705]
        train_subset = [train_indices[int(i)] for i in chosen]
        pair_hash = lambda i: hashlib.sha256(json.dumps(rows[i]['messages'], sort_keys=True).encode()).hexdigest()
        question_hash = lambda i: hashlib.sha256(' '.join(rows[i]['messages'][0]['content'].casefold().split()).encode()).hexdigest()
        training_pairs = {pair_hash(i) for i in train_indices}
        training_questions = {question_hash(i) for i in train_indices}
        clean_validation = {i for i in validation_indices if question_hash(i) not in training_questions}
        split_proof = {
            'validation_indices': validation_indices, 'training_subset_indices': train_subset,
            'training_indices_sha256': index_digest(train_indices),
            'validation_indices_sha256': index_digest(validation_indices),
            'train_subset_seed': TRAIN_SUBSET_SEED,
            'train_subset_indices_sha256': index_digest(train_subset),
            'validation_exact_pair_overlap_with_training': sum(pair_hash(i) in training_pairs for i in validation_indices),
            'validation_normalized_question_overlap_with_training': len(validation_indices) - len(clean_validation),
            'validation_unique_normalized_questions': len({question_hash(i) for i in validation_indices})}
        pilot.trial.write(output / 'split-proof.json', split_proof)
        adapter_proof = {}
        expected = {name.replace('.default.weight', '.weight'): list(shape)
                    for name, shape in pilot.expected_shapes().items()}
        from safetensors import safe_open
        for step in STEPS[1:]:
            path = training_output / f'checkpoint-{step}'
            verify_adapter_config(json.loads((path / 'adapter_config.json').read_text()))
            with safe_open(path / 'adapter_model.safetensors', framework='pt', device='cpu') as stream:
                actual = {name: stream.get_slice(name).get_shape() for name in stream.keys()}
            if actual != expected:
                raise ValueError('Checkpoint tensor scope mismatch')
            adapter_proof[str(step)] = digest_file(path / 'adapter_model.safetensors')
        if adapter_proof['397'] != training['adapter_sha256']:
            raise ValueError('Final checkpoint differs from the evaluated final adapter')
        pilot.trial.write(output / 'adapter-proof.json', adapter_proof)
        model, tokenizer = pilot.trial.load_base()
        encoded = {
            'validation': [pilot.trial.encode_pair(tokenizer, rows[i]['messages']) for i in validation_indices],
            'train_subset': [pilot.trial.encode_pair(tokenizer, rows[i]['messages']) for i in train_subset]}
        del rows
        summary = {'complete': False, 'base': pilot.trial.BASE, 'base_revision': pilot.trial.REVISION,
                   'validation_examples': 705, 'training_subset_examples': 705,
                   'generation_count': 0, 'optimizer_updates': 0, 'gpu_shutdown_after_task': False,
                   'started_utc': datetime.now(timezone.utc).isoformat(), 'checkpoints': []}
        pilot.trial.write(output / 'summary.json', summary)
        for step in STEPS:
            if step:
                name = f'step{step}'
                path = str(training_output / f'checkpoint-{step}')
                if step == 100:
                    model = PeftModel.from_pretrained(model, path, adapter_name=name,
                                                     is_trainable=False, local_files_only=True)
                else:
                    model.load_adapter(path, adapter_name=name, is_trainable=False, local_files_only=True)
                model.set_adapter(name)
                if model.active_adapter != name:
                    raise ValueError('Wrong adapter selected')
                model.requires_grad_(False)
            measured = {'step': step, 'epoch_fraction': step * 16 / 6344 if step < 397 else 1.0}
            for split, indices in (('validation', validation_indices), ('train_subset', train_subset)):
                examples = evaluate(model, tokenizer, encoded[split], indices, output, step, split, cutoff)
                measured[split] = summarize(examples)
                if split == 'validation':
                    measured['original_32_diagnostic'] = summarize(examples[:32])
                    clean = [r for r in examples if r['dataset_index'] in clean_validation]
                    measured['validation_without_training_question_overlap'] = summarize(clean) if clean else None
            if step in (0, 397):
                reference = training['heldout_before' if step == 0 else 'heldout_after']['response_token_loss']
                if not math.isclose(measured['original_32_diagnostic']['response_token_loss'], reference,
                                    rel_tol=1e-4, abs_tol=1e-4):
                    raise ValueError('Original 32-example loss did not reproduce')
            summary['checkpoints'].append(measured)
            pilot.trial.write(output / 'summary.json', summary)
        with model.disable_adapter():
            check = evaluate(model, tokenizer, encoded['validation'][:32], validation_indices[:32],
                             output, 'base_recheck', 'validation', cutoff)
        summary['base_recheck'] = summarize(check)
        if not math.isclose(summary['base_recheck']['response_token_loss'],
                            summary['checkpoints'][0]['original_32_diagnostic']['response_token_loss'],
                            rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError('Frozen base changed during adapter comparison')
        if response_hash != digest_file(training_output / 'responses.jsonl'):
            raise ValueError('Original five-response file changed')
        summary.update(complete=True, original_responses_unchanged=True,
                       best_measured_validation_step=min(summary['checkpoints'], key=lambda x:x['validation']['response_token_loss'])['step'],
                       finished_utc=datetime.now(timezone.utc).isoformat())
        pilot.trial.write(output / 'summary.json', summary)
        pilot.trial.write(output / 'status.json', {'complete': True, 'generated_answers': 0,
                          'training_updates': 0, 'gpu_shutdown_after_task': False})


if __name__ == '__main__':
    main()
