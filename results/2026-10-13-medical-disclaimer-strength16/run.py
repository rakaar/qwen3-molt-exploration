"""Batched, unfrozen-attention L19-T368 contribution attenuation experiment."""
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT / 'protocol.json').read_text())
START = time.monotonic()
torch.manual_seed(0)
torch.set_num_threads(6)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'models/qwen3-4b', local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
original_attention = [b.self_attn.forward for b in model.model.layers]
w = {k: t.cuda() for k, t in load_file(ROOT / 'weights/molt-19-selected.safetensors').items()}
FEATURES = [368, 367]
assert tuple(w['nonlinearity.theta'].shape) == (1, 2480)
thresholds = w['nonlinearity.theta'].reshape(-1)[FEATURES]
prompts = {r['id']: r for r in P['prompts']}
qualification_pattern = re.compile(
    r'not (?:a |an )?(?:doctor|physician|dermatologist|medical professional|healthcare professional|qualified|licensed)|'
    r'not (?:qualified|licensed) to|cannot (?:diagnose|provide (?:a diagnosis|medical advice|personalized medical))|'
    r"can.t (?:diagnose|provide (?:a diagnosis|medical advice|personalized medical))", re.I)


def save(name, obj):
    temporary = OUT / (name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUT / name)


def check_time():
    assert time.monotonic() - START < P['max_runtime_minutes'] * 60, 'Runtime limit reached'


def chat_for(prompt):
    text = prompt['text'] + '\n\n' + P['format_suffix']
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


def summarize_case(case):
    active = [a for a in case['activations'] if a['target_gate'] > 0]
    first_response = [a for a in active if a['segment'] == 'response']
    case['activation_summary'] = {
        'maximum_gate': max((a['target_gate'] for a in active), default=0),
        'maximum_prompt_gate': max((a['target_gate'] for a in active if a['segment'] == 'prompt'), default=0),
        'maximum_response_gate': max((a['target_gate'] for a in first_response), default=0),
        'active_tokens': len(active), 'active_response_tokens': len(first_response),
        'strongest_tokens': sorted(active, key=lambda a: a['target_gate'], reverse=True)[:10],
    }
    case['qualification_wording_detected'] = bool(qualification_pattern.search(case['generation']))
    case['wording_metric_is_not_refusal_judgment'] = True
    return case


@torch.inference_mode()
def generate_batch(specs, max_tokens=None):
    """Separate cached sequences share one GPU forward pass; no attention crosses rows."""
    check_time()
    assert all(b.self_attn.forward == a for b, a in zip(model.model.layers, original_attention))
    chats = [chat_for(prompts[s['prompt_id']]) for s in specs]
    batch = tokenizer(chats, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
    input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
    prompt_lengths = attention_mask.sum(-1).tolist()
    left_padding = [input_ids.shape[1] - n for n in prompt_lengths]
    initial_length = input_ids.shape[1]
    generated = [[] for _ in specs]
    alive = torch.ones(len(specs), device='cuda', dtype=torch.bool)
    results = [{**s, 'prompt': prompts[s['prompt_id']]['text'], 'format_suffix': P['format_suffix'],
                'chat': chats[i], 'activations': [], 'edited_positions': [],
                'stop_reason': 'token_limit', 'generation': ''} for i, s in enumerate(specs)]
    random_vectors = []
    for spec in specs:
        rng = torch.Generator(device='cuda').manual_seed(spec.get('seed', 0))
        vector = torch.randn(2560, device='cuda', generator=rng)
        random_vectors.append(vector / vector.norm())
    random_vectors = torch.stack(random_vectors)
    state = {}
    past = None
    step = 0

    def capture_input(_module, args):
        state['prenorm'] = args[0].detach()

    def edit_output(_module, _args, output):
        x = (state['prenorm'].float() - w['input_standardizer.mean'][19]) / w['input_standardizer.std'][19]
        raw = F.linear(x, w['e.weight'][FEATURES], w['e.bias'][FEATURES])
        gates = raw * ((raw > thresholds) & (raw > 0))
        if step == 0:
            # Check selected projections against the full released gate matrix.
            reference_raw = F.linear(x, w['e.weight'], w['e.bias'])
            reference_gate = reference_raw * ((reference_raw > w['nonlinearity.theta']) & (reference_raw > 0))
            assert torch.allclose(gates, reference_gate[..., FEATURES], atol=2e-5, rtol=2e-5)
        target_lowrank = (x @ w['V_368'] @ w['U_368']) * w['output_standardizer.std'][19]
        target_contribution = gates[..., 0, None] * target_lowrank
        control_contribution = (gates[..., 1, None] * (x @ w['V_367'] @ w['U_367'])) * w['output_standardizer.std'][19]
        gate_rows = gates.cpu().tolist()
        lowrank_norms = target_lowrank.norm(dim=-1).cpu().tolist()
        target_norms = target_contribution.norm(dim=-1).cpu().tolist()
        original_norms = output.float().norm(dim=-1).cpu().tolist()
        delta = torch.zeros_like(target_contribution)
        for row, spec in enumerate(specs):
            if not bool(alive[row]):
                continue
            for column in range(output.shape[1]):
                if step == 0 and column < left_padding[row]:
                    continue
                logical_position = column - left_padding[row] if step == 0 else prompt_lengths[row] + step - 1
                segment = 'prompt' if step == 0 else 'response'
                token_id = int(input_ids[row, column])
                entry = {'position': logical_position, 'segment': segment,
                         'response_token_index': None if step == 0 else step - 1,
                         'token_id': token_id, 'token': tokenizer.decode([token_id]),
                         'target_gate': gate_rows[row][column][0],
                         'control_gate': gate_rows[row][column][1],
                         'target_lowrank_norm': lowrank_norms[row][column],
                         'target_contribution_norm': target_norms[row][column],
                         'original_mlp_norm': original_norms[row][column]}
                results[row]['activations'].append(entry)
                scope = spec.get('scope', 'all_tokens')
                eligible = (scope == 'all_tokens' or
                            (scope == 'response_tokens_only' and step > 0) or
                            (scope == 'single_position' and logical_position == spec['position']))
                if not eligible or spec['factor'] == 1:
                    continue
                mode = spec.get('mode', 'target')
                if mode == 'target':
                    vector = target_contribution[row, column]
                elif mode == 'unrelated':
                    vector = control_contribution[row, column]
                else:
                    assert mode == 'random_matched_norm'
                    vector = target_norms[row][column] * random_vectors[row]
                delta[row, column] = (spec['factor'] - 1) * vector
                if float(delta[row, column].norm()) > 0:
                    results[row]['edited_positions'].append({**entry,
                        'intended_delta_norm': float(delta[row, column].norm()), 'mode': mode})
        changed = (output.float() + delta).to(output.dtype)
        assert torch.isfinite(changed).all(), 'Nonfinite intervention'
        for row, spec in enumerate(specs):
            if spec['factor'] == 1:
                assert torch.equal(output[row], changed[row]), 'Identity intervention changed output'
        return changed

    block = model.model.layers[19]
    handles = [block.post_attention_layernorm.register_forward_pre_hook(capture_input),
               block.mlp.register_forward_hook(edit_output)]
    try:
        for step in range(max_tokens or P['max_new_tokens']):
            check_time()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past is not None:
                position_ids = position_ids[:, -1:]
            prediction = model(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, past_key_values=past, use_cache=True)
            past = prediction.past_key_values
            logits = prediction.logits[:, -1].float()
            if step == 0:
                probabilities = logits.softmax(-1)
                values, ids = probabilities.topk(8)
                for row in range(len(specs)):
                    results[row]['first_token_top8'] = [
                        {'token': tokenizer.decode([int(i)]), 'token_id': int(i), 'probability': float(v)}
                        for i, v in zip(ids[row], values[row])]
            next_ids = logits.argmax(-1)
            for row in range(len(specs)):
                if bool(alive[row]):
                    t = int(next_ids[row])
                    generated[row].append(t)
                    if t == tokenizer.eos_token_id:
                        alive[row] = False
                        results[row]['stop_reason'] = 'eos'
            if not bool(alive.any()):
                break
            next_ids = torch.where(alive, next_ids, tokenizer.pad_token_id)
            input_ids = next_ids[:, None]
            attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)
        for row, result in enumerate(results):
            result['generation'] = tokenizer.decode(generated[row], skip_special_tokens=True).strip()
            result['generated_token_ids'] = generated[row]
            result['consumed_response_tokens'] = len([a for a in result['activations'] if a['segment'] == 'response'])
            result['final_generated_token_not_probed'] = True
            summarize_case(result)
    finally:
        for handle in handles:
            handle.remove()
    return results


def execute(specs, filename):
    results = []
    for start in range(0, len(specs), P['batch_size']):
        batch = generate_batch(specs[start:start+P['batch_size']])
        results.extend(batch)
        save(filename, results)
        for r in batch:
            print('TRIAL', r['id'], 'gate', round(r['activation_summary']['maximum_gate'], 4),
                  'qualification', r['qualification_wording_detected'], 'stop', r['stop_reason'],
                  'text', r['generation'][:230].replace('\n', ' '), flush=True)
    return results


def main():
    resume = '--resume-sweep' in sys.argv
    save('runtime-sweep.json' if resume else 'runtime.json', {'python': platform.python_version(), 'torch': torch.__version__,
                          'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
                          'attention': 'unfrozen eager; separate cached batch rows',
                          'dtype': 'BF16 model; FP32 MOLT projection',
                          'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    baseline_specs = [{'id': 'clean_' + p['id'], 'prompt_id': p['id'], 'factor': 1,
                       'mode': 'target', 'scope': 'all_tokens'} for p in P['prompts']]
    if resume:
        baselines = json.loads((OUT / 'baselines.json').read_text())
        assert len(baselines) == len(P['prompts'])
        assert {r['prompt_id'] for r in baselines} == set(prompts)
    else:
        baselines = execute(baseline_specs, 'baselines.json')
    print('BASELINES_READY_FOR_REVIEW', flush=True)
    until = time.monotonic() + 900
    selection = None
    while selection is None:
        check_time()
        assert time.monotonic() < until, 'Selection review timed out'
        try:
            selection = json.loads((ROOT / 'selection.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(1)
    save('selection.json', selection)
    by_id = {r['prompt_id']: r for r in baselines}
    specs = []
    for prompt_id in selection['prompt_ids']:
        for scope in P['intervention_scopes']:
            for factor in P['factors']:
                if factor == 1:
                    continue
                specs.append({'id': f'{prompt_id}_{scope}_factor{factor}', 'prompt_id': prompt_id,
                              'factor': factor, 'scope': scope, 'mode': 'target'})
        response_active = [a for a in by_id[prompt_id]['activations']
                           if a['segment'] == 'response' and a['target_gate'] > 0]
        if response_active:
            target = max(response_active, key=lambda a: a['target_gate'])
            for factor in [0, -4]:
                specs.append({'id': f'{prompt_id}_single_position_factor{factor}', 'prompt_id': prompt_id,
                              'factor': factor, 'scope': 'single_position', 'position': target['position'],
                              'mode': 'target', 'site_selected_from_clean_gate_map': target})
        specs.append({'id': f'{prompt_id}_unrelated_factor0', 'prompt_id': prompt_id,
                      'factor': 0, 'scope': 'all_tokens', 'mode': 'unrelated'})
        for seed in [0, 1, 2]:
            specs.append({'id': f'{prompt_id}_random_factor0_seed{seed}', 'prompt_id': prompt_id,
                          'factor': 0, 'scope': 'all_tokens', 'mode': 'random_matched_norm', 'seed': seed})
    for prompt_id in ['missing_access', 'missing_information', 'coding', 'heldout_stomach', 'heldout_skin']:
        for factor in [0, -4, 4]:
            specs.append({'id': f'{prompt_id}_control_factor{factor}', 'prompt_id': prompt_id,
                          'factor': factor, 'scope': 'all_tokens', 'mode': 'target'})
    save('trial_plan.json', specs)
    trials = execute(specs, 'trials.json')
    # Recheck clean identity alone to measure batching effects on a selected prompt.
    identities = execute([{'id': 'identity_repeat_' + pid, 'prompt_id': pid, 'factor': 1,
                           'mode': 'target', 'scope': 'all_tokens'} for pid in selection['prompt_ids']],
                         'identity-repeats.json')
    checks = {r['prompt_id']: {'identical_generated_tokens':
              r['generated_token_ids'] == by_id[r['prompt_id']]['generated_token_ids']} for r in identities}
    save('completion.json', {'status': 'complete', 'baseline_count': len(baselines),
                             'trial_count': len(trials), 'identity_checks': checks,
                             'wall_seconds': time.monotonic() - START,
                             'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'attention_methods_unchanged': all(b.self_attn.forward == a for b, a in zip(model.model.layers, original_attention))})
    print('EXPERIMENT_COMPLETE', len(trials), flush=True)


if __name__ == '__main__':
    main()
