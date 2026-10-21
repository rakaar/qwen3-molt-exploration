"""Private native-Qwen MOLT steering pilot; no reconstruction replacement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / 'protocol.json').read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--strengths', type=float, nargs='+')
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'results.json').exists(), 'Use a fresh stage; never overwrite results.'
    started = time.monotonic()
    expected = json.loads((ROOT / 'expected-base-verification.json').read_text())
    assert expected['repo'] == P['base_repo'] and expected['revision'] == P['base_revision']
    for record in expected['files']:
        path = args.model_dir / record['file']
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], record['file']
    torch.set_num_threads(6)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert model.config.hidden_size == 2560 and len(model.model.layers) == 36
    assert torch.cuda.is_bf16_supported()
    native_attention = [block.self_attn.forward for block in model.model.layers]
    assets = json.loads((ROOT / 'weights/manifest.json').read_text())
    assert assets['repo'] == P['molt_repo'] and assets['revision'] == P['molt_revision']
    weights, cpu_weights = {}, {}
    for record in assets['features']:
        name = f"L{record['layer']}-T{record['transform']}"
        path = ROOT / 'weights' / record['file']
        assert sha(path) == record['sha256']
        with np.load(path) as data:
            cpu_weights[name] = {key: data[key].copy() for key in data.files}
        weights[name] = {key: torch.from_numpy(value).float().cuda()
                         for key, value in cpu_weights[name].items()}
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    strengths = args.strengths or P['strengths']
    groups = args.groups or list(P['groups'])
    cases = [case for case in P['cases'] if args.cases is None or case['id'] in args.cases]
    assert cases and (args.cases is None or len(cases) == len(args.cases))
    assert all(group in P['groups'] for group in groups)
    result = {
        'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': P, 'stage_strengths': strengths, 'stage_groups': groups,
        'stage_cases': [case['id'] for case in cases],
        'base_hashes_verified': True, 'assets': assets, 'rows': [],
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                    'python': platform.python_version(), 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name()},
        'source_sha256': {name: sha(ROOT / name) for name in
                          ['run.py', 'protocol.json', 'expected-base-verification.json']},
    }

    def generate(case, kind, group=None, strength=0):
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': case['user']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
        first_character = chat.index(case['user'])
        last_character = first_character + len(case['user'])
        user_positions = [i for i, (left, right) in enumerate(encoded['offset_mapping'])
                          if right > first_character and left < last_character and right > left]
        assert user_positions and tokenizer.decode([encoded['input_ids'][i] for i in user_positions]) == case['user']
        prompt_length = len(encoded['input_ids'])
        ids = torch.tensor([encoded['input_ids']], device='cuda')
        mask, cache = torch.ones_like(ids), None
        output_tokens, traces, state, handles, cpu_checks = [], [], {}, [], []
        active_features = P['groups'][group] if group is not None else []
        step = 0
        stop_reason = 'token_limit'

        if kind != 'native':
            for feature in P['features']:
                layer = feature['layer']
                name = f"L{layer}-T{feature['transform']}"
                w = weights[name]

                def capture(_module, inputs, name=name, w=w):
                    h = inputs[0].detach().float()
                    z = (h - w['input_mean']) / w['input_std']
                    raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
                    gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, torch.zeros_like(raw))
                    low = (z @ w['V'] @ w['U']) * w['output_std']
                    assert torch.isfinite(raw).all() and torch.isfinite(low).all()
                    state[name] = {'raw': raw, 'gate': gate, 'low': low}
                    if step == 0:
                        # Independent float64 CPU calculation checks slices, orientation and numerics.
                        cw = cpu_weights[name]
                        sample = -1
                        h_cpu = h[0, sample].cpu().double().numpy()
                        z_cpu = (h_cpu - cw['input_mean']) / cw['input_std']
                        raw_cpu = float(np.dot(z_cpu, cw['e']) + cw['bias'][0])
                        low_cpu = np.einsum('r,rd->d', np.einsum('d,dr->r', z_cpu, cw['V']), cw['U']) * cw['output_std']
                        actual_raw = float(raw[0, sample])
                        actual_low = low[0, sample].cpu().numpy()
                        assert abs(raw_cpu - actual_raw) < 3e-4 * max(1, abs(raw_cpu))
                        assert np.allclose(actual_low, low_cpu, rtol=4e-4, atol=5e-5), name
                        cpu_checks.append({'feature': name, 'raw_cpu': raw_cpu,
                            'raw_gpu': actual_raw, 'max_lowrank_abs_error': float(np.max(np.abs(actual_low - low_cpu)))})

                def edit(_module, _inputs, output, name=name, layer=layer):
                    current = state[name]
                    n = output.shape[1]
                    eligible = [i >= user_positions[0] for i in range(n)] if step == 0 else [True] * n
                    selected = [flag and name in active_features and kind == 'additive' for flag in eligible]
                    selection = torch.tensor(selected, device='cuda').reshape(1, n)
                    coefficients = selection * float(strength)
                    delta = coefficients.unsqueeze(-1) * current['low']
                    changed = None
                    if any(selected):
                        changed = (output.float() + delta).to(output.dtype)
                        assert torch.isfinite(changed).all()
                        assert torch.equal(changed[~selection], output[~selection])
                        actual = (changed.float() - output.float()).norm(dim=-1)[0].cpu().tolist()
                    else:
                        actual = [0.] * n
                    values = {key: value[0].detach().cpu().tolist() for key, value in {
                        'raw': current['raw'], 'gate': current['gate'],
                        'lowrank_norm': current['low'].norm(dim=-1),
                        'native_mlp_norm': output.float().norm(dim=-1),
                        'intended_delta_norm': delta.norm(dim=-1),
                    }.items()}
                    token_ids = ids[0].tolist()
                    for i in range(n):
                        full_position = i if step == 0 else prompt_length + step - 1
                        natural_gate = values['gate'][i]
                        traces.append({'feature': name, 'layer': layer,
                            'full_chat_position': full_position,
                            'segment': 'generated' if step else 'prompt',
                            'generated_position': step - 1 if step else None,
                            'token_id': token_ids[i], 'token': tokenizer.decode([token_ids[i]]),
                            'natural_raw': values['raw'][i], 'natural_gate': natural_gate,
                            'threshold': float(weights[name]['theta'][0]),
                            'gate_offset': float(strength) if selected[i] else 0.,
                            'effective_estimated_gate': natural_gate + (float(strength) if selected[i] else 0.),
                            'patch_applied': selected[i],
                            'lowrank_norm': values['lowrank_norm'][i],
                            'native_mlp_norm': values['native_mlp_norm'][i],
                            'intended_delta_norm': values['intended_delta_norm'][i],
                            'actual_bf16_delta_norm': actual[i]})
                    return changed

                block = model.model.layers[layer]
                handles.extend([block.post_attention_layernorm.register_forward_pre_hook(capture),
                                block.mlp.register_forward_hook(edit)])
        trial_start = time.monotonic()
        try:
            for step in range(P['decoding']['max_new_tokens']):
                assert time.monotonic() - started < P['max_runtime_minutes'] * 60, 'Experiment runtime bound reached'
                prediction = model(input_ids=ids, attention_mask=mask,
                    past_key_values=cache, use_cache=True)
                cache = prediction.past_key_values
                token = int(prediction.logits[0, -1].argmax())
                output_tokens.append(token)
                if token in eos:
                    stop_reason = 'eos'
                    break
                ids = torch.tensor([[token]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
        finally:
            for handle in handles:
                handle.remove()
        assert all(block.self_attn.forward == before
                   for block, before in zip(model.model.layers, native_attention))
        if kind != 'native':
            assert len(traces) == len(P['features']) * (prompt_length + len(output_tokens) - 1)
        answer = tokenizer.decode(output_tokens, skip_special_tokens=True)
        trial_id = f"{case['id']}__{kind}" if group is None else f"{case['id']}__{group}__plus{strength:g}"
        trace_file = trial_id + '.trace.json'
        if traces:
            write_json(out / trace_file, traces)
        summary = {}
        for feature in P['features']:
            name = f"L{feature['layer']}-T{feature['transform']}"
            sample = [row for row in traces if row['feature'] == name]
            applied = [row for row in sample if row['patch_applied']]
            summary[name] = {'processed_tokens': len(sample), 'edited_tokens': len(applied),
                'positive_natural_gate_tokens': sum(row['natural_gate'] > 0 for row in sample),
                'max_natural_gate': max((row['natural_gate'] for row in sample), default=0),
                'mean_actual_edit_norm': float(np.mean([row['actual_bf16_delta_norm'] for row in applied])) if applied else 0,
                'max_actual_edit_norm': max((row['actual_bf16_delta_norm'] for row in applied), default=0),
                'mean_edit_to_native_norm_ratio': float(np.mean([row['actual_bf16_delta_norm'] / max(row['native_mlp_norm'], 1e-12) for row in applied])) if applied else 0}
        return {'id': trial_id, 'case_id': case['id'], 'question': case['user'],
            'kind': kind, 'group': group, 'strength': strength, 'answer': answer,
            'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
            'output_token_ids': output_tokens, 'stop_reason': stop_reason,
            'elapsed_seconds': time.monotonic() - trial_start,
            'prompt_chat': chat, 'prompt_token_ids': encoded['input_ids'],
            'user_positions': user_positions, 'cpu_checks': cpu_checks,
            'feature_summary': summary, 'trace_file': trace_file if traces else None,
            'trace_sha256': sha(out / trace_file) if traces else None}

    for case in cases:
        baseline = generate(case, 'native')
        result['rows'].append(baseline)
        observed = generate(case, 'observe')
        assert baseline['output_token_ids'] == observed['output_token_ids'], 'Observation-only hooks changed generation'
        observed['tokens_identical_to_native'] = True
        result['rows'].append(observed)
        write_json(out / 'results.json', result)
        print(json.dumps({'case': case['id'], 'baseline': baseline['answer'], 'baseline_tokens': len(baseline['output_token_ids'])}), flush=True)
        for strength in strengths:
            for group in groups:
                row = generate(case, 'additive', group, strength)
                row['tokens_identical_to_native'] = row['output_token_ids'] == baseline['output_token_ids']
                result['rows'].append(row)
                write_json(out / 'results.json', result)
                print(json.dumps({'id': row['id'], 'answer': row['answer'], 'stop_reason': row['stop_reason'],
                    'tokens': len(row['output_token_ids']), 'elapsed_seconds': row['elapsed_seconds']}, ensure_ascii=False), flush=True)
    result.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - started, attention_unmodified=True,
        observation_controls_passed=True)
    write_json(out / 'results.json', result)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
