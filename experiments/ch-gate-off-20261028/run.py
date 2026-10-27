"""Input-word-only contribution ablation of L3-T2334 in a resident Qwen3 model."""
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def run_experiment(model, tok, weights):
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'results.json').exists(), 'Inspect previous results before retrying.'
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    cases = json.loads((ROOT / 'cases.json').read_text())
    assert protocol['layer'] == 3 and protocol['transform'] == 2334
    assert len(cases) == 12
    assert torch.__version__ == protocol['runtime']['torch']
    assert transformers.__version__ == protocol['runtime']['transformers']
    assert not model.training and next(model.parameters()).dtype == torch.bfloat16
    assert all(len(m._forward_hooks) + len(m._forward_pre_hooks) == 0 for m in model.modules())
    assert weights['U'].shape == (32, 2560) and weights['V'].shape == (2560, 32)
    assert all(t.dtype == torch.float32 and t.is_cuda for t in weights.values())
    expected_slice = json.loads((ROOT / 'selected-transform.json').read_text())
    assert expected_slice['transform'] == 2334
    assert sha(ROOT / expected_slice['file']) == expected_slice['sha256']
    for key, tensor in weights.items():
        with np.load(ROOT / expected_slice['file'], allow_pickle=False) as data:
            assert np.array_equal(tensor.cpu().numpy(), data[key]), key
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    block = model.model.layers[3]
    attention_functions = [layer.self_attn.forward for layer in model.model.layers]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    started = time.monotonic()
    result = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': protocol, 'cases': [], 'source_sha256': {
                  name: sha(ROOT / name) for name in ['run.py', 'protocol.json', 'cases.json', 'selected-transform.json']},
              'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                          'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
                          'cuda': torch.version.cuda}, 'transform_slice_verified_against_resident_tensors': True}

    def deadline():
        assert time.monotonic() - started < protocol['max_runtime_minutes'] * 60

    def next_stats(logits, goal_ids):
        logits = logits.float()
        log_probs = logits.log_softmax(-1)
        values, indices = log_probs.exp().topk(6)
        return {'target_probability_mass': float(log_probs[goal_ids].exp().sum()),
                'target_tokens': [{'token_id': i, 'text': tok.decode([i]), 'logit': float(logits[i]),
                                   'probability': float(log_probs[i].exp())} for i in goal_ids],
                'logsumexp': float(logits.logsumexp(-1)),
                'top_tokens': [{'token_id': int(i), 'text': tok.decode([int(i)]), 'probability': float(p)}
                               for i, p in zip(indices, values)]}

    @torch.inference_mode()
    def run(case, condition, forced=None):
        deadline()
        chat = tok.apply_chat_template([{'role': 'user', 'content': case['user']}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        user_start = chat.index(case['user'])
        start, end = [user_start + x for x in case['target_span']]
        assert chat[start:end] == case['word']
        positions = [i for i, (a, b) in enumerate(enc['offset_mapping'])
                     if b > a and b > start and a < end]
        assert positions
        prompt_ids = enc['input_ids']
        goal_variants = [case['expected'], ' ' + case['expected']]
        goal_encodings = [tok.encode(s, add_special_tokens=False) for s in goal_variants]
        assert all(len(x) == 1 for x in goal_encodings), goal_encodings
        goal_ids = sorted({x[0] for x in goal_encodings})
        ids = torch.tensor([prompt_ids], device='cuda')
        mask = torch.ones_like(ids)
        cache = None
        state = {'prefill': True}
        handles, edits, scores, generated = [], [], [], []
        fraction = {'native': None, 'zero_control': 0.0, 'gate_off': 1.0}[condition]
        vector_file = None

        def observe_input(_module, inputs):
            if not state['prefill']:
                return None
            residual = inputs[0].detach().float()[0, positions]
            z = (residual - weights['input_mean']) / weights['input_std']
            raw = F.linear(z, weights['e'].reshape(1, -1), weights['bias']).squeeze(-1)
            independent = (z * weights['e']).sum(-1) + weights['bias'][0]
            assert torch.allclose(raw, independent, atol=3e-5, rtol=3e-5)
            gate = torch.where((raw > weights['theta'][0]) & (raw > 0), raw, 0.0)
            operator = (z @ weights['V'] @ weights['U']) * weights['output_std']
            state.update(z=z, raw=raw, gate=gate, operator=operator)
            return None

        def edit_output(_module, _inputs, output):
            nonlocal vector_file
            if not state['prefill']:
                return None
            assert output.shape[1] == len(prompt_ids)
            delta = -fraction * state['gate'].unsqueeze(-1) * state['operator']
            changed = output.clone()
            changed[0, positions] = (output[0, positions].float() + delta).to(output.dtype)
            actual = changed[0, positions].float() - output[0, positions].float()
            other = torch.ones(output.shape[1], device=output.device, dtype=torch.bool)
            other[positions] = False
            assert torch.equal(output[:, other], changed[:, other])
            if fraction == 0 or bool((state['gate'] == 0).all()):
                assert torch.equal(output, changed)
            for j, position in enumerate(positions):
                edits.append({'layer': 3, 'transform': 2334, 'position': position,
                              'token_id': prompt_ids[position], 'token': tok.decode([prompt_ids[position]]),
                              'raw': float(state['raw'][j]), 'threshold': float(weights['theta'][0]),
                              'natural_gate': float(state['gate'][j]),
                              'effective_estimated_gate': (1 - fraction) * float(state['gate'][j]),
                              'operator_norm': float(state['operator'][j].norm()),
                              'intended_delta_norm': float(delta[j].norm()),
                              'actual_bf16_delta_norm': float(actual[j].norm()),
                              'native_mlp_norm': float(output[0, position].float().norm()),
                              'max_bf16_rounding_error': float((actual[j] - delta[j]).abs().max()),
                              'all_other_positions_identical_at_edit_boundary': True})
            if forced is None:
                vector_file = f"{case['id']}--{condition}.npz"
                np.savez_compressed(OUT / vector_file,
                                    positions=np.array(positions), z=state['z'].cpu().numpy(),
                                    raw=state['raw'].cpu().numpy(), gate=state['gate'].cpu().numpy(),
                                    operator=state['operator'].cpu().numpy(),
                                    native_mlp=output[0, positions].float().cpu().numpy(),
                                    edited_mlp=changed[0, positions].float().cpu().numpy(),
                                    intended_delta=delta.cpu().numpy(), actual_delta=actual.cpu().numpy())
            return changed

        if condition != 'native':
            handles = [block.post_attention_layernorm.register_forward_pre_hook(observe_input),
                       block.mlp.register_forward_hook(edit_output)]
        stop = 'token_limit'
        try:
            limit = len(forced) if forced is not None else protocol['max_new_tokens']
            for step in range(limit):
                deadline()
                state['prefill'] = step == 0
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                logits = pred.logits[0, -1].float()
                greedy = int(logits.argmax())
                token = int(forced[step]) if forced is not None else greedy
                stat = next_stats(logits, goal_ids)
                log_probs = logits.log_softmax(-1)
                stat.update(output_position=step, selected_token_id=token, selected_text=tok.decode([token]),
                            selected_token_probability=float(log_probs[token].exp()),
                            greedy_token_id=greedy, greedy_text=tok.decode([greedy]))
                scores.append(stat)
                generated.append(token)
                if token in eos:
                    stop = 'eos'
                    break
                ids = torch.tensor([[token]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            assert all(b.self_attn.forward == f for b, f in zip(model.model.layers, attention_functions))
            assert len(edits) == (len(positions) if condition != 'native' else 0)
        finally:
            for handle in handles:
                handle.remove()
        assert not block.mlp._forward_hooks and not block.post_attention_layernorm._forward_pre_hooks
        return {'condition': condition, 'forced_baseline_continuation': forced is not None,
                'chat': chat, 'prompt_token_ids': prompt_ids, 'target_positions': positions,
                'target_tokens': [tok.decode([prompt_ids[i]]) for i in positions],
                'goal_token_variants': dict(zip(goal_variants, [x[0] for x in goal_encodings])),
                'output_token_ids': generated, 'answer': tok.decode(generated, skip_special_tokens=True),
                'stop_reason': stop, 'steps': scores, 'edits': edits, 'vector_file': vector_file,
                'vector_sha256': sha(OUT / vector_file) if vector_file else None,
                'generated_positions_edited': 0, 'attention_functions_unmodified': True}

    def answer_slot(native):
        text = native['answer']
        match = re.search(r"(?<![A-Za-z])([A-Za-z]{2})[ \t\r\n.!?'\"`]*$", text)
        if match is None:
            return {'index': 0, 'prefix': '', 'method': 'first_output_fallback_no_final_two_letter_answer',
                    'native_answer_atom': None}
        start, end = match.span(1)
        ids = native['output_token_ids']
        for j in range(len(ids)):
            left = len(tok.decode(ids[:j], skip_special_tokens=True))
            right = len(tok.decode(ids[:j + 1], skip_special_tokens=True))
            if right > start and left < end:
                return {'index': j, 'prefix': tok.decode(ids[:j], skip_special_tokens=True),
                        'method': 'first_token_overlapping_final_two_letter_answer',
                        'native_answer_atom': match.group(1), 'native_piece': tok.decode([ids[j]])}
        raise AssertionError('Native answer character span did not map to a token.')

    try:
        with torch.inference_mode():
            for case in cases:
                native = run(case, 'native')
                zero = run(case, 'zero_control')
                assert zero['output_token_ids'] == native['output_token_ids']
                assert zero['steps'] == native['steps'], 'Zero-strength hook changed next-token statistics.'
                ablated = run(case, 'gate_off')
                forced = run(case, 'gate_off', forced=native['output_token_ids'])
                for key in ['target_probability_mass', 'target_tokens', 'logsumexp', 'top_tokens',
                            'greedy_token_id', 'greedy_text', 'output_position']:
                    assert forced['steps'][0][key] == ablated['steps'][0][key], key
                assert forced['edits'] == ablated['edits']
                slot = answer_slot(native)
                k = slot['index']
                base_p = native['steps'][k]['target_probability_mass']
                off_p = forced['steps'][k]['target_probability_mass']
                if all(e['natural_gate'] == 0 for e in ablated['edits']):
                    assert ablated['output_token_ids'] == native['output_token_ids']
                    assert forced['steps'] == native['steps']
                summary = {'case_id': case['id'], 'word': case['word'], 'letter_case': case['letter_case'],
                           'group': case['group'], 'native_answer': native['answer'],
                           'gate_off_answer': ablated['answer'], 'answer_slot': slot,
                           'native_p_answer_slot': base_p, 'gate_off_p_answer_slot': off_p,
                           'answer_slot_change_pp': 100 * (off_p - base_p),
                           'native_p_first_output': native['steps'][0]['target_probability_mass'],
                           'gate_off_p_first_output': ablated['steps'][0]['target_probability_mass'],
                           'target_tokens': ablated['target_tokens'],
                           'natural_gates': [e['natural_gate'] for e in ablated['edits']],
                           'actual_edit_norms': [e['actual_bf16_delta_norm'] for e in ablated['edits']],
                           'zero_control_exact': True}
                result['cases'].append({'case': case, 'summary': summary, 'native': native,
                                        'zero_control': zero, 'gate_off_generation': ablated,
                                        'gate_off_baseline_prefix': forced})
                save(OUT / 'results.json', result)
                print('RESULT', json.dumps(summary, ensure_ascii=False), flush=True)
        result.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started,
                      only_edited_feature={'layer': 3, 'transform': 2334},
                      hooks_remaining=sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()),
                      all_zero_controls_exact=True, all_edits_confined_to_target_input_word=True)
        assert result['hooks_remaining'] == 0 and len(result['cases']) == 12
        save(OUT / 'results.json', result)
        print('COMPLETE', len(result['cases']), 'cases; only L3-T2334 edited.', flush=True)
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save(OUT / 'results.json', result)
        raise
    return result
