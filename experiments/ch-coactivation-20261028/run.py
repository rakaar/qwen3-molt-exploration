"""Scan native coactivation, then ablate selected contributions on eligible prompts."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def answer_slot(native, tokenizer):
    match = re.search(r'''(?<![A-Za-z])([A-Za-z]{2})[ \t\r\n.!?'"`]*$''', native['answer'])
    if not match:
        return {'index': 0, 'prefix': '', 'method': 'first_output_fallback'}
    start, end = match.span(1)
    ids = native['output_token_ids']
    for j in range(len(ids)):
        a = len(tokenizer.decode(ids[:j], skip_special_tokens=True))
        b = len(tokenizer.decode(ids[:j+1], skip_special_tokens=True))
        if b > start and a < end:
            return {'index': j, 'prefix': tokenizer.decode(ids[:j], skip_special_tokens=True),
                    'method': 'first_token_overlapping_final_two_letter_answer',
                    'native_answer_atom': match.group(1)}
    raise AssertionError('Answer span not mapped to output tokens.')


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'results.json').exists(), 'Inspect previous results before retrying.'
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    cases = json.loads((ROOT / 'cases.json').read_text())
    expected = json.loads((ROOT / 'baseline-reference.json').read_text())
    base = json.loads((ROOT / 'expected-base-verification.json').read_text())
    assert len(cases) == 64 and base['revision'] == protocol['base_revision']
    assert torch.__version__ == protocol['runtime']['torch'], torch.__version__
    assert transformers.__version__ == protocol['runtime']['transformers']
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for row in base['files']:
        path = ROOT / 'model' / row['file']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / 'model', local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').to('cuda').eval().requires_grad_(False)
    assert model.config.model_type == 'qwen3' and model.config.num_hidden_layers == 36
    assert model.config.hidden_size == 2560
    weights = {}
    for row in protocol['features']:
        path = ROOT / 'weights' / row['file']
        assert sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            weights[row['layer']] = {k: torch.from_numpy(arrays[k].copy()).float().cuda()
                                     for k in arrays.files}
        w = weights[row['layer']]
        assert w['U'].shape == (32, 2560) and w['V'].shape == (2560, 32)
        assert all(torch.isfinite(x).all() for x in w.values())
        assert (w['input_std'] > 0).all() and (w['output_std'] > 0).all()
    assert set(weights) == {3, 29}
    attention_functions = [block.self_attn.forward for block in model.model.layers]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    started = time.monotonic()
    result = {'status': 'running', 'phase': 'native_scan', 'protocol': protocol,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'scan': [], 'ablations': [],
              'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                          'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
                          'cuda': torch.version.cuda, 'dtype': 'bfloat16', 'base_files_verified': len(base['files'])},
              'source_sha256': {name: sha(ROOT / name) for name in
                                ['run.py', 'protocol.json', 'cases.json', 'expected-base-verification.json']}}

    def deadline():
        assert time.monotonic() - started < protocol['max_runtime_minutes'] * 60

    def token_ids(variants):
        encodings = [tokenizer.encode(s, add_special_tokens=False) for s in variants]
        assert all(len(ids) == 1 for ids in encodings), (variants, encodings)
        return sorted({ids[0] for ids in encodings})

    def stats(logits, goal_ids, expected_ids):
        logits = logits.float()
        lp = logits.log_softmax(-1)
        p = lp.exp()
        values, indices = p.topk(6)
        log_mass = float(lp[goal_ids].logsumexp(-1))
        other_mask = torch.ones_like(lp, dtype=torch.bool)
        other_mask[goal_ids] = False
        log_other = float(lp[other_mask].logsumexp(-1))
        return {'target_probability_mass': float(p[goal_ids].sum()),
                'expected_prefix_probability_mass': float(p[expected_ids].sum()),
                'target_log_probability_mass': log_mass,
                'target_log_odds_vs_all_other_tokens': log_mass - log_other,
                'target_tokens': [{'token_id': i, 'text': tokenizer.decode([i]),
                                   'logit': float(logits[i]), 'probability': float(p[i])} for i in goal_ids],
                'logsumexp': float(logits.logsumexp(-1)),
                'top_tokens': [{'token_id': int(i), 'text': tokenizer.decode([int(i)]), 'probability': float(v)}
                               for i, v in zip(indices, values)]}

    @torch.inference_mode()
    def run(case, condition, forced=None):
        deadline()
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': case['user']}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
        user_start = chat.index(case['user'])
        start, end = [user_start + x for x in case['target_span']]
        assert chat[start:end] == case['word']
        positions = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                     if b > a and b > start and a < end]
        assert positions
        prompt_ids = encoded['input_ids']
        ch = 'CH' if case['letter_case'] == 'upper' else 'ch'
        goal_ids = token_ids([ch, ' ' + ch])
        expected_ids = token_ids([case['expected'], ' ' + case['expected']])
        fractions = {'zero_control': {3: 0., 29: 0.}, 'L3_off': {3: 1., 29: 0.},
                     'L29_off': {3: 0., 29: 1.}, 'both_off': {3: 1., 29: 1.}}
        state = {'prefill': True}
        handles, edits, vectors, scores, generated = [], [], [], [], []

        def input_observer(layer):
            def observe(_module, inputs):
                if not state['prefill']:
                    return None
                w = weights[layer]
                residual = inputs[0].detach().float()[0, positions]
                z = (residual - w['input_mean']) / w['input_std']
                raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
                independent = (z * w['e']).sum(-1) + w['bias'][0]
                assert torch.allclose(raw, independent, atol=3e-5, rtol=3e-5)
                gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, 0.)
                operator = (z @ w['V'] @ w['U']) * w['output_std']
                state[layer] = {'z': z, 'raw': raw, 'gate': gate, 'operator': operator}
            return observe

        def output_editor(layer):
            def edit(_module, _inputs, output):
                if not state['prefill']:
                    return None
                assert output.shape[1] == len(prompt_ids)
                s, w = state[layer], weights[layer]
                fraction = fractions[condition][layer]
                delta = -fraction * s['gate'].unsqueeze(-1) * s['operator']
                changed = output.clone()
                changed[0, positions] = (output[0, positions].float() + delta).to(output.dtype)
                actual = changed[0, positions].float() - output[0, positions].float()
                other = torch.ones(output.shape[1], device=output.device, dtype=torch.bool)
                other[positions] = False
                assert torch.equal(output[:, other], changed[:, other])
                if fraction == 0. or bool((s['gate'] == 0).all()):
                    assert torch.equal(output, changed)
                for j, position in enumerate(positions):
                    edits.append({'layer': layer, 'transform': 2334, 'position': position,
                                  'token': tokenizer.decode([prompt_ids[position]]),
                                  'raw': float(s['raw'][j]), 'threshold': float(w['theta'][0]),
                                  'gate': float(s['gate'][j]), 'removed_fraction': fraction,
                                  'operator_norm': float(s['operator'][j].norm()),
                                  'intended_delta_norm': float(delta[j].norm()),
                                  'actual_bf16_delta_norm': float(actual[j].norm()),
                                  'all_other_positions_identical_at_edit_boundary': True})
                if forced is None:
                    file = f"{case['id']}--{condition}--L{layer}.npz"
                    np.savez_compressed(OUT / file, positions=np.array(positions),
                                        z=s['z'].cpu().numpy(), raw=s['raw'].cpu().numpy(),
                                        gate=s['gate'].cpu().numpy(), operator=s['operator'].cpu().numpy(),
                                        native_mlp=output[0, positions].float().cpu().numpy(),
                                        edited_mlp=changed[0, positions].float().cpu().numpy(),
                                        intended_delta=delta.cpu().numpy(), actual_delta=actual.cpu().numpy())
                    vectors.append({'file': file, 'sha256': sha(OUT / file), 'layer': layer,
                                    'removed_fraction': fraction})
                return changed
            return edit

        if condition != 'native':
            for layer in (3, 29):
                block = model.model.layers[layer]
                handles.extend([block.post_attention_layernorm.register_forward_pre_hook(input_observer(layer)),
                                block.mlp.register_forward_hook(output_editor(layer))])
        ids = torch.tensor([prompt_ids], device='cuda')
        mask, cache = torch.ones_like(ids), None
        stop = 'token_limit'
        try:
            for step in range(len(forced) if forced is not None else protocol['max_new_tokens']):
                deadline()
                state['prefill'] = step == 0
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                logits = pred.logits[0, -1].float()
                greedy = int(logits.argmax())
                token = int(forced[step]) if forced is not None else greedy
                row = stats(logits, goal_ids, expected_ids)
                row.update(output_position=step, selected_token_id=token,
                           selected_token_probability=float(logits.log_softmax(-1)[token].exp()),
                           greedy_token_id=greedy, greedy_text=tokenizer.decode([greedy]))
                scores.append(row)
                generated.append(token)
                if token in eos:
                    stop = 'eos'
                    break
                ids = torch.tensor([[token]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            assert len(edits) == (2 * len(positions) if condition != 'native' else 0)
            assert all(block.self_attn.forward == f for block, f in zip(model.model.layers, attention_functions))
        finally:
            for handle in handles:
                handle.remove()
        assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()) == 0
        return {'condition': condition, 'forced_baseline_continuation': forced is not None,
                'prompt_token_ids': prompt_ids, 'target_positions': positions,
                'target_tokens': [tokenizer.decode([prompt_ids[i]]) for i in positions],
                'goal_token_ids': goal_ids, 'expected_prefix_token_ids': expected_ids,
                'output_token_ids': generated, 'answer': tokenizer.decode(generated, skip_special_tokens=True),
                'steps': scores, 'edits': edits, 'vectors': vectors, 'stop_reason': stop,
                'generated_positions_edited': 0, 'attention_unmodified': True}

    try:
        for case in cases:
            native = run(case, 'native')
            zero = run(case, 'zero_control')
            assert native['output_token_ids'] == zero['output_token_ids']
            assert native['steps'] == zero['steps'], 'Zero hooks changed native output statistics.'
            reference_exact = None
            if case['id'] in expected:
                ref = expected[case['id']]
                assert native['prompt_token_ids'] == ref['prompt_token_ids']
                assert native['output_token_ids'] == ref['output_token_ids']
                reference_exact = all(native['steps'][i][k] == old[k] for i, old in enumerate(ref['steps'])
                                      for k in ['target_probability_mass', 'target_tokens', 'logsumexp',
                                                'selected_token_probability', 'greedy_token_id'])
                assert reference_exact, f"Previous baseline changed: {case['id']}"
            active = {str(layer): any(e['gate'] > 0 for e in zero['edits'] if e['layer'] == layer)
                      for layer in (3, 29)}
            same_token = sorted(set(e['position'] for e in zero['edits'] if e['layer'] == 3 and e['gate'] > 0)
                                & set(e['position'] for e in zero['edits'] if e['layer'] == 29 and e['gate'] > 0))
            summary = {'case_id': case['id'], 'word': case['word'], 'letter_case': case['letter_case'],
                       'group': case['group'], 'active_L3': active['3'], 'active_L29': active['29'],
                       'both_active': all(active.values()), 'both_active_same_token': bool(same_token),
                       'same_token_positions': same_token,
                       'eligible_for_ablation': case['group'] == 'ch_words' and all(active.values()),
                       'native_answer': native['answer'], 'zero_control_exact': True,
                       'previous_native_baseline_exact': reference_exact, 'answer_slot': answer_slot(native, tokenizer),
                       'gates': [{k: e[k] for k in ['layer', 'position', 'token', 'raw', 'threshold', 'gate']}
                                 for e in zero['edits']]}
            result['scan'].append({'case': case, 'summary': summary, 'native': native, 'zero_control': zero})
            save(OUT / 'results.json', result)
            print('SCAN', json.dumps(summary), flush=True)
        eligible = [row for row in result['scan'] if row['summary']['eligible_for_ablation']]
        result.update(phase='eligible_ablations', scan_complete=True,
                      eligible_case_ids=[row['case']['id'] for row in eligible])
        save(OUT / 'scan-summary.json', {'protocol': protocol, 'scan': [x['summary'] for x in result['scan']],
                                       'eligible_case_ids': result['eligible_case_ids']})
        save(OUT / 'results.json', result)
        print('SCAN_COMPLETE', len(eligible), 'eligible of 48 CH prompts.', flush=True)
        for scanned in eligible:
            case, native = scanned['case'], scanned['native']
            slot = scanned['summary']['answer_slot']
            k = slot['index']
            ablation = {'case': case, 'native_summary': scanned['summary'], 'conditions': {}}
            for condition in ['L3_off', 'L29_off', 'both_off']:
                free = run(case, condition)
                forced = run(case, condition, forced=native['output_token_ids'])
                assert free['edits'] == forced['edits']
                for key in ['target_probability_mass', 'expected_prefix_probability_mass',
                            'target_log_probability_mass', 'target_log_odds_vs_all_other_tokens',
                            'target_tokens', 'logsumexp', 'top_tokens', 'greedy_token_id']:
                    assert free['steps'][0][key] == forced['steps'][0][key], key
                base_p = native['steps'][k]['target_probability_mass']
                off_p = forced['steps'][k]['target_probability_mass']
                summary = {'condition': condition, 'native_answer': native['answer'], 'answer': free['answer'],
                           'greedy_response_changed': native['output_token_ids'] != free['output_token_ids'],
                           'answer_slot': slot, 'native_p_answer_slot': base_p, 'p_answer_slot': off_p,
                           'answer_slot_change_pp': 100 * (off_p - base_p),
                           'native_p_first_output': native['steps'][0]['target_probability_mass'],
                           'p_first_output': forced['steps'][0]['target_probability_mass'],
                           'native_log_odds_answer_slot': native['steps'][k]['target_log_odds_vs_all_other_tokens'],
                           'log_odds_answer_slot': forced['steps'][k]['target_log_odds_vs_all_other_tokens'],
                           'gates': [{key: e[key] for key in ['layer', 'position', 'token', 'raw', 'threshold',
                                                             'gate', 'removed_fraction', 'actual_bf16_delta_norm']}
                                     for e in free['edits']]}
                ablation['conditions'][condition] = {'summary': summary, 'generation': free,
                                                     'same_native_prefix': forced}
                print('ABLATION', case['id'], json.dumps(summary), flush=True)
            # Earlier-layer edits are identical with or without the later-layer edit.
            a = ablation['conditions']['L3_off']['generation']['edits']
            b = ablation['conditions']['both_off']['generation']['edits']
            assert [e for e in a if e['layer'] == 3] == [e for e in b if e['layer'] == 3]
            # The L29 state is measured before its own edit, downstream of the same L3 intervention.
            for left, right in zip([e for e in a if e['layer'] == 29], [e for e in b if e['layer'] == 29]):
                assert all(left[key] == right[key] for key in ['position', 'raw', 'gate', 'operator_norm'])
            # L29-only editing preserves the naturally observed L3/L29 input states.
            for left, right in zip(scanned['zero_control']['edits'],
                                   ablation['conditions']['L29_off']['generation']['edits']):
                assert all(left[key] == right[key] for key in ['layer', 'position', 'raw', 'gate', 'operator_norm'])
            result['ablations'].append(ablation)
            save(OUT / 'results.json', result)
        result.update(status='complete', phase='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started, scan_prompts=len(cases),
                      eligible_prompts=len(eligible), intervention_comparisons=3 * len(eligible),
                      hooks_remaining=sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()),
                      all_zero_controls_exact=True, attention_unmodified=True, generated_positions_edited=0,
                      child_feature_manipulated=False)
        assert result['hooks_remaining'] == 0
        save(OUT / 'results.json', result)
        print('COMPLETE', result['eligible_prompts'], 'eligible prompts;',
              result['intervention_comparisons'], 'intervention comparisons.', flush=True)
        return model, tokenizer, weights, result
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        save(OUT / 'results.json', result)
        raise


if __name__ == '__main__':
    model, tokenizer, features, results = main()
