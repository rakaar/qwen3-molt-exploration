"""Subtract the current Child MOLT contribution at input and generated child tokens."""
import argparse
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
P = json.loads((ROOT / 'protocol.json').read_text())
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
START = time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    args = ap.parse_args()
    assert not (OUT / 'results.json').exists(), 'Inspect existing results before retrying.'
    expected = json.loads((ROOT / 'expected-base-verification.json').read_text())
    assert expected['repo'] == P['base_repo'] and expected['revision'] == P['base_revision']
    for item in expected['files']:
        path = args.model_dir / item['file']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['file']
    save('base-verification.json', expected)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.__version__ == P['runtime']['torch']
    assert transformers.__version__ == P['runtime']['transformers']
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
    native_attentions = [b.self_attn.forward for b in model.model.layers]
    block = model.model.layers[3]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    cases = json.loads((ROOT / 'cases.json').read_text())
    assert len(cases) == 2
    assets = json.loads((ROOT / 'weights/manifest.json').read_text())
    assert assets['revision'] == P['molt_revision']
    weights = {}
    for feature in assets['features']:
        path = ROOT / 'weights' / feature['file']
        assert sha(path) == feature['sha256']
        with np.load(path) as arrays:
            weights[feature['transform']] = {k: torch.from_numpy(arrays[k].copy()).float().cuda() for k in arrays.files}
    assert set(weights) == {1372, 2334}
    result = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': P, 'assets': assets, 'rows': [],
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                    'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
                    'cuda': torch.version.cuda},
        'source_sha256': {n: sha(ROOT / n) for n in ['run.py', 'protocol.json', 'cases.json']}}

    def is_child(token):
        return tok.decode([token]).strip().casefold() == 'child'

    @torch.inference_mode()
    def run(case, condition, forced=None):
        chat = tok.apply_chat_template([{'role': 'user', 'content': case['user']}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        assert chat == case['chat'] and enc['input_ids'] == case['prompt_token_ids']
        start = chat.index(case['user']) + case['target_span'][0]
        end = chat.index(case['user']) + case['target_span'][1]
        positions = [i for i, (a, b) in enumerate(enc['offset_mapping']) if b > a and b > start and a < end]
        assert positions == case['target_positions'] == [27]
        assert tok.decode([enc['input_ids'][27]]) == 'child'
        target = positions[0]
        prompt_len = len(enc['input_ids'])
        ids = torch.tensor([enc['input_ids']], device='cuda')
        mask = torch.ones_like(ids)
        cache = None
        state = {'prefill': True, 'offset': 0}
        traces, edits, scoring, token_ids, vector_files = [], [], [], [], []
        handles = []
        hooked = condition['id'] != 'native' or forced is not None

        def eligible(j, token):
            if condition['feature'] is None:
                return False
            if state['prefill']:
                return state['offset'] + j == target
            return condition['edit_scope'] == 'input_and_generated' and is_child(token)

        def observe_input(_module, inputs):
            state['features'] = {}
            for feature, w in weights.items():
                z = (inputs[0].detach().float() - w['input_mean']) / w['input_std']
                raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
                independent = (z * w['e']).sum(-1) + w['bias'][0]
                assert torch.allclose(raw, independent, atol=3e-5, rtol=3e-5)
                gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, torch.zeros_like(raw))
                low = (z @ w['V'] @ w['U']) * w['output_std']
                state['features'][feature] = {'z': z, 'raw': raw, 'gate': gate, 'low': low}

        def edit_and_observe_output(_module, _inputs, output):
            assert state['features']
            tokens = ids[0].tolist()
            js = [j for j, token in enumerate(tokens) if eligible(j, token)]
            for j, token in enumerate(tokens):
                full_position = state['offset'] + j
                if state['prefill'] and full_position not in [target, prompt_len - 1]:
                    continue
                traces.append({'segment': 'prompt' if state['prefill'] else 'response',
                    'position': full_position, 'token_id': token, 'token': tok.decode([token]),
                    'target_input': state['prefill'] and full_position == target,
                    'matches_child_token': is_child(token), 'edit_eligible': j in js,
                    'features': {f'L3-T{f}': {'raw': float(v['raw'][0,j]), 'gate': float(v['gate'][0,j]),
                        'lowrank_norm': float(v['low'][0,j].norm()),
                        'contribution_norm': float((v['gate'][0,j] * v['low'][0,j]).norm())}
                        for f, v in state['features'].items()}})
            if not js:
                return None
            changed = output.clone()
            components = state['features'][condition['feature']]
            for j in js:
                gate, low = components['gate'][0,j], components['low'][0,j]
                delta = -condition['removal_fraction'] * gate * low
                changed[0,j] = (output[0,j].float() + delta).to(output.dtype)
                actual = changed[0,j].float() - output[0,j].float()
                full_position = state['offset'] + j
                if condition['removal_fraction'] == 0:
                    assert torch.equal(output[0,j], changed[0,j])
                if state['prefill'] and condition['removal_fraction'] == 1:
                    assert float(gate) > 0 and float(actual.norm()) > 0
                edit = {'feature': condition['feature'], 'segment': 'prompt' if state['prefill'] else 'response',
                    'position': full_position, 'token_id': tokens[j], 'token': tok.decode([tokens[j]]),
                    'natural_gate': float(gate), 'removed_gate': condition['removal_fraction'] * float(gate),
                    'effective_estimated_gate': (1-condition['removal_fraction']) * float(gate),
                    'lowrank_norm': float(low.norm()), 'intended_delta_norm': float(delta.norm()),
                    'actual_bf16_delta_norm': float(actual.norm()), 'native_mlp_norm': float(output[0,j].float().norm()),
                    'other_positions_at_intervention_boundary_identical': True,
                    'max_bf16_rounding_error': float((actual-delta).abs().max())}
                edits.append(edit)
                if forced is None:
                    name = f"{case['id']}-{condition['id']}-pos{full_position}-vectors.npz"
                    assert not (OUT/name).exists()
                    np.savez_compressed(OUT/name, z=components['z'][0,j].cpu().numpy(),
                        lowrank=low.cpu().numpy(), gate=np.array([float(gate)], dtype=np.float32),
                        native_mlp=output[0,j].float().cpu().numpy(),
                        edited_mlp=changed[0,j].float().cpu().numpy(),
                        intended_delta=delta.cpu().numpy(), actual_delta=actual.cpu().numpy())
                    vector_files.append({'file': name, 'sha256': sha(OUT/name), 'position': full_position})
            other = torch.ones(output.shape[1], device=output.device, dtype=torch.bool)
            other[js] = False
            assert torch.equal(output[:,other], changed[:,other])
            return changed

        if hooked:
            handles = [block.post_attention_layernorm.register_forward_pre_hook(observe_input),
                       block.mlp.register_forward_hook(edit_and_observe_output)]
        stop, first_top = 'token_limit', None
        limit = len(forced) if forced is not None else P['decoding']['max_new_tokens']
        try:
            for step in range(limit):
                assert time.monotonic() - START < P['max_runtime_minutes'] * 60
                state.update(prefill=step==0, offset=0 if step==0 else prompt_len+step-1)
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                logits = pred.logits[0,-1].float()
                log_probs = logits.log_softmax(-1)
                greedy = int(logits.argmax())
                token = int(forced[step]) if forced is not None else greedy
                if step == 0:
                    values, indices = log_probs.exp().topk(8)
                    first_top = [{'token_id': int(t), 'token': tok.decode([int(t)]), 'probability': float(v)}
                                 for t, v in zip(indices, values)]
                scoring.append({'output_position': step, 'token_id': token, 'token': tok.decode([token]),
                    'log_probability': float(log_probs[token]), 'probability': float(log_probs[token].exp()),
                    'greedy_token_id': greedy, 'greedy_token': tok.decode([greedy]), 'eos': token in eos})
                token_ids.append(token)
                if token in eos:
                    stop = 'eos'
                    break
                ids = torch.tensor([[token]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            assert all(b.self_attn.forward == f for b, f in zip(model.model.layers, native_attentions))
        finally:
            for handle in handles:
                handle.remove()

        # Audit full decoded words, so exact-token matching cannot silently miss a split child.
        text = tok.decode(token_ids, skip_special_tokens=True)
        word_matches = []
        boundaries = [len(tok.decode(token_ids[:i], skip_special_tokens=True)) for i in range(len(token_ids)+1)]
        for m in re.finditer(r'(?<!\w)child(?!\w)', text, flags=re.IGNORECASE):
            covering = [i for i in range(len(token_ids)) if boundaries[i+1] > m.start() and boundaries[i] < m.end()]
            assert len(covering) == 1 and is_child(token_ids[covering[0]]), (text, covering)
            i = covering[0]
            assert i < len(token_ids)-1, 'Generated child must be processed before termination.'
            word_matches.append({'text': m.group(), 'output_position': i, 'full_position': prompt_len+i,
                                 'token_id': token_ids[i], 'covered_by_matching_processed_token': True})
        expected_edit_positions = []
        if condition['feature'] is not None:
            expected_edit_positions.append(target)
            if condition['edit_scope'] == 'input_and_generated':
                expected_edit_positions.extend(prompt_len+i for i, token in enumerate(token_ids[:-1]) if is_child(token))
        assert [e['position'] for e in edits] == expected_edit_positions
        if condition['id'] in ['native', 'zero_control']:
            assert token_ids == case['output_token_ids'], (case['id'],condition['id'])
            if forced is not None:
                assert all(s['token_id']==s['greedy_token_id'] for s in scoring)
        assert not block.mlp._forward_hooks and not block.post_attention_layernorm._forward_pre_hooks
        return {'case_id': case['id'], 'condition': condition, 'user': case['user'],
            'chat': chat, 'prompt_token_ids': enc['input_ids'], 'input_target_position': target,
            'output_token_ids': token_ids, 'answer': text,
            'stop_reason': stop, 'first_prediction_top_tokens': first_top,
            'native_generation_unhooked': not hooked, 'forced_baseline_continuation': forced is not None,
            'token_scores': scoring, 'sum_log_probability_without_eos': sum(s['log_probability'] for s in scoring if not s['eos']),
            'traces': traces, 'edits': edits, 'vector_files': vector_files,
            'generated_child_word_occurrences': word_matches,
            'processed_response_positions_edited': sum(e['segment']=='response' and e['actual_bf16_delta_norm']>0 for e in edits),
            'all_generated_child_word_occurrences_covered': True,
            'attention_functions_unmodified': True}

    for case in cases:
        for condition in P['conditions']:
            generated = run(case, condition)
            fixed_prefix = run(case, condition, forced=case['output_token_ids'])
            result['rows'].append({'case_id': case['id'], 'condition_id': condition['id'],
                'generation': generated, 'baseline_continuation_scores': fixed_prefix})
            save('results.json', result)
            print(case['id'],condition['id'],json.dumps(generated['answer']),
                'EDITS='+json.dumps(generated['edits']),flush=True)
    assert len(result['rows']) == 8
    result.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-START,all_sanity_controls_match_prior_native_tokens=True,
        ablation_confined_to_requested_child_positions=True,
        all_generated_child_word_occurrences_covered=True)
    save('results.json',result)
    print('COMPLETE: two cases, four fixed conditions, input and generated child coverage audited.',flush=True)


if __name__=='__main__':
    main()
