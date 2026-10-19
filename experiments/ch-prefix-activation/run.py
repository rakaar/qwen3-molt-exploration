"""Read-only gate traces, with unhooked native generation and token-exact replay."""
import argparse
import hashlib
import json
import platform
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


def save(name, data):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def deadline():
    assert time.monotonic() - START < P['max_runtime_minutes'] * 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    args = ap.parse_args()
    assert not (OUT / 'measurements.json').exists(), 'Completed/incomplete results require inspection before retry.'
    base = json.loads((ROOT / 'expected-base-verification.json').read_text())
    assert base['repo'] == P['base_repo'] and base['revision'] == P['base_revision']
    for f in base['files']:
        path = args.model_dir / f['file']
        assert path.stat().st_size == f['bytes'] and sha(path) == f['sha256'], f['file']
    save('base-verification.json', base)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.__version__ == P['runtime']['torch']
    assert transformers.__version__ == P['runtime']['transformers']
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
    attentions = [b.self_attn.forward for b in model.model.layers]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    special = set(tok.all_special_ids)
    cases = json.loads((ROOT / 'cases.json').read_text())
    baselines = {'protocol': P, 'rows': []}
    with torch.inference_mode():
        for case in cases:
            deadline()
            chat = tok.apply_chat_template([{'role': 'user', 'content': case['user']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
            tokens = enc['input_ids']
            user_start = chat.index(case['user'])
            start, end = [user_start + value for value in case['target_span']]
            target_positions = [i for i, (a, b) in enumerate(enc['offset_mapping'])
                if b > a and b > start and a < end]
            assert target_positions and chat[start:end] == case['target']
            ids = torch.tensor([tokens], device='cuda')
            mask = torch.ones_like(ids)
            cache = None
            generated = []
            stop = 'token_limit'
            first_top = None
            for step in range(P['decoding']['max_new_tokens']):
                deadline()
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                token = int(pred.logits[0, -1].argmax())
                if step == 0:
                    prob = pred.logits[0, -1].float().softmax(-1)
                    values, indices = prob.topk(5)
                    first_top = [{'token_id': int(t), 'token': tok.decode([int(t)]), 'probability': float(v)}
                                 for t, v in zip(indices, values)]
                generated.append(token)
                if token in eos:
                    stop = 'eos'
                    break
                ids = torch.tensor([[token]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            row = {**case, 'chat': chat, 'prompt_token_ids': tokens,
                'prompt_token_offsets': enc['offset_mapping'], 'target_positions': target_positions,
                'output_token_ids': generated, 'answer': tok.decode(generated, skip_special_tokens=True),
                'stop_reason': stop, 'first_prediction_top_tokens': first_top,
                'baseline_generation_no_hooks': True}
            baselines['rows'].append(row)
            save('baseline.json', baselines)
            print('NATIVE', case['id'], json.dumps(row['answer']), flush=True)
            del pred, cache

    assets = json.loads((ROOT / 'weights/manifest.json').read_text())
    assert assets['revision'] == P['molt_revision']
    weights = {}
    for f in assets['features']:
        path = ROOT / 'weights' / f['file']
        assert sha(path) == f['sha256']
        with np.load(path) as data:
            weights[f['transform']] = {k: torch.from_numpy(data[k].copy()).float().cuda() for k in data.files}
    assert set(weights) == {2334, 1372}
    state = {}

    def observe_input(_module, inputs):
        for feature, w in weights.items():
            z = (inputs[0].detach().float() - w['input_mean']) / w['input_std']
            raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
            independent = (z * w['e']).sum(-1) + w['bias'][0]
            assert torch.allclose(raw, independent, atol=3e-5, rtol=3e-5)
            gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, torch.zeros_like(raw))
            low = (z @ w['V'] @ w['U']) * w['output_std']
            state[feature] = {'raw': raw[0].cpu().tolist(), 'gate': gate[0].cpu().tolist(),
                'lowrank_norm': low.norm(dim=-1)[0].cpu().tolist(),
                'contribution_norm': (gate.unsqueeze(-1) * low).norm(dim=-1)[0].cpu().tolist()}
        return None

    def observe_output(_module, _inputs, output):
        state['native_mlp_norm'] = output.detach().float().norm(dim=-1)[0].cpu().tolist()
        return None

    def entries(ids, segment, offset, target_positions):
        rows = []
        for j, token in enumerate(ids):
            position = offset + j
            rows.append({'segment': segment, 'full_chat_position': position,
                'token_id': token, 'token': tok.decode([token]), 'special': token in special,
                'target_token': position in target_positions,
                'native_mlp_norm': state['native_mlp_norm'][j],
                'features': {f'L3-T{feature}': {k: v[j] for k, v in state[feature].items()}
                             for feature in weights}})
        return rows

    block = model.model.layers[3]
    handles = [block.post_attention_layernorm.register_forward_pre_hook(observe_input),
               block.mlp.register_forward_hook(observe_output)]
    result = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': P, 'assets': assets,
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                    'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
                    'cuda': torch.version.cuda},
        'source_sha256': {name: sha(ROOT / name) for name in ['run.py', 'protocol.json', 'cases.json']},
        'rows': []}
    try:
        with torch.inference_mode():
            for row in baselines['rows']:
                deadline()
                ids = torch.tensor([row['prompt_token_ids']], device='cuda')
                mask = torch.ones_like(ids)
                pred = model(input_ids=ids, attention_mask=mask, use_cache=True)
                cache = pred.past_key_values
                assert int(pred.logits[0, -1].argmax()) == row['output_token_ids'][0], row['id']
                observed = {**row, 'prompt_tokens': entries(row['prompt_token_ids'], 'prompt', 0,
                    row['target_positions']), 'answer_tokens': []}
                for j, token in enumerate(row['output_token_ids']):
                    if token in eos:
                        break
                    mask = torch.cat([mask, torch.ones((1, 1), device='cuda', dtype=mask.dtype)], dim=1)
                    pred = model(input_ids=torch.tensor([[token]], device='cuda'), attention_mask=mask,
                                 past_key_values=cache, use_cache=True)
                    cache = pred.past_key_values
                    observed['answer_tokens'] += entries([token], 'answer', len(row['prompt_token_ids']) + j, [])
                    if j + 1 < len(row['output_token_ids']):
                        assert int(pred.logits[0, -1].argmax()) == row['output_token_ids'][j + 1], (row['id'], j)
                assert all(b.self_attn.forward == f for b, f in zip(model.model.layers, attentions))
                observed['all_native_greedy_predictions_verified'] = True
                result['rows'].append(observed)
                save('measurements.json', result)
                target_rows = [t for t in observed['prompt_tokens'] if t['target_token']]
                print('OBSERVED', row['id'], json.dumps({'source': [{t['token']:t['features']} for t in target_rows],
                    'answer': [{t['token']:t['features']} for t in observed['answer_tokens']]}), flush=True)
                del pred, cache
    finally:
        for handle in handles:
            handle.remove()
    result.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic() - START, native_attention_and_mlp_unmodified=True,
        all_observation_hooks_return_none=True, native_generation_replay_token_identical=True)
    save('measurements.json', result)
    print('COMPLETE', len(result['rows']), 'native generations and token-exact observation replays.', flush=True)


if __name__ == '__main__':
    main()
