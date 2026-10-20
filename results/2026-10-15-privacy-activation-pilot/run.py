"""Unmodified generation, then token-exact observation-only replay at three layers."""
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
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, data):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    args = ap.parse_args()
    assert not (OUT / 'measurements.json').exists(), 'Do not overwrite completed results.'
    base = json.loads((ROOT / 'expected-base-verification.json').read_text())
    assert base['repo'] == P['base_repo'] and base['revision'] == P['base_revision']
    for item in base['files']:
        file = args.model_dir / item['file']
        assert file.stat().st_size == item['bytes'] and sha(file) == item['sha256'], item['file']
    save('base-verification.json', base)
    cases = json.loads((ROOT / 'cases.json').read_text())
    torch.manual_seed(0)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert torch.__version__ == P['runtime']['torch']
    assert transformers.__version__ == P['runtime']['transformers']
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    special = set(tok.all_special_ids)
    attention = [b.self_attn.forward for b in model.model.layers]
    baseline = {'protocol': P, 'rows': []}
    with torch.inference_mode():
        for c in cases:
            chat = tok.apply_chat_template([{'role': 'system', 'content': c['system']},
                {'role': 'user', 'content': c['user']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = tok(chat, add_special_tokens=False)['input_ids']
            ids = torch.tensor([encoded], device='cuda')
            mask = torch.ones_like(ids)
            cache = None
            tokens = []
            stop = 'token_limit'
            for step in range(P['decoding']['max_new_tokens']):
                assert time.monotonic() - START < P['max_runtime_minutes'] * 60
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                token = int(pred.logits[0, -1].argmax())
                tokens.append(token)
                if token in eos:
                    stop = 'eos'
                    break
                ids = torch.tensor([[token]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            row = {**c, 'chat': chat, 'prompt_token_ids': encoded, 'output_token_ids': tokens,
                   'answer': tok.decode(tokens, skip_special_tokens=True), 'stop_reason': stop,
                   'baseline_generation_no_hooks': True}
            baseline['rows'].append(row)
            save('baseline.json', baseline)
            print('NATIVE', c['id'], json.dumps(row['answer']), flush=True)
            del pred, cache
    assets = json.loads((ROOT / 'weights/manifest.json').read_text())
    assert assets['revision'] == P['molt_revision']
    weights = {}
    for f in assets['features']:
        file = ROOT / 'weights' / f['file']
        assert sha(file) == f['sha256']
        with np.load(file) as data:
            weights[f['layer']] = {key: torch.from_numpy(data[key].copy()).float().cuda() for key in data.files}
    state = {}
    handles = []
    for feature in P['features']:
        layer = feature['layer']
        w = weights[layer]

        def observe_input(_module, inputs, layer=layer, w=w):
            z = (inputs[0].detach().float() - w['input_mean']) / w['input_std']
            raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
            # Independent dot-product form checks indexing and projection arithmetic.
            check = (z * w['e']).sum(-1) + w['bias'][0]
            assert torch.allclose(raw, check, atol=3e-5, rtol=3e-5)
            gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, torch.zeros_like(raw))
            low = (z @ w['V'] @ w['U']) * w['output_std']
            state[layer] = {'raw': raw[0].cpu().tolist(), 'gate': gate[0].cpu().tolist(),
                'lowrank_norm': low.norm(dim=-1)[0].cpu().tolist(),
                'contribution_norm': (gate.unsqueeze(-1) * low).norm(dim=-1)[0].cpu().tolist()}
            # Returning None leaves the model input untouched.

        def observe_output(_module, _inputs, output, layer=layer):
            state[layer]['native_mlp_norm'] = output.detach().float().norm(dim=-1)[0].cpu().tolist()
            # Returning None leaves the native MLP output untouched.

        block = model.model.layers[layer]
        handles += [block.post_attention_layernorm.register_forward_pre_hook(observe_input),
                    block.mlp.register_forward_hook(observe_output)]
    result = {'status': 'running', 'protocol': P, 'assets': assets,
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                    'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
                    'cuda': torch.version.cuda},
        'source_sha256': {name: sha(ROOT / name) for name in ['run.py', 'protocol.json', 'cases.json']},
        'rows': [], 'started_at_utc': datetime.now(timezone.utc).isoformat()}

    def entries(token_ids, segment, offset):
        values = []
        for j, token in enumerate(token_ids):
            features = {}
            for f in P['features']:
                layer = f['layer']
                features[f"L{layer}-T{f['transform']}"] = {k: v[j] for k, v in state[layer].items()}
            values.append({'segment': segment, 'position': j if segment == 'prompt' else offset,
                'full_chat_position': offset + j if segment == 'prompt' else offset,
                'token_id': token, 'token': tok.decode([token]), 'special': token in special,
                'features': features})
        return values

    try:
        with torch.inference_mode():
            for c in baseline['rows']:
                ids = torch.tensor([c['prompt_token_ids']], device='cuda')
                mask = torch.ones_like(ids)
                pred = model(input_ids=ids, attention_mask=mask, use_cache=True)
                cache = pred.past_key_values
                assert int(pred.logits[0, -1].argmax()) == c['output_token_ids'][0]
                row = {**c, 'prompt_tokens': entries(c['prompt_token_ids'], 'prompt', 0), 'answer_tokens': []}
                for j, token in enumerate(c['output_token_ids']):
                    if token in eos:
                        break
                    mask = torch.cat([mask, torch.ones((1, 1), device='cuda', dtype=mask.dtype)], dim=1)
                    pred = model(input_ids=torch.tensor([[token]], device='cuda'), attention_mask=mask,
                                 past_key_values=cache, use_cache=True)
                    cache = pred.past_key_values
                    record = entries([token], 'response', len(c['prompt_token_ids']) + j)[0]
                    record['position'] = j
                    row['answer_tokens'].append(record)
                    if j + 1 < len(c['output_token_ids']):
                        assert int(pred.logits[0, -1].argmax()) == c['output_token_ids'][j + 1], (c['id'], j)
                assert all(b.self_attn.forward == f for b, f in zip(model.model.layers, attention))
                row['summaries'] = {}
                for feature in P['features']:
                    name = f"L{feature['layer']}-T{feature['transform']}"
                    summary = {}
                    for segment, records in [('response', row['answer_tokens']), ('prompt', row['prompt_tokens'])]:
                        visible = [r for r in records if not r['special']]
                        gates = [r['features'][name]['gate'] for r in visible]
                        summary[segment] = {'visible_tokens': len(gates), 'active_tokens': sum(g > 0 for g in gates),
                            'mean_gate': sum(gates) / len(gates), 'max_gate': max(gates),
                            'mean_active_gate': sum(gates) / sum(g > 0 for g in gates) if any(gates) else None}
                    row['summaries'][name] = summary
                row['all_saved_greedy_predictions_verified'] = True
                result['rows'].append(row)
                save('measurements.json', result)
                print('OBSERVED', c['id'], json.dumps(row['summaries']), flush=True)
                del pred, cache
    finally:
        for handle in handles:
            handle.remove()
    result.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - START,
                  attention_unmodified=True, all_observation_hooks_return_none=True,
                  native_generation_and_observed_replay_token_identical=True)
    save('measurements.json', result)
    print('COMPLETE: four native generations; three read-only gate traces each; token-exact replay.', flush=True)


if __name__ == '__main__':
    main()
