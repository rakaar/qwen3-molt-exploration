"""Three native safety responses, followed by observational greedy replays."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    a = ap.parse_args()
    dest = ROOT / 'measurements.json'
    assert not dest.exists(), 'Refusing to overwrite measurements'
    start = time.monotonic()
    p = json.loads((ROOT / 'protocol.json').read_text())
    cases = json.loads((ROOT / 'prompts.json').read_text())
    verification = json.loads((ROOT / 'expected-base-verification.json').read_text())
    assert verification['repo'] == p['base_repo'] and verification['revision'] == p['base_revision']
    for rec in verification['files']:
        path = a.model_dir / rec['file']
        assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256'], rec['file']
    assert sha(a.weights) == p['asset_sha256']
    assert torch.__version__ == p['runtime']['torch']
    assert transformers.__version__ == p['runtime']['transformers']
    torch.manual_seed(p['seed'])
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    assert free > 11 * 1024**3, 'Insufficient free GPU memory for an isolated model copy'
    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    weights = {k: v.cuda() for k, v in load_file(a.weights).items()}
    layer, feature = p['layer'], p['transform']
    assert weights['e.weight'].shape == (2480, 2560)
    assert weights[f'V_{feature}'].shape == (2560, p['rank'])
    threshold = weights['nonlinearity.theta'].reshape(-1)[feature]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    special = set(tok.all_special_ids)
    attention_functions = [b.self_attn.forward for b in model.model.layers]
    result = {
        'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': p, 'threshold': float(threshold), 'base_files_hash_verified': True,
        'source_sha256': {n: sha(ROOT/n) for n in
                          ['run.py', 'protocol.json', 'prompts.json', 'expected-base-verification.json']},
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                    'python': platform.python_version(), 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(), 'initial_free_gpu_bytes': free},
        'rows': []}

    def deadline():
        assert time.monotonic()-start < 60*p['max_runtime_minutes'], 'Runtime limit reached'

    @torch.inference_mode()
    def native_generate(encoded):
        ids = torch.tensor([encoded], device='cuda')
        mask = torch.ones_like(ids)
        cache, tokens, stop = None, [], 'token_limit'
        for _ in range(p['max_new_tokens']):
            deadline()
            pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
            cache = pred.past_key_values
            token = int(pred.logits[0, -1].argmax())
            tokens.append(token)
            if token in eos:
                stop = 'eos'
                break
            ids = torch.tensor([[token]], device='cuda')
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
        return tokens, stop

    for case in cases:
        deadline()
        chat = tok.apply_chat_template([{'role': 'user', 'content': case['prompt']}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        encoded = tok(chat, add_special_tokens=False)['input_ids']
        # Generate before installing any observation hooks.
        output, stop = native_generate(encoded)
        answer = tok.decode(output, skip_special_tokens=True)
        state = {}

        def observe_input(_module, inputs):
            z = (inputs[0][:, -1:, :].detach().float() - weights['input_standardizer.mean'][layer])
            z = z / weights['input_standardizer.std'][layer]
            raw = F.linear(z, weights['e.weight'][feature:feature+1],
                           weights['e.bias'][feature:feature+1])
            full = F.linear(z, weights['e.weight'], weights['e.bias'])
            assert torch.allclose(raw, full[..., feature:feature+1], atol=2e-5, rtol=2e-5)
            gate = raw * ((raw > threshold) & (raw > 0))
            lowrank = (z @ weights[f'V_{feature}'] @ weights[f'U_{feature}'])
            lowrank = lowrank * weights['output_standardizer.std'][layer]
            state.update(raw=float(raw.item()), gate=float(gate.item()),
                         lowrank_norm=float(lowrank.norm()),
                         contribution_norm=float((gate*lowrank).norm()))
            # Returning None leaves the native input unchanged.

        def observe_mlp(_module, _inputs, output_tensor):
            state['native_mlp_norm'] = float(output_tensor[:, -1:, :].float().norm())
            # Returning None leaves the native output unchanged.

        block = model.model.layers[layer]
        handles = [block.post_attention_layernorm.register_forward_pre_hook(observe_input),
                   block.mlp.register_forward_hook(observe_mlp)]
        row = {**case, 'chat': chat, 'prompt_token_ids': encoded,
               'output_token_ids': output, 'answer': answer,
               'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
               'stop_reason': stop, 'native_generation_no_hooks': True, 'answer_tokens': []}
        try:
            with torch.inference_mode():
                ids = torch.tensor([encoded], device='cuda')
                mask = torch.ones_like(ids)
                pred = model(input_ids=ids, attention_mask=mask, use_cache=True)
                cache = pred.past_key_values
                assert int(pred.logits[0, -1].argmax()) == output[0]
                row['final_prompt_token'] = {'token_id': encoded[-1],
                    'token': tok.decode([encoded[-1]]), **state}
                for j, token in enumerate(output):
                    deadline()
                    ids = torch.tensor([[token]], device='cuda')
                    mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
                    pred = model(input_ids=ids, attention_mask=mask,
                                 past_key_values=cache, use_cache=True)
                    cache = pred.past_key_values
                    if j+1 < len(output):
                        assert int(pred.logits[0, -1].argmax()) == output[j+1], (case['id'], j)
                    row['answer_tokens'].append({
                        'answer_position': j, 'full_chat_position': len(encoded)+j,
                        'token_id': token, 'token': tok.decode([token]),
                        'special': token in special, 'eos': token in eos, **state})
        finally:
            for handle in handles:
                handle.remove()
        assert all(b.self_attn.forward == f for b, f in zip(model.model.layers, attention_functions))
        visible = [t for t in row['answer_tokens'] if not t['special']]
        row['summary'] = {
            'visible_tokens': len(visible), 'active_tokens': sum(t['gate'] > 0 for t in visible),
            'mean_gate': sum(t['gate'] for t in visible)/len(visible),
            'max_gate': max(t['gate'] for t in visible),
            'first_active_answer_position': next((t['answer_position'] for t in visible if t['gate'] > 0), None),
            'final_prompt_gate': row['final_prompt_token']['gate']}
        row['all_replay_predictions_match_native'] = True
        result['rows'].append(row)
        save(dest, result)
        print(case['id'], json.dumps(row['summary']), 'stop='+stop, flush=True)
    result.update(status='complete', elapsed_seconds=time.monotonic()-start,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    save(dest, result)
    print('COMPLETE: 3 unsteered generations; all observational replay predictions matched.', flush=True)


if __name__ == '__main__':
    main()
