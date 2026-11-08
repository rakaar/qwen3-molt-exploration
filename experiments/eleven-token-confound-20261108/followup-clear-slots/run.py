"""Native generations and read-only all-input T2375 traces for frozen rewrites."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for key in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'VAST_API_KEY', 'VAST_AI_KEY']:
    os.environ.pop(key, None)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    expected = json.loads((HERE / 'expected-base-verification.json').read_text())
    assert expected['revision'] == protocol['base_revision']
    out = HERE / 'artifacts'
    out.mkdir(exist_ok=True)
    assert not (out / 'results.json').exists(), 'Preserve prior captures'
    for row in expected['files']:
        path = args.model_dir / row['file']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    weights_path = HERE / 'weights/L29-T2375.npz'
    manifest = json.loads((HERE / 'weights/manifest.json').read_text())
    feature = manifest['features'][0]
    assert manifest['revision'] == protocol['molt_revision']
    assert (feature['layer'], feature['transform']) == (29, 2375)
    assert sha(weights_path) == feature['sha256']
    assert torch.__version__ == '2.14.0+cu126' and transformers.__version__ == '4.57.6'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    encodings = {
        c['id']: tokenizer(c['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        for c in protocol['cases']
    }
    for variant in protocol['distinct_final_token_variants']:
        for ordinal in [10, 11, 12]:
            final_ids = [encodings[c['id']]['input_ids'][-1] for c in protocol['cases']
                         if c['variant'] == variant and c['ordinal'] == ordinal]
            assert len(final_ids) == len(set(final_ids)) == 3
            assert 374 not in final_ids, 'A rewrite still ends with the original is token'
    with np.load(weights_path, allow_pickle=False) as a:
        w = {k: torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
    anchors = json.loads((HERE / 'anchors.json').read_text())
    assert sha(HERE / 'anchors.npz') == anchors['arrays_sha256']
    with np.load(HERE / 'anchors.npz', allow_pickle=False) as a:
        reference = {k: a[k].copy() for k in a.files}
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
    assert len(model.model.layers) == 36 and model.config.hidden_size == 2560
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    started = time.monotonic()

    @torch.inference_mode()
    def generate(prompt_ids, observe):
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        state, steps, output = {}, [], []

        def observer(module, inputs):
            if state['step'] != 0:
                return
            x = inputs[0].detach().float()
            z = (x - w['input_mean']) / w['input_std']
            raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
            gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, 0.)
            operator = (z @ w['V'] @ w['U']) * w['output_std']
            state['vectors'] = {k: v[0].cpu().numpy().copy() for k, v in {
                'residual': x, 'z': z, 'raw': raw, 'gate': gate,
                'operator': operator, 'contribution': gate[..., None] * operator}.items()}

        handle = model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observer) if observe else None
        ids = torch.tensor([prompt_ids], device='cuda')
        mask, cache = torch.ones_like(ids), None
        first_logits = None
        try:
            for step in range(protocol['max_new_tokens']):
                assert time.monotonic() - started < protocol['maximum_runtime_minutes'] * 60
                state['step'] = step
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                logits = pred.logits[0, -1].float()
                assert torch.isfinite(logits).all()
                values = logits.cpu().numpy()
                if step == 0:
                    first_logits = values.copy()
                chosen = int(logits.argmax())
                steps.append({'token_id': chosen, 'token': tokenizer.decode([chosen]),
                              'probability': float(logits.softmax(-1)[chosen]),
                              'logits_sha256': hashlib.sha256(values.tobytes()).hexdigest()})
                output.append(chosen)
                if chosen in eos:
                    break
                ids = torch.tensor([[chosen]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
        finally:
            if handle:
                handle.remove()
        return {'output_token_ids': output, 'response': tokenizer.decode(output, skip_special_tokens=True),
                'steps': steps}, state.get('vectors'), first_logits

    result = {'status': 'running', 'protocol': protocol, 'cases': [],
              'protocol_sha256': sha(HERE / 'protocol.json'), 'source_sha256': sha(Path(__file__)),
              'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                          'gpu': torch.cuda.get_device_name(), 'feature_sha256': feature['sha256'],
                          'native_dtype': 'BF16', 'observer_dtype': 'FP32', 'attention': 'eager', 'TF32': False}}
    save(out / 'results.json', result)
    for case in protocol['cases']:
        encoding = encodings[case['id']]
        ids = encoding['input_ids']
        native, _, plain_logits = generate(ids, False)
        observed, vectors, observed_logits = generate(ids, True)
        assert native == observed and np.array_equal(plain_logits, observed_logits)
        anchor = next((a for a in anchors['cases'] if a['prompt'] == case['prompt']), None)
        if anchor:
            assert ids == anchor['token_ids']
            assert np.array_equal(plain_logits, reference[anchor['name'] + '_first_logits'])
            assert np.array_equal(vectors['residual'], reference[anchor['name'] + '_L29_input'])
        tokens = [tokenizer.decode([i]) for i in ids]
        trace = [{'position': i, 'token_id': ident, 'token': token,
                  'raw': float(vectors['raw'][i]), 'gate': float(vectors['gate'][i]),
                  'operator_norm': float(np.linalg.norm(vectors['operator'][i].astype(np.float64))),
                  'contribution_norm': float(np.linalg.norm(vectors['contribution'][i].astype(np.float64)))}
                 for i, (ident, token) in enumerate(zip(ids, tokens))]
        path = out / (case['id'] + '.npz')
        np.savez_compressed(path, **vectors, input_ids=np.array(ids), first_logits=plain_logits)
        row = {**case, 'input_ids': ids, 'tokens': tokens, 'token_offsets': encoding['offset_mapping'],
               'native': native, 'trace': trace, 'final_input': trace[-1],
               'active_input_tokens': [x for x in trace if x['gate'] > 0],
               'peak_gate_position': int(np.argmax(vectors['gate'])),
               'observer_all_logits_and_output_exact': True, 'original_anchor_exact': bool(anchor),
               'vectors_file': path.name, 'vectors_sha256': sha(path)}
        result['cases'].append(row)
        save(out / 'results.json', result)
        print(json.dumps({'case': case['id'], 'last_token': tokens[-1],
                          'active_tokens': [(x['position'], x['token'], x['gate']) for x in trace if x['gate'] > 0],
                          'response': native['response']}, ensure_ascii=False), flush=True)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    result.update(status='complete', hooks_remaining=0, runtime_seconds=time.monotonic() - started)
    save(out / 'results.json', result)


if __name__ == '__main__':
    main()
