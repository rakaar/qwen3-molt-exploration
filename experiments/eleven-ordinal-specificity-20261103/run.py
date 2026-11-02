"""Observe native L29-T2375 for matched month and Roman ordinal prompts."""
import hashlib
import json
import os
from pathlib import Path
import random

for key in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'VAST_API_KEY', 'VAST_AI_KEY']:
    os.environ.pop(key, None)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
BASE = Path('/workspace/eleven-activation-20261102')
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
assert not (OUT / 'results.json').exists(), 'Preserve existing results'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


assert torch.__version__ == '2.14.0+cu126' and transformers.__version__ == '4.57.6'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
expected = json.loads((BASE / 'expected-base-verification.json').read_text())
for row in expected['files']:
    path = BASE / 'model' / row['file']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
manifest = json.loads((BASE / 'weights/manifest.json').read_text())
feature = manifest['features'][0]
assert (feature['layer'], feature['transform']) == (29, 2375)
weight_file = BASE / 'weights' / feature['file']
assert sha(weight_file) == feature['sha256']
with np.load(weight_file, allow_pickle=False) as a:
    weights = {k: torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
tok = AutoTokenizer.from_pretrained(BASE / 'model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(BASE / 'model', local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])


@torch.inference_mode()
def run(prompt, observe):
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    steps, state = [], {}

    def observer(_module, inputs):
        if state['step'] != 0:
            return
        residual = inputs[0].detach().float()
        z = (residual - weights['input_mean']) / weights['input_std']
        raw = F.linear(z, weights['e'].reshape(1, -1), weights['bias']).squeeze(-1)
        gate = torch.where((raw > weights['theta'][0]) & (raw > 0), raw, 0.)
        operator = (z @ weights['V'] @ weights['U']) * weights['output_std']
        state['vectors'] = {k: v[0].cpu().numpy() for k, v in {
            'residual': residual, 'z': z, 'raw': raw, 'gate': gate,
            'operator': operator, 'contribution': gate[..., None] * operator}.items()}
        # Returning None leaves the native computation unchanged.

    handle = model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observer) if observe else None
    ids = torch.tensor([prompt_ids], device='cuda')
    mask, cache, generated = torch.ones_like(ids), None, []
    try:
        for step in range(12):
            state['step'] = step
            pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
            cache = pred.past_key_values
            logits = pred.logits[0, -1].float()
            chosen = int(logits.argmax())
            assert torch.isfinite(logits).all()
            steps.append({'token_id': chosen, 'token': tok.decode([chosen]),
                          'probability': float(logits.softmax(-1)[chosen]),
                          'logits_sha256': hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()})
            generated.append(chosen)
            if chosen in eos:
                break
            ids = torch.tensor([[chosen]], device='cuda')
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
    finally:
        if handle is not None:
            handle.remove()
    result = {'prompt': prompt, 'prompt_token_ids': prompt_ids,
              'prompt_tokens': [tok.decode([i]) for i in prompt_ids],
              'output_token_ids': generated, 'response': tok.decode(generated, skip_special_tokens=True),
              'steps': steps}
    return result, state.get('vectors')


seed = 20261103
extra = random.Random(seed).sample([3, 4, 5, 6, 7, 8, 9, 10, 12], 2)
ordinals = [11, 1, 2, *extra]
words = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth',
         7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth', 11: 'eleventh', 12: 'twelfth'}
result = {'status': 'running', 'feature': 'L29-T2375', 'prompt_format': 'raw-prefix',
          'intervention': None, 'max_new_tokens': 12, 'seed': seed, 'sampled_ordinals': extra,
          'sampling_population': [3, 4, 5, 6, 7, 8, 9, 10, 12], 'cases': [],
          'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                      'gpu': torch.cuda.get_device_name(), 'dtype': 'BF16 native model; FP32 MOLT observer',
                      'attention': 'eager', 'tf32': False, 'base_revision': expected['revision'],
                      'molt_revision': manifest['revision'], 'feature_sha256': feature['sha256'],
                      'source_sha256': sha(Path(__file__)), 'verified_base_files': expected['files']}}
for domain, template in [('month', 'the {ordinal} month of the year is'),
                         ('roman', 'the {ordinal} positive integer in roman numerals is')]:
    for ordinal in ordinals:
        prompt = template.format(ordinal=words[ordinal])
        native, _ = run(prompt, False)
        observed, vectors = run(prompt, True)
        assert native == observed, 'Observer changed native logits or answers'
        assert native['prompt_tokens'][-1] == ' is'
        path = OUT / f'{domain}-{ordinal:02d}-vectors.npz'
        np.savez_compressed(path, **vectors)
        row = {'domain': domain, 'ordinal': ordinal, 'ordinal_word': words[ordinal],
               'native': native, 'observer_full_logits_exact': True,
               'position': len(native['prompt_token_ids']) - 1, 'token': ' is',
               'raw': float(vectors['raw'][-1]), 'gate': float(vectors['gate'][-1]),
               'operator_l2': float(np.linalg.norm(vectors['operator'][-1].astype(np.float64))),
               'contribution_l2': float(np.linalg.norm(vectors['contribution'][-1].astype(np.float64))),
               'vectors': {'file': path.name, 'sha256': sha(path)}}
        result['cases'].append(row)
        save(OUT / 'results.json', result)
        print(json.dumps({k: row[k] for k in ['domain', 'ordinal', 'gate', 'contribution_l2']}, ensure_ascii=False), flush=True)
result['status'] = 'complete'
save(OUT / 'results.json', result)
print('ORDINAL_SPECIFICITY_COMPLETE', flush=True)
