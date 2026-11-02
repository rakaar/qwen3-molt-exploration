"""Read-only L29-T2375 observation on raw month-completion prompts."""
import hashlib
import json
import os
from pathlib import Path

for key in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'VAST_API_KEY', 'VAST_AI_KEY']:
    os.environ.pop(key, None)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


assert torch.__version__ == '2.14.0+cu126'
assert transformers.__version__ == '4.57.6'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
expected = json.loads((ROOT / 'expected-base-verification.json').read_text())
verified = []
for row in expected['files']:
    path = ROOT / 'model' / row['file']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['file']
    verified.append(row)
manifest = json.loads((ROOT / 'weights/manifest.json').read_text())
feature = manifest['features'][0]
assert (feature['layer'], feature['transform']) == (29, 2375)
weight_file = ROOT / 'weights' / feature['file']
assert sha(weight_file) == feature['sha256']
with np.load(weight_file, allow_pickle=False) as array:
    weights = {k: torch.from_numpy(array[k].copy()).float().cuda() for k in array.files}
tok = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'model', local_files_only=True, dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval().requires_grad_(False)
assert model.config.hidden_size == 2560 and model.config.num_hidden_layers == 36
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])
save(OUT / 'runtime.json', {
    'base_revision': expected['revision'], 'verified_base_files': verified,
    'molt_revision': manifest['revision'], 'feature_sha256': feature['sha256'],
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
    'dtype': str(next(model.parameters()).dtype), 'attention': 'eager',
    'tf32': False, 'prompt_format': 'raw', 'intervention': None,
    'hook_location': 'L29 post_attention_layernorm forward-pre-hook; no return value',
    'source_sha256': {p.name: sha(p) for p in ROOT.iterdir()
                      if p.is_file() and p.suffix in ['.py', '.json', '.sh']}})


def hook_count():
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@torch.inference_mode()
def run(case_id, prompt, observe=False):
    assert hook_count() == 0
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    generated, steps, trace, residuals, standardized, raws, gates = [], [], [], [], [], [], []
    state = {'step': 0, 'last_id': None}

    def observer(_module, inputs):
        residual = inputs[0].detach().float()
        z = (residual - weights['input_mean']) / weights['input_std']
        raw = F.linear(z, weights['e'].reshape(1, -1), weights['bias']).squeeze(-1)
        gate = torch.where((raw > weights['theta'][0]) & (raw > 0), raw, 0.)
        token_ids = prompt_ids if state['step'] == 0 else [state['last_id']]
        positions = list(range(len(prompt_ids))) if state['step'] == 0 else [len(prompt_ids) + state['step'] - 1]
        assert len(token_ids) == residual.shape[1]
        for j, (position, token_id) in enumerate(zip(positions, token_ids)):
            trace.append({'position': position, 'token_id': token_id, 'token': tok.decode([token_id]),
                          'phase': 'input' if state['step'] == 0 else 'generated',
                          'raw': float(raw[0, j]), 'natural_gate': float(gate[0, j])})
        residuals.append(residual[0].cpu().numpy())
        standardized.append(z[0].cpu().numpy())
        raws.append(raw[0].cpu().numpy())
        gates.append(gate[0].cpu().numpy())
        # Returning None leaves the layer inputs unchanged. There is no output-edit hook.

    handle = model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observer) if observe else None
    ids = torch.tensor([prompt_ids], device='cuda')
    attention_mask, cache, stop = torch.ones_like(ids), None, 'token_limit'
    try:
        for step in range(32):
            state['step'] = step
            pred = model(input_ids=ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
            cache = pred.past_key_values
            logits = pred.logits[0, -1].float()
            assert torch.isfinite(logits).all()
            probabilities = logits.softmax(-1)
            values, indices = probabilities.topk(6)
            chosen = int(logits.argmax())
            steps.append({'step': step, 'token_id': chosen, 'token': tok.decode([chosen]),
                          'selected_probability': float(probabilities[chosen]),
                          'logits_sha256': hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
                          'top_tokens': [{'token_id': int(i), 'token': tok.decode([int(i)]), 'probability': float(v)}
                                         for i, v in zip(indices, values)]})
            generated.append(chosen)
            state['last_id'] = chosen
            if chosen in eos:
                stop = 'eos'
                break
            ids = torch.tensor([[chosen]], device='cuda')
            attention_mask = torch.cat([attention_mask, torch.ones_like(ids)], dim=1)
    finally:
        if handle is not None:
            handle.remove()
    assert hook_count() == 0
    result = {'case_id': case_id, 'prompt': prompt, 'prompt_token_ids': prompt_ids,
              'output_token_ids': generated, 'answer': tok.decode(generated, skip_special_tokens=True),
              'stop_reason': stop, 'steps': steps, 'trace': trace, 'intervention': None}
    if observe:
        assert len(trace) == len(prompt_ids) + len(generated) - 1
        vector_file = OUT / f'{case_id}--natural-vectors.npz'
        np.savez_compressed(vector_file, residual=np.concatenate(residuals), z=np.concatenate(standardized),
                            raw=np.concatenate(raws), gate=np.concatenate(gates))
        result['vectors'] = {'file': vector_file.name, 'sha256': sha(vector_file)}
    return result


cases = [
    ('numeric_11', 'the 11th month of the year is'),
    ('spelled_11_anchor', 'the eleventh month of the year is'),
    ('numeric_12_control', 'the 12th month of the year is'),
]
results = {'status': 'running', 'feature': 'L29-T2375', 'intervention': None, 'cases': []}
for case_id, prompt in cases:
    native = run(case_id, prompt)
    observed = run(case_id, prompt, observe=True)
    assert native['output_token_ids'] == observed['output_token_ids']
    assert native['steps'] == observed['steps'], 'Observer changed full-vocabulary logits'
    row = {'case_id': case_id, 'native': native, 'observed': observed,
           'observer_full_logits_exact': True}
    if case_id == 'spelled_11_anchor':
        anchor = json.loads((ROOT / 'english-reference.json').read_text())[0]
        assert native['output_token_ids'] == anchor['output_token_ids']
        error = max(abs(a['selected_probability'] - b['selected_probability'])
                    for a, b in zip(native['steps'], anchor['steps']))
        assert error < 1e-4
        row['historical_anchor_max_probability_error'] = error
    results['cases'].append(row)
    save(OUT / 'results.json', results)
    print(json.dumps({'case_id': case_id, 'response': native['answer'],
                      'active_input_tokens': [t for t in observed['trace']
                                              if t['phase'] == 'input' and t['natural_gate'] > 0]},
                     ensure_ascii=False), flush=True)
results.update(status='complete', hooks_remaining=hook_count())
save(OUT / 'results.json', results)
print('PROBE_COMPLETE', flush=True)
