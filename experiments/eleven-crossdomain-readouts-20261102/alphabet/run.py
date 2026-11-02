"""Read the saved full MLP, removed MOLT, and edited MLP through one frozen output readout."""
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
assert not (OUT / 'results.json').exists(), 'Do not overwrite prior readouts'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


assert torch.__version__ == '2.14.0+cu126' and transformers.__version__ == '4.57.6'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
expected = json.loads((ROOT / 'expected-base-verification.json').read_text())
for row in expected['files']:
    path = ROOT / 'model' / row['file']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['file']
case = json.loads((ROOT / 'case.json').read_text())
reference = json.loads((ROOT / 'reference.json').read_text())
assert sha(ROOT / 'reference-vectors.npz') == reference['vectors_sha256']
with np.load(ROOT / 'reference-vectors.npz', allow_pickle=False) as a:
    j = list(a['positions']).index(case['position'])
    full = torch.from_numpy(a['native_mlp'][0, j].copy()).float().cuda()
    contribution = -torch.from_numpy(a['intended_delta'][0, j].copy()).float().cuda()
    after = torch.from_numpy(a['edited_mlp'][0, j].copy()).float().cuda()
manifest = json.loads((ROOT / 'weights/manifest.json').read_text())
feature = manifest['features'][0]
assert (feature['layer'], feature['transform']) == (29, 2375)
assert sha(ROOT / 'weights' / feature['file']) == feature['sha256']
with np.load(ROOT / 'weights' / feature['file'], allow_pickle=False) as a:
    weights = {k: torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
tok = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(ROOT / 'model', local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
prompt = case['prompt']
ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device='cuda')
assert ids.shape[1] == case['position'] + 1 and ids[0].tolist() == reference['native']['prompt_token_ids']
state = {}


def observe_mlp_input(_module, inputs):
    state['mlp_input_residual'] = inputs[0].detach().float()


def observe_mlp(_module, _inputs, output):
    state['mlp_output'] = output[0, -1].detach().float()


def observe_norm_input(_module, inputs):
    state['final_residual'] = inputs[0][0, -1].detach().float()


def observe_norm_output(_module, _inputs, output):
    state['final_norm_output'] = output[0, -1].detach()


handles = [model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observe_mlp_input),
           model.model.layers[29].mlp.register_forward_hook(observe_mlp),
           model.model.norm.register_forward_pre_hook(observe_norm_input),
           model.model.norm.register_forward_hook(observe_norm_output)]
with torch.inference_mode():
    try:
        prediction = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(state['mlp_output'], full), 'Saved original MLP output did not reproduce'
    z = (state['mlp_input_residual'] - weights['input_mean']) / weights['input_std']
    raw = F.linear(z, weights['e'].reshape(1, -1), weights['bias']).squeeze(-1)
    gate = torch.where((raw > weights['theta'][0]) & (raw > 0), raw, 0.)
    operator = (z @ weights['V'] @ weights['U']) * weights['output_std']
    recomputed_contribution = (gate[..., None] * operator)[0, -1]
    assert torch.allclose(recomputed_contribution, contribution, atol=1e-5, rtol=1e-5)
    assert torch.equal((full - contribution).to(torch.bfloat16).float(), after)
    actual_logits = prediction.logits[0, -1].float()
    assert int(actual_logits.argmax()) == reference['native']['steps'][0]['token_id']
    assert float(actual_logits[int(actual_logits.argmax())]) == reference['native']['steps'][0]['selected_logit']
    residual = state['final_residual']
    epsilon = model.model.norm.variance_epsilon
    scale = torch.rsqrt(residual.square().mean() + epsilon)
    gamma = model.model.norm.weight.detach().float()
    manual_norm = model.model.norm.weight * (residual * scale).to(torch.bfloat16)
    assert torch.equal(manual_norm, state['final_norm_output']), 'Native final RMSNorm did not reconstruct'
    output_matrix = model.lm_head.weight.detach().float()
    parts = ['full_mlp', 'removed_molt', 'after_removal']
    vectors = torch.stack([full, contribution, after])
    scores = (vectors * gamma * scale) @ output_matrix.T
    raw_scores = vectors @ output_matrix.T
    ideal_after = full - contribution
    ideal_after_scores = (ideal_after * gamma * scale) @ output_matrix.T
    closure_error = float((scores[0] - scores[1] - ideal_after_scores).abs().max())
    assert closure_error < 1e-4
    assert torch.isfinite(scores).all()

    def token_row(token_id):
        return {'token_id': int(token_id), 'token': tok.decode([int(token_id)], clean_up_tokenization_spaces=False)}

    rankings, sampled_ids = {}, set()
    for index, name in enumerate(parts):
        values, top = scores[index].topk(30)
        low_values, low = scores[index].topk(30, largest=False)
        rankings[name] = {
            'highest': [dict(token_row(i), score=float(v)) for i, v in zip(top, values)],
            'lowest': [dict(token_row(i), score=float(v)) for i, v in zip(low, low_values)]}
        sampled_ids.update(int(i) for i in top)
        sampled_ids.update(int(i) for i in low)
    target_rows = []
    for text in case['target_tokens']:
        encoded = tok.encode(text, add_special_tokens=False)
        assert len(encoded) == 1
        token_id = encoded[0]
        sampled_ids.add(token_id)
        target_rows.append(dict(token_row(token_id), scores={name: float(scores[i, token_id])
            for i, name in enumerate(parts)}, ranks={name: int((scores[i] > scores[i, token_id]).sum()) + 1
            for i, name in enumerate(parts)}, ideal_after_score=float(ideal_after_scores[token_id]),
            raw_scores={name: float(raw_scores[i, token_id]) for i, name in enumerate(parts)},
            actual_native_final_logit=float(actual_logits[token_id])))
    sampled_ids = torch.tensor(sorted(sampled_ids), device='cuda')
    np.savez_compressed(OUT / 'readout-arrays.npz',
        vectors=vectors.cpu().numpy(), scores=scores.cpu().numpy(), raw_scores=raw_scores.cpu().numpy(),
        ideal_after_vector=ideal_after.cpu().numpy(), ideal_after_scores=ideal_after_scores.cpu().numpy(),
        final_residual=residual.cpu().numpy(), final_gamma=gamma.cpu().numpy(),
        final_scale=scale.cpu().numpy(), epsilon=np.array(epsilon),
        actual_native_final_logits=actual_logits.cpu().numpy(), final_norm_output=state['final_norm_output'].float().cpu().numpy(),
        sampled_token_ids=sampled_ids.cpu().numpy(), sampled_output_rows=output_matrix[sampled_ids].cpu().numpy())
    result = {'status': 'complete', 'prompt': prompt, 'position': case['position'], 'token': ' is', 'case': case,
        'parts': parts, 'part_definitions': {'full_mlp': 'Native L29 MLP output M',
            'removed_molt': 'Positive intended MOLT contribution c, before BF16 rounding',
            'after_removal': 'Actual edited MLP output, BF16(M - c)'},
        'method': 'W_out @ (gamma * vector / sqrt(mean(native_final_residual^2) + epsilon))',
        'normalization': 'One frozen native final RMSNorm scale and learned gamma, shared across all vectors',
        'normalization_scale': float(scale), 'epsilon': epsilon, 'vocabulary_size': int(scores.shape[1]),
        'rankings': rankings, 'target_rows': target_rows,
        'native_mlp_reproduced_exactly': True, 'native_final_norm_reconstructed_exactly': True,
        'maximum_contribution_reconstruction_error': float((recomputed_contribution - contribution).abs().max()),
        'ideal_additivity_max_absolute_error': closure_error,
        'actual_bf16_score_closure_max_absolute_error': float((scores[0] - scores[1] - scores[2]).abs().max()),
        'actual_after_vs_ideal_after_l2': float((after - ideal_after).norm()),
        'arrays_sha256': sha(OUT / 'readout-arrays.npz'),
        'runtime': {'base_revision': expected['revision'], 'verified_base_files': expected['files'],
            'torch': torch.__version__, 'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
            'dtype': 'BF16 model; FP32 diagnostic readout', 'attention': 'eager', 'tf32': False,
            'output_matrix': 'model.lm_head.weight',
            'source_sha256': {p.name: sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py', '.json']}},
        'limits': ['These are direct vocabulary scores, not answer probabilities or a complete forward pass from the edited state.',
                   'Only one prompt and position are analyzed; contributions from the incoming residual and later layers are not included in the three MLP-part scores.',
                   'BF16 rounding makes actual-after scores slightly different from ideal M-minus-c scores.']}
    save(OUT / 'results.json', result)
    print(json.dumps({'status': 'complete', 'target_rows': target_rows,
        'removed_molt_highest': rankings['removed_molt']['highest'][:8],
        'removed_molt_lowest': rankings['removed_molt']['lowest'][:8],
        'case': case['id'],
        'ideal_additivity_max_absolute_error': closure_error}, ensure_ascii=False), flush=True)
