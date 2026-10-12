"""Matched token-9 residual comparison with ordinary, unfrozen attention."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT / 'protocol.json').read_text())
START = time.monotonic()
torch.manual_seed(0)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    temporary = OUT / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUT / name)


def cosine(a, b):
    a, b = a.detach().cpu().double().flatten(), b.detach().cpu().double().flatten()
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if denominator > 0 else None


tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'models/qwen3-4b', local_files_only=True, dtype=torch.bfloat16,
    attn_implementation='eager').cuda().eval().requires_grad_(False)
model.config.use_cache = False
attention_forwards = [block.self_attn.forward for block in model.model.layers]
tc = {k: v.cuda() for k, v in load_file(ROOT / 'models/transcoder/layer_22.safetensors').items()}
mw_path = ROOT / 'assets/molt-22-selected.safetensors'
assert sha(mw_path) == P['asset_provenance']['files'][mw_path.name]
mw = {k: v.cuda() for k, v in load_file(mw_path).items()}
assert tuple(tc['W_enc'].shape) == tuple(tc['W_dec'].shape) == (163840, 2560)


def encode(prompt, word):
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    batch = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                      return_tensors='pt')
    offsets = batch.pop('offset_mapping')[0].tolist()
    start = text.index(word, text.index(prompt))
    positions = [i for i, (a, b) in enumerate(offsets)
                 if b > a and b > start and a < start + len(word)]
    assert positions == [9]
    ids = batch['input_ids'].cuda()
    assert ids.shape == (1, 29)
    tokens = [{'position': i, 'id': int(v), 'text': tokenizer.decode([int(v)])}
              for i, v in enumerate(ids[0])]
    return ids, positions[0], tokens


def summarize(logits):
    p = logits[-1].float().softmax(-1)
    values, indices = p.topk(5)
    return {'P_cold': float(p[87072]), 'P_large': float(p[16767]),
            'top_token': tokenizer.decode([int(indices[0])]),
            'top_tokens': [{'id': int(i), 'text': tokenizer.decode([int(i)]),
                            'probability': float(v)} for i, v in zip(indices, values)]}


@torch.inference_mode()
def forward(ids, pos, delta=None, probe_attention=False):
    assert all(block.self_attn.forward == original for block, original in
               zip(model.model.layers, attention_forwards))
    assert all(not block.mlp._forward_hooks for block in model.model.layers)
    caps = SimpleNamespace(residuals={}, outside_positions_unchanged=True)
    handles = []

    def pre(_module, args):
        caps.prenorm = args[0][0].detach().clone()

    def mlp(_module, args, output):
        caps.mlp_input = args[0][0].detach().clone()
        caps.mlp_output = output[0].detach().clone()
        if delta is None:
            return output
        changed = output.clone()
        changed[0, pos] = output[0, pos] + delta.to(output.dtype)
        mask = torch.ones(output.shape[1], device=output.device, dtype=torch.bool)
        mask[pos] = False
        assert torch.equal(changed[:, mask], output[:, mask])
        caps.edited_mlp_output = changed[0].detach().clone()
        return changed

    block = model.model.layers[22]
    handles.extend([block.post_attention_layernorm.register_forward_pre_hook(pre),
                    block.mlp.register_forward_hook(mlp)])
    for layer, block in enumerate(model.model.layers):
        def residual_hook(_module, _args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            caps.residuals[layer] = hidden[0, pos].detach().cpu().float()
        handles.append(block.register_forward_hook(residual_hook))
    try:
        outputs = model(ids, use_cache=False, output_attentions=probe_attention)
        caps.logits = outputs.logits[0].detach()
        caps.attention = [a.detach().cpu() for a in outputs.attentions] if probe_attention else None
    finally:
        for handle in handles:
            handle.remove()
    caps.residuals = torch.stack([caps.residuals[i] for i in range(36)])
    assert torch.equal(caps.residuals[22],
                       (caps.prenorm[pos] + getattr(caps, 'edited_mlp_output', caps.mlp_output)[pos]).cpu().float())
    return caps


def tc_contribution(caps, pos, feature):
    return caps.tc_activations[pos, feature].float() * tc['W_dec'][feature].float()


def molt_parts(caps, pos, feature):
    x = (caps.prenorm.float() - mw['input_standardizer.mean'][22]) / mw['input_standardizer.std'][22]
    raw = F.linear(x, mw['e.weight'], mw['e.bias'])
    gate = raw * ((raw > mw['nonlinearity.theta']) & (raw > 0))
    ungated = ((x[pos] @ mw[f'V_{feature}']) @ mw[f'U_{feature}']) * mw['output_standardizer.std'][22]
    return {'gate': float(gate[pos, feature]), 'contribution': gate[pos, feature] * ungated}


@torch.inference_mode()
def main():
    ids, pos, tokens = encode(P['receiver'], 'small')
    donor_ids, donor_pos, donor_tokens = encode(P['donor'], 'hot')
    assert donor_pos == pos
    clean = forward(ids, pos, probe_attention=True)
    donor = forward(donor_ids, donor_pos)
    for caps in [clean, donor]:
        caps.tc_activations = F.relu(F.linear(caps.mlp_input, tc['W_enc'], tc['b_enc']))
    prior = torch.load(ROOT / 'assets/prior-layer22-activations.pt', map_location='cpu', weights_only=True)
    prior_checks = {}
    for name, caps in [('receiver', clean), ('donor', donor)]:
        prior_checks[name + '_mlp_input_bitwise_equal'] = torch.equal(caps.mlp_input[pos].cpu(), prior[name + '_mlp_input'])
        prior_checks[name + '_tc_activations_bitwise_equal'] = torch.equal(caps.tc_activations[pos].cpu(), prior[name + '_activations'])

    tc_ids = P['transcoder_small_ids'] + P['transcoder_hot_ids']
    tc_unit = sum(tc_contribution(donor, pos, feature) - tc_contribution(clean, pos, feature)
                  for feature in tc_ids)
    molt_ids = P['molt_small_ids'] + P['molt_hot_ids']
    molt_rec = {feature: molt_parts(clean, pos, feature) for feature in molt_ids}
    molt_don = {feature: molt_parts(donor, pos, feature) for feature in molt_ids}
    molt_delta = P['molt_strength'] * sum(molt_don[f]['contribution'] - molt_rec[f]['contribution'] for f in molt_ids)
    fixed = forward(ids, pos, molt_delta, probe_attention=True)
    assert torch.equal(clean.residuals[:22], fixed.residuals[:22])
    fixed_change = fixed.residuals - clean.residuals
    R = {'protocol': P, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
         'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
         'dtype': 'BF16', 'attention': model.config._attn_implementation},
         'code_sha256': sha(Path(__file__)), 'tokens': tokens, 'donor_tokens': donor_tokens,
         'clean': summarize(clean.logits), 'donor': summarize(donor.logits), 'molt_fixed': summarize(fixed.logits),
         'checks': prior_checks, 'trials': [], 'status': 'running',
         'features': {'transcoder': [{'id': f, 'layer': 22, 'clean_activation': float(clean.tc_activations[pos, f]),
                       'donor_activation': float(donor.tc_activations[pos, f])} for f in tc_ids],
                      'molt': [{'id': f, 'layer': 22, 'clean_gate': molt_rec[f]['gate'],
                       'donor_gate': molt_don[f]['gate']} for f in molt_ids]},
         'geometry': {'molt_injected_norm': float(molt_delta.norm()), 'tc_unit_norm': float(tc_unit.norm()),
                      'intended_edit_cosine': cosine(molt_delta, tc_unit),
                      'clean_residual_norm': float(clean.residuals[22].norm()),
                      'molt_residual_norm': float(fixed.residuals[22].norm()),
                      'molt_actual_residual_change_norm': float(fixed_change[22].norm())}}
    save('results.json', R)
    tc_residuals = []
    tc_next_token_logits = []
    tc19 = None
    for strength in P['transcoder_strengths']:
        delta = strength * tc_unit
        probe = strength == P['transcoder_reference_strength']
        caps = forward(ids, pos, delta, probe_attention=probe)
        assert torch.equal(clean.residuals[:22], caps.residuals[:22])
        if strength == 0:
            assert torch.equal(clean.logits, caps.logits)
            assert torch.equal(clean.residuals, caps.residuals)
            R['checks']['zero_strength_bitwise_equal'] = True
        change = caps.residuals - clean.residuals
        row = {'strength': strength, **summarize(caps.logits),
               'cosine_residual': cosine(fixed.residuals[22], caps.residuals[22]),
               'cosine_residual_changes': cosine(fixed_change[22], change[22]),
               'cosine_intended_edits': cosine(molt_delta, delta.cpu()),
               'tc_injected_norm': float(delta.norm()), 'tc_residual_norm': float(caps.residuals[22].norm()),
               'tc_actual_change_norm': float(change[22].norm()),
               'relative_l2_to_molt_residual': float((caps.residuals[22] - fixed.residuals[22]).norm() / fixed.residuals[22].norm()),
               'downstream_residual_cosines': [cosine(fixed.residuals[i], caps.residuals[i]) for i in range(36)],
               'downstream_change_cosines': [cosine(fixed_change[i], change[i]) for i in range(36)]}
        tc_residuals.append(caps.residuals.numpy())
        tc_next_token_logits.append(caps.logits[-1].cpu().float().numpy())
        R['trials'].append(row)
        save('results.json', R)
        print('STRENGTH', strength, 'cos_residual', round(row['cosine_residual'], 7),
              'cos_change', row['cosine_residual_changes'], 'P_cold', round(row['P_cold'], 7), flush=True)
        if probe:
            tc19 = caps

    def attention_changes(caps):
        return [{'layer': i, 'maximum_absolute_change': float((a.float() - b.float()).abs().max())}
                for i, (a, b) in enumerate(zip(clean.attention, caps.attention))]

    R['attention_changes'] = {'molt4.5': attention_changes(fixed), 'transcoder19.5': attention_changes(tc19)}
    assert all(x['maximum_absolute_change'] == 0 for method in R['attention_changes'].values() for x in method[:23])
    assert all(any(x['maximum_absolute_change'] > 0 for x in method[23:]) for method in R['attention_changes'].values())
    repeated_molt = forward(ids, pos, molt_delta)
    repeated_tc = forward(ids, pos, 19.5 * tc_unit)
    assert torch.equal(repeated_molt.logits, fixed.logits) and torch.equal(repeated_molt.residuals, fixed.residuals)
    assert torch.equal(repeated_tc.logits, tc19.logits) and torch.equal(repeated_tc.residuals, tc19.residuals)
    restored = forward(ids, pos)
    assert torch.equal(restored.logits, clean.logits) and torch.equal(restored.residuals, clean.residuals)
    R['checks'].update({'attention_forwards_unchanged': True, 'downstream_attention_changed_both_methods': True,
                       'only_layer22_token9_directly_edited': True, 'residual_addition_boundary_verified': True,
                       'reference_conditions_repeat_bitwise': True, 'restoration_bitwise_equal': True,
                       'upstream_residuals_unchanged_every_trial': True})
    prior_reference = json.loads((ROOT / 'assets/prior-reference.json').read_text())
    R['prior_probability_comparison'] = []
    for old in prior_reference['conditions']:
        new = R['molt_fixed'] if old['group'] == 'molt22-pair' else summarize(tc19.logits)
        R['prior_probability_comparison'].append({'condition': old['group'], 'prior_P_cold': old['P_cold'],
                                                 'current_P_cold': new['P_cold'], 'absolute_difference': abs(new['P_cold'] - old['P_cold'])})
    np.savez_compressed(OUT / 'residual-vectors.npz', strengths=np.array(P['transcoder_strengths']),
                        clean=clean.residuals.numpy(), donor=donor.residuals.numpy(),
                        molt=fixed.residuals.numpy(), transcoder=np.stack(tc_residuals),
                        molt_injected_delta=molt_delta.cpu().numpy(), tc_unit_delta=tc_unit.cpu().numpy())
    np.savez_compressed(OUT / 'next-token-logits.npz', clean=clean.logits[-1].cpu().float().numpy(),
                        donor=donor.logits[-1].cpu().float().numpy(),
                        molt=fixed.logits[-1].cpu().float().numpy(),
                        transcoder=np.stack(tc_next_token_logits))
    columns = ['strength', 'P_cold', 'P_large', 'cosine_residual', 'cosine_residual_changes',
               'cosine_intended_edits', 'tc_injected_norm', 'tc_residual_norm', 'tc_actual_change_norm',
               'relative_l2_to_molt_residual', 'top_token']
    with (OUT / 'strength-sweep.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(R['trials'])
    R.update({'status': 'complete', 'seconds': time.monotonic() - START,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
    save('results.json', R)
    save('export-checksums.json', {p.name: sha(p) for p in sorted(OUT.iterdir())
                                  if p.is_file() and p.name != 'export-checksums.json'})
    print('COMPLETE', len(R['trials']), 'trials', flush=True)


if __name__ == '__main__':
    main()
