"""Single-layer original-model contribution edits with fully live attention."""
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT / 'protocol.json').read_text())
LABELS = {int(k): v for k, v in json.loads((ROOT / 'assets/label-map.json').read_text()).items()}
torch.manual_seed(0)
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
START = time.monotonic()


def save(name, data):
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(OUT / name)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'models/qwen3-4b', local_files_only=True, dtype=torch.bfloat16,
    attn_implementation='eager').cuda().eval().requires_grad_(False)
model.config.use_cache = False
assert model.config._attn_implementation == 'eager'
attention_forwards = [layer.self_attn.forward for layer in model.model.layers]
tc = {k: v.cuda() for k, v in load_file(ROOT / 'models/transcoder/layer_22.safetensors').items()}
assert tc['W_enc'].dtype == tc['W_dec'].dtype == torch.bfloat16
assert tuple(tc['W_enc'].shape) == tuple(tc['W_dec'].shape) == (163840, 2560)
mw = {k: v.cuda() for k, v in load_file(ROOT / 'assets/molt-22-selected.safetensors').items()}
R = {'protocol': P, 'runtime': {'python': platform.python_version(),
     'torch': torch.__version__, 'transformers': transformers.__version__,
     'gpu': torch.cuda.get_device_name(), 'dtype': 'BF16', 'attention': 'eager, fully live'},
     'code_sha256': digest(Path(__file__)), 'prompts': [], 'candidates': [],
     'groups': [], 'trials': [], 'generation': [], 'checks': {}, 'status': 'running'}
TRACES = {}


def encode(prompt, word):
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    batch = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                      return_tensors='pt')
    offsets = batch.pop('offset_mapping')[0].tolist()
    begin = text.index(word, text.index(prompt))
    positions = [i for i, (a, b) in enumerate(offsets)
                 if b > a and b > begin and a < begin + len(word)]
    assert len(positions) == 1
    ids = batch['input_ids'].cuda()
    return ids, positions[0], [{'position': i, 'id': int(v),
            'text': tokenizer.decode([int(v)])} for i, v in enumerate(ids[0])]


@torch.inference_mode()
def capture(ids):
    caps = SimpleNamespace()
    def mlp(_module, args, out):
        caps.input = args[0][0].detach().clone()
        caps.output = out[0].detach().clone()
    def pre(_module, args):
        caps.prenorm = args[0][0].detach().clone()
    layer = model.model.layers[22]
    handles = [layer.mlp.register_forward_hook(mlp),
               layer.post_attention_layernorm.register_forward_pre_hook(pre)]
    try:
        caps.logits = model(ids, use_cache=False).logits[0].detach()
    finally:
        for handle in handles:
            handle.remove()
    caps.activations = F.relu(F.linear(caps.input, tc['W_enc'], tc['b_enc']))
    return caps


def summarize(logits):
    p = logits[-1].float().softmax(-1)
    values, indices = p.topk(8)
    return {'P_cold': float(p[87072]), 'P_large': float(p[16767]),
            'P_Cold': float(p[76418]), 'P_space_cold': float(p[9255]),
            'top_token': tokenizer.decode([int(indices[0])]),
            'top_tokens': [{'id': int(i), 'text': tokenizer.decode([int(i)]),
                            'probability': float(v)} for i, v in zip(indices, values)]}


@torch.inference_mode()
def forward(ids, delta, pos, trace_key=None, attention_probe=False):
    assert all(layer.self_attn.forward == original for layer, original in
               zip(model.model.layers, attention_forwards))
    assert not any(layer.mlp._forward_hooks for layer in model.model.layers)
    def hook(_module, _args, out):
        res = out.clone()
        res[0, pos] = out[0, pos] + delta.to(out.dtype)
        mask = torch.ones(out.shape[1], dtype=torch.bool, device='cuda')
        mask[pos] = False
        assert torch.equal(res[:, mask], out[:, mask])
        assert torch.equal(res[0, pos], out[0, pos] + delta.to(out.dtype))
        if trace_key:
            TRACES[trace_key] = {'before': out[0, pos].cpu(), 'after': res[0, pos].cpu(),
                                 'delta_fp32': delta.cpu(), 'position': pos, 'layer': 22,
                                 'full_before': out.cpu(), 'full_after': res.cpu(),
                                 'other_positions_unchanged': True}
        return res
    handle = model.model.layers[22].mlp.register_forward_hook(hook)
    try:
        out = model(ids, use_cache=False, output_attentions=attention_probe)
        logits = out.logits[0].detach()
        if trace_key:
            TRACES[trace_key]['next_token_logits'] = logits[-1].cpu()
        attention = [a.detach().cpu() for a in out.attentions] if attention_probe else None
    finally:
        handle.remove()
    return logits, attention


def vector(caps, pos, feature_ids):
    if not feature_ids:
        return torch.zeros(2560, device='cuda')
    return (caps.activations[pos, feature_ids].float()[:, None] *
            tc['W_dec'][feature_ids].float()).sum(0)


def parts(group, rec, don, pos):
    small_clean = vector(rec, pos, group['small'])
    small_target = vector(don, pos, group['small'])
    hot_clean = vector(rec, pos, group['hot'])
    hot_target = vector(don, pos, group['hot'])
    return small_clean, small_target, hot_clean, hot_target


def trial(prompt_id, ids, pos, delta, group, mode, strength, component='combined',
          trace=False, extra=None):
    index = len(R['trials'])
    key = f'{prompt_id}/{group}/{mode}/{component}/{strength}/{index}'
    logits, _ = forward(ids, delta, pos, key if trace else None)
    row = {'key': key, 'prompt_id': prompt_id, 'group': group, 'mode': mode,
           'component': component, 'strength': strength, 'layer': 22, 'position': pos,
           'delta_norm': float(delta.norm()), **summarize(logits)}
    if extra:
        row.update(extra)
    R['trials'].append(row)
    with (OUT / 'trials.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    if index % 20 == 0 or row['P_cold'] >= .99:
        print('TRIAL', index, prompt_id, group, mode, strength, row['top_token'],
              round(row['P_cold'], 7), flush=True)
    return row


def make_group(groups, name, small, hot):
    small, hot = sorted(set(small)), sorted(set(hot))
    if not small or not hot:
        return
    if any(x['small'] == small and x['hot'] == hot for x in groups):
        return
    groups.append({'name': name, 'small': small, 'hot': hot,
                   'n_features': len(small) + len(hot)})


def reconstruction(caps, pos):
    reconstructed = caps.activations[pos].float() @ tc['W_dec'].float()
    if 'b_dec' in tc:
        reconstructed += tc['b_dec'].float()
    # Some formats include a linear skip path; explicitly record rather than ignore.
    if 'W_skip' in tc:
        reconstructed += caps.input[pos].float() @ tc['W_skip'].float().T
    actual = caps.output[pos].float()
    return {'relative_l2_error': float((reconstructed - actual).norm() / actual.norm()),
            'cosine': float(F.cosine_similarity(reconstructed, actual, dim=0)),
            'native_norm': float(actual.norm()), 'reconstructed_norm': float(reconstructed.norm()),
            'keys': sorted(tc)}


def molt_contribution(caps, pos, index):
    x = (caps.prenorm.float() - mw['input_standardizer.mean'][22]) / mw['input_standardizer.std'][22]
    raw = F.linear(x, mw['e.weight'], mw['e.bias'])
    gate = raw * ((raw > mw['nonlinearity.theta']) & (raw > 0))
    uv = (x[pos] @ mw[f'V_{index}']) @ mw[f'U_{index}']
    uv *= mw['output_standardizer.std'][22]
    return gate[pos, index] * uv


@torch.inference_mode()
def generate(prompt_id, receiver, donor, group, mode, strength):
    ids, pos, _ = encode(receiver, 'small')
    dids, dpos, _ = encode(donor, 'hot')
    donor_caps = capture(dids)
    generated = []
    for step in range(8):
        caps = capture(ids)
        a, b, c, d = parts(group, caps, donor_caps, pos)
        delta = strength * (b - a + d - c) if mode == 'coupled' else -a + strength * (d - c)
        logits, _ = forward(ids, delta, pos)
        nxt = int(logits[-1].argmax())
        generated.append(nxt)
        if nxt in {tokenizer.eos_token_id, 151645}:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device='cuda')], dim=1)
    R['generation'].append({'prompt_id': prompt_id, 'group': group['name'],
                            'mode': mode, 'strength': strength, 'tokens': generated,
                            'text': tokenizer.decode(generated, skip_special_tokens=True),
                            'history': 'receiver-generated; donor answer never supplied'})


@torch.inference_mode()
def main():
    ids, pos, tokens = encode(P['receiver'], 'small')
    dids, dpos, dtokens = encode(P['donor'], 'hot')
    assert pos == dpos == 9 and ids.shape == dids.shape == (1, 29)
    rec, don = capture(ids), capture(dids)
    R['prompts'].append({'id': 'primary', 'receiver': P['receiver'], 'donor': P['donor'],
                        'position': pos, 'readout_position': 28, 'tokens': tokens,
                        'donor_tokens': dtokens, 'baseline': summarize(rec.logits),
                        'donor_baseline': summarize(don.logits)})
    R['reconstruction'] = {'receiver': reconstruction(rec, pos), 'donor': reconstruction(don, pos)}
    # Exact encoder/decoder row equality with the previously downloaded reference selection.
    blob = (ROOT / 'assets/published-tc22-selected.bin').read_bytes()
    for i, feature in enumerate(P['published_small_ids']):
        row = torch.frombuffer(bytearray(blob[i * 10242:(i + 1) * 10242]), dtype=torch.bfloat16)
        assert torch.equal(tc['W_enc'][feature].cpu(), row[:2560])
        assert torch.equal(tc['W_dec'][feature].cpu(), row[2560:5120])
        assert torch.equal(tc['b_enc'][feature].cpu(), row[5120])
    R['checks']['published_weight_rows_exact'] = True
    torch.save({'receiver_activations': rec.activations[pos].cpu(),
                'donor_activations': don.activations[pos].cpu(),
                'receiver_mlp_input': rec.input[pos].cpu(), 'donor_mlp_input': don.input[pos].cpu(),
                'receiver_mlp_output': rec.output[pos].cpu(), 'donor_mlp_output': don.output[pos].cpu(),
                'receiver_next_token_logits': rec.logits[-1].cpu(),
                'donor_next_token_logits': don.logits[-1].cpu(),
                'position': pos}, OUT / 'full-layer22-activations.pt')
    for role in ['small', 'hot']:
        for textid, label in P[role + '_pool'].items():
            feature = int(textid)
            clean, donor = float(rec.activations[pos, feature]), float(don.activations[pos, feature])
            R['candidates'].append({'role': role, 'feature': feature, 'layer': 22,
                'label': label, 'clean_activation': clean, 'donor_activation': donor,
                'difference': donor - clean,
                'eligible': clean > donor and clean > 0 if role == 'small' else donor > clean and donor > 0})
    sources = sorted([c for c in R['candidates'] if c['role'] == 'small' and c['eligible']],
                     key=lambda x: x['clean_activation'] - x['donor_activation'], reverse=True)
    hots = sorted([c for c in R['candidates'] if c['role'] == 'hot' and c['eligible']],
                  key=lambda x: x['difference'], reverse=True)
    print('ACTIVE_SMALL', [(c['feature'], c['label'], c['clean_activation']) for c in sources], flush=True)
    print('ACTIVE_HOT', [(c['feature'], c['label'], c['donor_activation']) for c in hots], flush=True)
    groups = []
    top_small = [c['feature'] for c in sources[:6]]
    top_hot = [c['feature'] for c in hots[:6]]
    published = P['published_small_ids']
    for hot in top_hot:
        make_group(groups, f'published3-hot{hot}', published, [hot])
    pair_sources = list(dict.fromkeys(published + top_small))
    for small in pair_sources:
        for hot in top_hot:
            make_group(groups, f'pair-{small}-{hot}', [small], [hot])
    for ks in [1, 3, 6, len(sources)]:
        for kh in [1, 3, 6, len(hots)]:
            make_group(groups, f'ranked-small{ks}-hot{kh}',
                       [x['feature'] for x in sources[:ks]], [x['feature'] for x in hots[:kh]])
    make_group(groups, 'published3-all-labelled-hot', published, [x['feature'] for x in hots])
    R['groups'] = groups
    # Eligibility and groups saved before any manipulated P(cold) measurement.
    save('selection-before-interventions.json', {'candidates': R['candidates'], 'groups': groups})
    all_ids = sorted({i for g in groups for i in g['small'] + g['hot']})
    if all_ids:
        save_file({'W_enc': tc['W_enc'][all_ids].cpu(), 'W_dec': tc['W_dec'][all_ids].cpu(),
                   'b_enc': tc['b_enc'][all_ids].cpu()}, OUT / 'selected-feature-weights.safetensors')
    save('selected-feature-indices.json', all_ids)
    zero = torch.zeros(2560, device='cuda')
    zlogits, _ = forward(ids, zero, pos, 'zero_delta')
    assert torch.equal(zlogits, rec.logits)
    R['checks']['zero_delta_bitwise_equal'] = True
    self_delta = vector(rec, pos, published) - vector(rec, pos, published)
    self_logits, _ = forward(ids, self_delta, pos, 'self_patch')
    assert torch.equal(self_logits, rec.logits)
    R['checks']['self_patch_bitwise_equal'] = True
    trial('primary', ids, pos, don.output[pos].float() - rec.output[pos].float(),
          'whole-native-mlp22', 'control', 1, trace=True)
    full_delta = (don.activations[pos].float() - rec.activations[pos].float()) @ tc['W_dec'].float()
    trial('primary', ids, pos, full_delta, 'whole-tc22-features', 'control', 1, trace=True)
    molt_delta = 4.5 * (molt_contribution(don, pos, 2339) - molt_contribution(rec, pos, 2339)
                        + molt_contribution(don, pos, 941) - molt_contribution(rec, pos, 941))
    trial('primary', ids, pos, molt_delta, 'molt22-pair', 'control', 4.5, trace=True,
          extra={'small_ids': [2339], 'hot_ids': [941]})
    save('results.json', R)
    for group in groups:
        a, b, c, d = parts(group, rec, don, pos)
        unit = b - a + d - c
        for strength in P['coupled_strengths']:
            trial('primary', ids, pos, strength * unit, group['name'], 'coupled', strength)
        # Remove small once, vary only the hot patch for published-source groups.
        if group['small'] == sorted(published):
            for strength in P['coupled_strengths']:
                trial('primary', ids, pos, -a + strength * (d - c), group['name'],
                      'zero_small_vary_hot', strength)
        save('results.json', R)
    exploratory = [t for t in R['trials'] if t['mode'] in ['coupled', 'zero_small_vary_hot']]
    if not exploratory:
        R['status'] = 'no_active_labelled_hot_features'
        save('results.json', R)
        return
    # Refine each of the three strongest recipes around its coarse peak, without
    # assuming monotonicity. Keep the original coarse observations intact.
    peak_by_recipe = {}
    for t in exploratory:
        key = (t['group'], t['mode'])
        if key not in peak_by_recipe or t['P_cold'] > peak_by_recipe[key]['P_cold']:
            peak_by_recipe[key] = t
    peaks = sorted(peak_by_recipe.values(), key=lambda x: x['P_cold'], reverse=True)[:3]
    for peak in peaks:
        group = next(g for g in groups if g['name'] == peak['group'])
        a, b, c, d = parts(group, rec, don, pos)
        strengths = P['coupled_strengths']
        j = strengths.index(peak['strength'])
        lo, hi = strengths[max(j - 1, 0)], strengths[min(j + 1, len(strengths) - 1)]
        for strength in torch.linspace(lo, hi, 25).tolist():
            delta = strength * (b - a + d - c) if peak['mode'] == 'coupled' else -a + strength * (d - c)
            trial('primary', ids, pos, delta, group['name'], peak['mode'], strength,
                  extra={'stage': 'local_peak_refinement'})
        save('results.json', R)
    exploratory = [t for t in R['trials'] if t['mode'] in ['coupled', 'zero_small_vary_hot']]
    best = max(exploratory, key=lambda x: x['P_cold'])
    pairs = [t for t in exploratory if next(g for g in groups if g['name'] == t['group'])['n_features'] == 2]
    best_pair = max(pairs, key=lambda x: x['P_cold']) if pairs else None
    successes = [t for t in exploratory if t['P_cold'] >= .99]
    smallest = min(successes, key=lambda x: (next(g for g in groups if g['name'] == x['group'])['n_features'], x['strength'])) if successes else None
    R['best'] = best
    R['best_pair'] = best_pair
    R['smallest_successful_tested'] = smallest
    chosen = smallest or best
    group = next(g for g in groups if g['name'] == chosen['group'])
    a, b, c, d = parts(group, rec, don, pos)
    strength = chosen['strength']
    delta = strength * (b - a + d - c) if chosen['mode'] == 'coupled' else -a + strength * (d - c)
    clean_logits, clean_attn = forward(ids, zero, pos, attention_probe=True)
    repeat_logits, edited_attn = forward(ids, delta, pos, 'chosen_repeat', attention_probe=True)
    repeat = summarize(repeat_logits)
    assert abs(repeat['P_cold'] - chosen['P_cold']) < 1e-7
    R['checks']['chosen_repeat_exact_probability'] = True
    R['checks']['attention_forward_functions_unchanged'] = all(
        layer.self_attn.forward == original for layer, original in zip(model.model.layers, attention_forwards))
    R['attention_probe'] = [{'layer': i, 'maximum_pattern_change': float((x.float() - y.float()).abs().max())}
                            for i, (x, y) in enumerate(zip(clean_attn, edited_attn))]
    torch.save({'clean': clean_attn, 'edited': edited_attn}, OUT / 'attention-patterns.pt')
    suppression_strength = strength if chosen['mode'] == 'coupled' else 1
    trial('primary', ids, pos, -suppression_strength * a, group['name'], 'control', suppression_strength,
          component='suppress_only', trace=True)
    trial('primary', ids, pos, strength * (d - c), group['name'], 'control', strength,
          component='inject_only', trace=True)
    rand = torch.randn_like(delta)
    rand *= delta.norm() / rand.norm()
    trial('primary', ids, pos, rand, 'random-norm-matched', 'control', strength, trace=True)
    generate('primary', P['receiver'], P['donor'], group, chosen['mode'], strength)
    for i, prompt in enumerate(P['heldout_prompts']):
        donor = prompt.replace('small', 'hot')
        hids, hpos, htoks = encode(prompt, 'small')
        hdids, hdpos, hdtoks = encode(donor, 'hot')
        assert hpos == hdpos and hids.shape == hdids.shape
        hrec, hdon = capture(hids), capture(hdids)
        prompt_id = f'heldout-{i}'
        R['prompts'].append({'id': prompt_id, 'receiver': prompt, 'donor': donor,
            'position': hpos, 'tokens': htoks, 'donor_tokens': hdtoks,
            'baseline': summarize(hrec.logits), 'donor_baseline': summarize(hdon.logits)})
        ha, hb, hc, hd = parts(group, hrec, hdon, hpos)
        hdelta = strength * (hb - ha + hd - hc) if chosen['mode'] == 'coupled' else -ha + strength * (hd - hc)
        trial(prompt_id, hids, hpos, hdelta, group['name'], chosen['mode'], strength, trace=True)
        generate(prompt_id, prompt, donor, group, chosen['mode'], strength)
    restored = model(ids, use_cache=False).logits[0]
    assert torch.equal(restored, rec.logits)
    R['checks']['restored_model_bitwise_equal'] = True
    R['checks']['all_edits_only_layer22_selected_position'] = True
    R['status'] = 'complete'
    R['seconds'] = time.monotonic() - START
    R['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    save('results.json', R)
    torch.save(TRACES, OUT / 'intervention-traces.pt')
    print('COMPLETE', json.dumps({'best': best, 'best_pair': best_pair,
          'smallest_successful_tested': smallest, 'trial_count': len(R['trials'])}), flush=True)


if __name__ == '__main__':
    main()
