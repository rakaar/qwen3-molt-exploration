"""Single-layer, single-position MOLT spider-to-ant contribution experiments."""
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
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
model.config.use_cache = False
original_attentions = [b.self_attn.forward for b in model.model.layers]
layers = sorted({x['layer'] for x in P['features']})
weights = {l: {k: v.cuda() for k, v in load_file(ROOT / f'weights/molt-{l}-selected.safetensors').items()} for l in layers}
number_ids = {digit: tokenizer.encode(digit, add_special_tokens=False)[0] for digit in ['6', '8', '4']}
assert all(len(tokenizer.encode(d, add_special_tokens=False)) == 1 for d in number_ids)


def save(name, value):
    temporary = OUT / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUT / name)


def check_time():
    assert time.monotonic() - START < P['max_runtime_minutes'] * 60, 'Runtime ceiling reached'


def encode(text, cue):
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True, return_tensors='pt')
    offsets = batch.pop('offset_mapping')[0].tolist()
    user_start = chat.index(text)
    start = chat.index(cue, user_start)
    def last_overlap(a, b):
        ids = [i for i, (x, y) in enumerate(offsets) if y > x and y > a and x < b]
        assert ids
        return ids[-1]
    sites = {'cue_end': last_overlap(start, start + len(cue)), 'readout': batch['input_ids'].shape[1] - 1}
    if '?' in text:
        q = chat.index('?', user_start)
        sites['question_end'] = last_overlap(q, q + 1)
        if 'have' in chat[start:q]:
            h = chat.index('have', start)
            sites['have'] = last_overlap(h, h + 4)
    ids = batch['input_ids'].cuda()
    return {'text': text, 'chat': chat, 'ids': ids, 'sites': sites,
            'tokens': [{'position': i, 'id': int(t), 'text': tokenizer.decode([int(t)])} for i, t in enumerate(ids[0])]}


def summarize(logits):
    probabilities = logits.float().softmax(-1)
    values, ids = probabilities.topk(5)
    return {'P6': float(probabilities[number_ids['6']]), 'P8': float(probabilities[number_ids['8']]),
            'P4': float(probabilities[number_ids['4']]), 'top_token': tokenizer.decode([int(ids[0])]),
            'top_tokens': [{'id': int(i), 'text': tokenizer.decode([int(i)]), 'probability': float(v)} for i, v in zip(ids, values)]}


@torch.inference_mode()
def forward(record, edit=None, probe=False):
    check_time()
    assert all(b.self_attn.forward == a for b, a in zip(model.model.layers, original_attentions))
    assert all(not b.mlp._forward_hooks for b in model.model.layers)
    caps = SimpleNamespace(prenorm={}, mlp_input={}, mlp_output={}, residuals={})
    handles = []
    for layer, block in enumerate(model.model.layers):
        if layer in layers:
            def pre(_module, args, layer=layer):
                caps.prenorm[layer] = args[0][0].detach().clone()
            def mlp(_module, args, output, layer=layer):
                caps.mlp_input[layer] = args[0][0].detach().clone()
                caps.mlp_output[layer] = output[0].detach().clone()
                if edit is None or edit['layer'] != layer:
                    return output
                pos = edit['position']
                changed = output.clone()
                changed[0, pos] = output[0, pos] + edit['delta'].to(output.dtype)
                mask = torch.ones(output.shape[1], device=output.device, dtype=torch.bool)
                mask[pos] = False
                assert torch.equal(output[:, mask], changed[:, mask])
                caps.edited_mlp_output = changed[0].detach().clone()
                return changed
            handles += [block.post_attention_layernorm.register_forward_pre_hook(pre), block.mlp.register_forward_hook(mlp)]
        def residual(_module, _args, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            caps.residuals[layer] = hidden[0].detach().clone()
        handles.append(block.register_forward_hook(residual))
    try:
        output = model(record['ids'], use_cache=False, output_attentions=probe)
        caps.logits = output.logits[0, -1].detach().clone()
        caps.attention = [x.detach().cpu() for x in output.attentions] if probe else None
    finally:
        for handle in handles:
            handle.remove()
    if edit:
        l, p = edit['layer'], edit['position']
        assert torch.equal(caps.residuals[l][p], caps.prenorm[l][p] + caps.edited_mlp_output[p])
    return caps


def parts(caps, layer, feature):
    w = weights[layer]
    x = (caps.prenorm[layer].float() - w['input_standardizer.mean'][layer]) / w['input_standardizer.std'][layer]
    assert torch.isfinite(x).all()
    raw = F.linear(x, w['e.weight'], w['e.bias'])
    gates = raw * ((raw > w['nonlinearity.theta']) & (raw > 0))
    lowrank = ((x @ w[f'V_{feature}']) @ w[f'U_{feature}']) * w['output_standardizer.std'][layer]
    return {'gates': gates[:, feature], 'all_gates': gates, 'lowrank': lowrank,
            'contributions': gates[:, feature, None] * lowrank}


def generation(record, edit=None):
    handle = None
    if edit:
        def hook(_module, _args, output):
            # Reapply to the receiver cue during uncached generation; a single original position.
            changed = output.clone()
            changed[0, edit['position']] = output[0, edit['position']] + edit['delta'].to(output.dtype)
            return changed
        handle = model.model.layers[edit['layer']].mlp.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model.generate(record['ids'], max_new_tokens=8, do_sample=False, use_cache=False,
                pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, record['ids'].shape[1]:], skip_special_tokens=True).strip()
    finally:
        if handle:
            handle.remove()


@torch.inference_mode()
def main():
    records, baselines, component = {}, {}, {}
    R = {'protocol': P, 'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
         'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(), 'attention': 'unfrozen eager', 'dtype': 'BF16'},
         'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'number_token_ids': number_ids,
         'baselines': {}, 'activation_maps': {}, 'trials': [], 'checks': {}, 'status': 'probing'}
    exported_baselines, logits, deltas, actual_changes = {}, [], [], []
    for pair in P['pairs']:
        for side in ['receiver', 'donor']:
            name = pair['id'] + '_' + side
            records[name] = encode(pair[side], pair[side + '_cue'])
    for control in P['controls']:
        records['control_' + control['id']] = encode(control['text'], control['cue'])
    for name, record in records.items():
        caps = forward(record)
        baselines[name] = caps
        answer = generation(record) if not name.startswith('control_') or any(c['id'] == name[8:] and c['expected'] is not None for c in P['controls']) else None
        R['baselines'][name] = {**summarize(caps.logits), 'generation': answer, 'sites': record['sites'], 'tokens': record['tokens'], 'text': record['text'], 'logits_index': len(logits)}
        logits.append(caps.logits.cpu().float().numpy())
        R['activation_maps'][name] = []
        for feature in P['features']:
            l, f = feature['layer'], feature['id']
            part = parts(caps, l, f)
            component[(name, l, f)] = part
            sites = record['sites']
            R['activation_maps'][name].append({'layer': l, 'id': f, 'label': feature['label'],
                'gates_by_token': part['gates'].cpu().tolist(),
                'site_gates': {s: float(part['gates'][p]) for s, p in sites.items()},
                'site_contribution_norms': {s: float(part['contributions'][p].norm()) for s, p in sites.items()},
                'cue_top_gates': [{'id': int(i), 'gate': float(v)} for v, i in zip(*part['all_gates'][sites['cue_end']].topk(10))]})
            prefix = f'{name}_L{l}'
            for kind in ['prenorm', 'mlp_input', 'mlp_output']:
                exported_baselines[prefix + '_' + kind] = getattr(caps, kind)[l].cpu().float().numpy()
            exported_baselines[prefix + '_gates'] = part['all_gates'].cpu().numpy()
        print('BASELINE', name, 'generation', answer, 'P6', R['baselines'][name]['P6'], 'P8', R['baselines'][name]['P8'], flush=True)
    valid = [p['id'] for p in P['pairs'] if R['baselines'][p['id'] + '_receiver']['generation'] == '8' and R['baselines'][p['id'] + '_donor']['generation'] == '6']
    R['checks']['valid_baseline_pairs'] = valid
    R['status'] = 'awaiting_baseline_review'
    save('results.json', R)
    np.savez_compressed(OUT / 'baseline-captures.npz', **exported_baselines)
    print('BASELINE_REVIEW_READY', valid, flush=True)
    if P['pause_after_baseline']:
        until = time.monotonic() + 600
        while not (ROOT / 'CONTINUE_SWEEP').exists():
            check_time()
            assert time.monotonic() < until, 'Baseline review timeout'
            time.sleep(1)
    assert P['primary_pair'] in valid, 'Primary baseline must be correct before manipulation'
    R['status'] = 'sweeping'
    primary_best = {}

    def trial(pair, feature, site, method, strength):
        rec_name, don_name = pair['id'] + '_receiver', pair['id'] + '_donor'
        l, f = feature['layer'], feature['id']
        rec, don = records[rec_name], records[don_name]
        rp, dp = rec['sites'][site], don['sites'][site]
        a, b = component[(rec_name, l, f)], component[(don_name, l, f)]
        if method == 'full_contribution':
            unit = b['contributions'][dp] - a['contributions'][rp]
        elif method == 'gate_only':
            unit = (b['gates'][dp] - a['gates'][rp]) * a['lowrank'][rp]
        elif method == 'ablate_receiver':
            unit = -a['contributions'][rp]
        else:
            assert method == 'full_mlp_positive_control'
            unit = baselines[don_name].mlp_output[l][dp].float() - baselines[rec_name].mlp_output[l][rp].float()
        delta = strength * unit
        caps = forward(rec, {'layer': l, 'position': rp, 'delta': delta})
        clean = baselines[rec_name]
        assert all(torch.equal(caps.residuals[k], clean.residuals[k]) for k in range(l))
        assert torch.equal(caps.mlp_input[l], clean.mlp_input[l])
        if strength == 0 or delta.norm() == 0:
            assert torch.equal(caps.logits, clean.logits)
        change = caps.edited_mlp_output[rp].float() - clean.mlp_output[l][rp].float()
        row = {'pair': pair['id'], 'split': pair['split'], 'layer': l, 'feature': f,
               'role': feature['role'], 'label': feature['label'], 'site': site, 'method': method, 'strength': strength,
               'receiver_position': rp, 'donor_position': dp, 'receiver_token': rec['tokens'][rp]['text'], 'donor_token': don['tokens'][dp]['text'],
               'receiver_gate': float(a['gates'][rp]), 'donor_gate': float(b['gates'][dp]),
               'receiver_contribution_norm': float(a['contributions'][rp].norm()), 'donor_contribution_norm': float(b['contributions'][dp].norm()),
               'intended_delta_norm': float(delta.norm()), 'actual_mlp_change_norm': float(change.norm()),
               **summarize(caps.logits), 'logits_index': len(logits), 'vector_index': len(deltas)}
        logits.append(caps.logits.cpu().float().numpy())
        deltas.append(delta.cpu().numpy())
        actual_changes.append(change.cpu().numpy())
        R['trials'].append(row)
        return row

    for pair in P['pairs']:
        if pair['id'] not in valid:
            continue
        for feature in P['features']:
            for site in P['sites']:
                if pair['split'] == 'heldout':
                    best = primary_best[(feature['layer'], feature['id'], site)]
                    strengths = sorted(set([1, 32, best['strength']]))
                else:
                    strengths = P['strengths']
                full_rows = [trial(pair, feature, site, 'full_contribution', s) for s in strengths]
                best = max(full_rows, key=lambda x: x['P6'])
                if pair['split'] == 'primary':
                    primary_best[(feature['layer'], feature['id'], site)] = best
                if pair['split'] != 'heldout':
                    for method in ['gate_only', 'ablate_receiver']:
                        for strength in P['control_strengths']:
                            trial(pair, feature, site, method, strength)
                    trial(pair, feature, site, 'full_mlp_positive_control', 1)
                print('SWEEP', pair['id'], f'L{feature["layer"]}-T{feature["id"]}', site, 'best strength', best['strength'], 'P6', best['P6'], 'P8', best['P8'], flush=True)
                save('results.json', R)

    best_trials = []
    for pair in P['pairs']:
        rows = [x for x in R['trials'] if x['pair'] == pair['id'] and x['method'] == 'full_contribution' and not x['role'].startswith('lexical')]
        if rows:
            best_trials.append(max(rows, key=lambda x: x['P6']))
    R['selected_generation_checks'] = []
    R['attention_changes'] = []
    for row in best_trials:
        rec_name = row['pair'] + '_receiver'
        edit = {'layer': row['layer'], 'position': row['receiver_position'], 'delta': torch.tensor(deltas[row['vector_index']], device='cuda')}
        clean = forward(records[rec_name], probe=True)
        caps = forward(records[rec_name], edit, probe=True)
        R['selected_generation_checks'].append({'pair': row['pair'], 'trial_vector_index': row['vector_index'], 'generation': generation(records[rec_name], edit)})
        changes = [{'layer': i, 'maximum_absolute_change': float((a.float() - b.float()).abs().max())} for i, (a, b) in enumerate(zip(clean.attention, caps.attention))]
        assert all(x['maximum_absolute_change'] == 0 for x in changes[:row['layer'] + 1])
        R['attention_changes'].append({'pair': row['pair'], 'trial_vector_index': row['vector_index'], 'changes': changes})
    clean = forward(records[P['primary_pair'] + '_receiver'])
    assert torch.equal(clean.logits, baselines[P['primary_pair'] + '_receiver'].logits)
    R['checks'].update({'all_trials_only_one_layer_and_one_position': True, 'all_trial_mlp_inputs_equal_clean': True,
        'all_trial_upstream_residuals_equal_clean': True, 'zero_or_inactive_edits_bitwise_equal_clean': True,
        'mlp_residual_addition_boundary_verified': True, 'restored_clean_logits_bitwise_equal': True,
        'attention_forward_methods_unchanged': True})
    R['status'] = 'complete'
    R['seconds'] = time.monotonic() - START
    np.savez_compressed(OUT / 'next-token-logits.npz', logits=np.stack(logits))
    np.savez_compressed(OUT / 'intervention-vectors.npz', intended=np.stack(deltas), actual_mlp_changes=np.stack(actual_changes))
    save('results.json', R)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != 'export-checksums.json'}
    save('export-checksums.json', manifest)
    print('COMPLETE', len(R['trials']), 'trials', json.dumps([{k:x[k] for k in ['pair','layer','feature','site','strength','P6','P8']} for x in best_trials]), flush=True)


if __name__ == '__main__':
    main()
