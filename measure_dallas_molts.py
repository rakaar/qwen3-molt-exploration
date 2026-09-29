"""Observe MOLT gates at original-model activations without replacing any MLP."""
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_reconstruction import load_molt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/dallas-molt-gates'
OUT.mkdir(parents=True, exist_ok=True)
previous = json.loads((ROOT / 'results/dallas-one-word/comparison.json').read_text())
ids = previous['results'][0]['intact']['prompt_token_ids']
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
position = 26
assert tokenizer.decode([ids[position]]) == ' Dallas'
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'models/qwen3-4b', dtype=torch.bfloat16, local_files_only=True,
    attn_implementation='sdpa',
).to('cuda').eval().requires_grad_(False)
captured, hooks = {}, []


def capture(layer):
    def hook(_module, args):
        captured[layer] = args[0][0].detach().cpu()
    return hook


for layer in [17, 25]:
    hooks.append(model.model.layers[layer].post_attention_layernorm.register_forward_pre_hook(capture(layer)))
with torch.inference_mode():
    logits = model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False).logits
    predicted = int(logits[0, -1].argmax())
assert predicted == previous['results'][0]['intact']['generated_token_ids'][0]
for hook in hooks:
    hook.remove()
del model, logits
gc.collect()
torch.cuda.empty_cache()
torch.save(captured, OUT / 'original-prenorm-inputs.pt')
report = {
    'question': previous['results'][0]['question'], 'system_prompt': previous['system_prompt'],
    'tokens': [{'position': p, 'id': i, 'text': tokenizer.decode([i])} for p, i in enumerate(ids)],
    'dallas_position': position, 'predicted_first_token': tokenizer.decode([predicted]),
    'method': 'Original Qwen3-4B BF16 activations before post_attention_layernorm; saved MOLT input standardization; FP32 MOLT gate and transform arithmetic',
    'scope': 'MOLTs evaluated on intact-model inputs, no replacement or intervention; layers 17 and 25 only',
    'contribution_measure': 'L2 norm of each gated transform output, multiplied dimensionwise by saved output std; output mean excluded',
    'layers': [],
}
with torch.inference_mode():
    for layer in [17, 25]:
        molt, config = load_molt(layer)
        x = captured[layer].cuda().float()
        z = molt.input_standardizer(x, layer)
        pre = molt.e(z)
        gates = molt.nonlinearity(pre)
        active = torch.where(gates[position] > 0)[0].tolist()
        ref_gates, ref_norm, ref_raw = molt(x[position:position+1], layer)
        torch.testing.assert_close(ref_gates[0], gates[position], rtol=1e-4, atol=1e-4)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z_bf = molt.input_standardizer(captured[layer].cuda(), layer)
            bf_gates = molt.nonlinearity(molt.e(z_bf))
        records, contributions, offset = [], [], 0
        for rank, U, V in zip(molt.ranks, molt.Us, molt.Vs):
            group = [i for i in active if offset <= i < offset+len(U)]
            for index in group:
                j = index-offset
                unweighted = (z[position] @ V[j]) @ U[j]
                weighted = gates[position, index] * unweighted
                contributions.append(weighted)
                raw_contribution = weighted * molt.output_standardizer.std[layer].float()
                all_acts = gates[:, index].float().cpu()
                records.append({
                    'transform_id': index, 'rank': rank,
                    'gate': float(gates[position, index]),
                    'gate_bf16': float(bf_gates[position, index]),
                    'preactivation': float(pre[position, index]),
                    'threshold': float(molt.nonlinearity.theta[0, index]),
                    'raw_output_l2': float(raw_contribution.norm()),
                    'standardized_output_l2': float(weighted.norm()),
                    'gate_by_position': all_acts.tolist(),
                    'peak_gate_position': int(all_acts.argmax()),
                    'peak_gate': float(all_acts.max()),
                })
            offset += len(U)
        total = torch.stack(contributions).sum(dim=0) if contributions else torch.zeros_like(z[position])
        torch.testing.assert_close(total, ref_norm[0], rtol=1e-4, atol=1e-4)
        reconstructed_raw = total * molt.output_standardizer.std[layer].float() + molt.output_standardizer.mean[layer].float()
        torch.testing.assert_close(reconstructed_raw, ref_raw[0], rtol=1e-4, atol=1e-4)
        summary = {
            'layer': layer, 'total_transforms': config['n_features'],
            'dallas_active_count': len(active),
            'dallas_active_ids_bf16': torch.where(bf_gates[position] > 0)[0].tolist(),
            'count_active_by_position': (gates > 0).sum(dim=1).cpu().tolist(),
            'decomposition_max_abs_error': float((total-ref_norm[0]).abs().max()),
            'active_transforms': sorted(records, key=lambda r: -r['gate']),
        }
        report['layers'].append(summary)
        (OUT / 'gates.json').write_text(json.dumps(report, indent=2))
        print('LAYER', layer, 'ACTIVE', len(active), flush=True)
        for item in summary['active_transforms']:
            print('TRANSFORM', item['transform_id'], 'GATE', item['gate'], 'RAW_L2', item['raw_output_l2'],
                  'PEAK_TOKEN', report['tokens'][item['peak_gate_position']]['text'], flush=True)
        del molt, x, z, pre, gates, bf_gates, z_bf, ref_gates, ref_norm, ref_raw, U, V
        gc.collect()
        torch.cuda.empty_cache()
print('MOLT_GATES_OK', flush=True)
