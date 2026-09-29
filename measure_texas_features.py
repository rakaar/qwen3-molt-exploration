"""Read-only encoding of selected Texas features on the intact Qwen3-4B prompt."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/texas-features'
manifest = json.loads((OUT / 'encoder-manifest.json').read_text())
old = json.loads((ROOT / 'results/dallas-one-word/comparison.json').read_text())
torch.set_num_threads(8)
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
text = tokenizer.apply_chat_template(
    [{'role': 'system', 'content': old['system_prompt']},
     {'role': 'user', 'content': old['results'][0]['question']}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
ids = inputs.input_ids[0].tolist()
assert ids == old['results'][0]['intact']['prompt_token_ids']
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'models/qwen3-4b', dtype=torch.bfloat16, local_files_only=True,
    attn_implementation='sdpa',
).to('cuda').eval().requires_grad_(False)
captured, hooks = {}, []


def capture(layer):
    def hook(_module, args):
        assert layer not in captured
        captured[layer] = args[0].detach().cpu()
    return hook


for spec in manifest['layers']:
    hooks.append(model.model.layers[spec['layer']].mlp.register_forward_pre_hook(capture(spec['layer'])))
with torch.inference_mode():
    logits = model(**inputs, use_cache=False).logits
    predicted = int(logits[0, -1].argmax())
for hook in hooks:
    hook.remove()
assert predicted == old['results'][0]['intact']['generated_token_ids'][0]
print('PREDICTED', tokenizer.decode([predicted]), flush=True)
results = []
for spec in manifest['layers']:
    blob = (OUT / spec['file']).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == spec['sha256']
    n = len(spec['features'])
    values = torch.frombuffer(bytearray(blob), dtype=torch.bfloat16)
    W = values[:n*2560].reshape(n, 2560)
    b = values[n*2560:]
    assert b.shape == (n,)
    x = captured[spec['layer']][0]
    # Matches circuit-tracer SingleLayerTranscoder.encode; compute in FP32
    # on the actual BF16 model inputs and checkpoint values to reduce rounding.
    pre = F.linear(x.float(), W.float(), b.float())
    acts = F.relu(pre)
    acts_bf16 = F.relu(F.linear(x.cuda(), W.cuda(), b.cuda())).float().cpu()
    for i, feature in enumerate(spec['features']):
        a = acts[:, i]
        results.append({**feature, 'activations': a.tolist(),
                        'preactivations': pre[:, i].tolist(),
                        'activations_bf16': acts_bf16[:, i].tolist(),
                        'active_positions': torch.where(a > 0)[0].tolist(),
                        'max_activation': float(a.max()),
                        'peak_position': int(a.argmax()),
                        'max_bf16_difference': float((a-acts_bf16[:, i]).abs().max())})
tokens = [{'position': p, 'id': token_id, 'text': tokenizer.decode([token_id]),
           'token': tokenizer.convert_ids_to_tokens(token_id)} for p, token_id in enumerate(ids)]
report = {'model': 'Qwen/Qwen3-4B', 'model_dtype': 'bfloat16',
          'model_variant': 'Original intact model; no MOLT or transcoder replacement',
          'hook': 'Each Hugging Face decoder layer MLP input, after post_attention_layernorm',
          'encoding': 'ReLU(W_enc x + b_enc), selected independent rows, FP32 arithmetic',
          'prompt': text, 'system_prompt': old['system_prompt'],
          'question': old['results'][0]['question'], 'tokens': tokens,
          'first_answer_token_id': predicted, 'first_answer_token': tokenizer.decode([predicted]),
          'scope': 'Prompt positions only, including chat formatting, before generating Austin',
          'selection': manifest['selection'], 'transcoder_revision': manifest['revision'],
          'features_tested': len(results),
          'features_active': sum(bool(r['active_positions']) for r in results),
          'features': results}
(OUT / 'activations.json').write_text(json.dumps(report, indent=2))
torch.save(captured, OUT / 'captured-mlp-inputs.pt')
print('DONE', report['features_tested'], 'tested;', report['features_active'], 'active', flush=True)
for r in sorted(results, key=lambda r: -r['max_activation'])[:15]:
    print(r['layer'], r['index'], round(r['max_activation'], 4),
          [(p, tokens[p]['text'], round(r['activations'][p], 4)) for p in r['active_positions']], flush=True)
