"""Dallas one-word question: intact Qwen vs full layer-17 MLP replacement."""
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from evaluate_reconstruction import load_molt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/dallas-one-word'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260929
LAYER = 17
questions = [{'category': 'two-hop factual', 'question': 'What is the capital of the state containing Dallas?'}]
system = 'Answer with exactly one word. Do not explain.'
report = {
    'layer_zero_based': LAYER, 'seed': SEED,
    'selection': 'Dallas question chosen by user before generation',
    'system_prompt': system,
    'decoding': {'do_sample': False, 'max_new_tokens': 32, 'enable_thinking': False, 'use_cache': True},
    'intervention': 'Replace only layer 17 raw MLP output with MOLT(pre-post-attention-normalization residual); apply throughout prompt and generation; no reconstruction-error add-back',
    'results': questions,
}
(OUT / 'questions.json').write_text(json.dumps(report, indent=2) + '\n')
torch.set_num_threads(8)
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    ROOT / 'models/qwen3-4b', dtype=torch.bfloat16, local_files_only=True,
    attn_implementation='sdpa',
).to('cuda').eval().requires_grad_(False)
generation_config = GenerationConfig(
    do_sample=False, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=32, use_cache=True,
    bos_token_id=model.generation_config.bos_token_id,
    eos_token_id=model.generation_config.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model.generation_config = generation_config
layer = model.model.layers[LAYER]
original_mlp = layer.mlp
original_calls = [0]
def count_original(_module, args, output):
    original_calls[0] += 1
counter_hook = original_mlp.register_forward_hook(count_original)


@torch.inference_mode()
def generate(question):
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
    started = time.monotonic()
    output = model.generate(
        **inputs, generation_config=generation_config,
        do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
    )
    ids = output[0, inputs.input_ids.shape[1]:].tolist()
    eos = generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    return {'response': tokenizer.decode(ids, skip_special_tokens=True),
            'generated_token_ids': ids, 'prompt_token_ids': inputs.input_ids[0].tolist(),
            'generated_tokens': len(ids), 'ended_with_eos': bool(ids and ids[-1] in eos),
            'hit_token_limit': len(ids) == 32 and ids[-1] not in eos,
            'seconds': time.monotonic() - started}


for row in questions:
    row['intact'] = generate(row['question'])
    print('INTACT', row['question'], repr(row['intact']['response']), flush=True)
baseline_call_count = original_calls[0]
assert baseline_call_count > 0
molt, config = load_molt(LAYER)
pending = {}
def capture_pre_norm(_module, args):
    assert 'x' not in pending
    pending['x'] = args[0]


class ReplacedMLP(torch.nn.Module):
    def __init__(self, replacement):
        super().__init__()
        self.replacement = replacement
        self.calls = 0
        self.token_positions = 0

    def forward(self, normalized_input):
        x = pending.pop('x')
        assert x.shape == normalized_input.shape
        self.calls += 1
        flat = x.reshape(-1, x.shape[-1])
        self.token_positions += len(flat)
        outputs = []
        for start in range(0, len(flat), 32):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, raw = self.replacement(flat[start:start+32], LAYER)
            assert bool(torch.isfinite(raw).all())
            outputs.append(raw.to(normalized_input.dtype))
        return torch.cat(outputs).reshape_as(x)


pre_hook = layer.post_attention_layernorm.register_forward_pre_hook(capture_pre_norm)
replacement = ReplacedMLP(molt)
layer.mlp = replacement
try:
    for row in questions:
        calls_before, tokens_before = replacement.calls, replacement.token_positions
        row['molt_replaced'] = generate(row['question'])
        row['molt_replaced']['replacement_calls'] = replacement.calls - calls_before
        row['molt_replaced']['replaced_token_positions'] = replacement.token_positions - tokens_before
        assert row['intact']['prompt_token_ids'] == row['molt_replaced']['prompt_token_ids']
        assert row['molt_replaced']['replacement_calls'] == row['molt_replaced']['generated_tokens']
        assert original_calls[0] == baseline_call_count, 'Original MLP ran during replacement'
        assert not pending
        print('REPLACED', row['question'], repr(row['molt_replaced']['response']), flush=True)
        (OUT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
finally:
    layer.mlp = original_mlp
    pre_hook.remove()
    counter_hook.remove()
report['verification'] = {'original_mlp_calls_during_baselines': baseline_call_count,
                          'original_mlp_calls_during_replacement': original_calls[0] - baseline_call_count,
                          'replacement_calls': replacement.calls,
                          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                          'original_module_restored': layer.mlp is original_mlp}
report['molt_config'] = config
(OUT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print('COMPARISON_OK', flush=True)
