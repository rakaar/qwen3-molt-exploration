import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent
base = root / 'models/qwen3-4b'
torch.set_num_threads(8)
started = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, local_files_only=True,
    attn_implementation='sdpa',
).to('cuda').eval()
model.requires_grad_(False)
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
with torch.inference_mode():
    logits = model(**inputs).logits
    assert torch.isfinite(logits).all(), 'Nonfinite base-model logits'
    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
reply = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print('Base-model smoke response:', repr(reply), flush=True)
report = {
    'purpose': 'Infrastructure and checkpoint-loading smoke test only; no reconstruction or Jacobian validation',
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
    'base_dtype': str(model.dtype), 'base_logits_finite': True,
    'prompt': 'What is 2 + 2? Reply with just the number.', 'response': reply,
    'molt_downloaded': False,
    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
    'seconds': time.monotonic() - started,
}
(root / 'smoke-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'molt_tensors'}, indent=2), flush=True)
print('SMOKE_CHECK_OK', flush=True)
