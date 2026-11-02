"""One-token L29-T2375 gate-off reproduction, using the audited native-model engine."""
import hashlib
import json
import os
from pathlib import Path
import runpy

for key in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'VAST_API_KEY', 'VAST_AI_KEY']:
    os.environ.pop(key, None)

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
assert not (OUT / 'results.json').exists(), 'Preserve previous results'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


assert torch.__version__ == '2.14.0+cu126' and transformers.__version__ == '4.57.6'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
expected = json.loads((ROOT / 'expected-base-verification.json').read_text())
for row in expected['files']:
    p = ROOT / 'model' / row['file']
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(ROOT / 'model', local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
assert model.config.hidden_size == 2560 and model.config.num_hidden_layers == 36
engine = runpy.run_path(str(ROOT / 'engine.py'))['Selective'](model, tok)
save(OUT / 'runtime.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
    'gpu': torch.cuda.get_device_name(), 'base_revision': expected['revision'], 'verified_base_files': expected['files'],
    'dtype': 'BF16', 'edit_dtype': 'FP32 then one BF16 cast', 'attention': 'eager', 'tf32': False,
    'source_sha256': {p.name: sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py', '.json']}})
case = engine.cases[0]
prompt = 'the eleventh month of the year is'
native = engine.run(case, prompt, 'native')
reference = json.loads((ROOT / 'native-reference.json').read_text())
assert native['output_token_ids'] == reference['output_token_ids']
anchor_error = max(abs(a['selected_probability'] - b['selected_probability'])
                   for a, b in zip(native['steps'], reference['steps']))
assert anchor_error < 1e-4
position = len(native['prompt_token_ids']) - 1
assert position == 7 and tok.decode([native['prompt_token_ids'][position]]) == ' is'
spec = {'kind': 'positions', 'positions': [position]}
zero = engine.run(case, prompt, 'zero-at-is', spec=spec, target=0., alpha=0., audit=True)
assert native['steps'] == zero['steps'] and native['output_token_ids'] == zero['output_token_ids']
assert all(t['actual_delta_norm'] == 0 for t in zero['trace'])
result = {'status': 'baseline_validated', 'prompt': prompt, 'edited_position': position,
          'edited_token': ' is', 'historical_anchor_max_probability_error': anchor_error,
          'native': native, 'zero_strength': zero}
save(OUT / 'results.json', result)
off = engine.run(case, prompt, 'off-at-is', spec=spec, target=0., alpha=1., audit=True)
assert [t['position'] for t in off['trace'] if t['selected']] == [position]
assert [t['position'] for t in off['trace'] if t['actual_delta_norm'] > 0] == [position]
assert all(t['actual_delta_norm'] == 0 for t in off['trace'] if t['phase'] == 'output')
result.update(status='complete', gate_off=off, hooks_remaining=engine.hooks())
assert result['hooks_remaining'] == 0
save(OUT / 'results.json', result)
print(json.dumps({'status': 'complete', 'native_response': native['answer'], 'gate_off_response': off['answer'],
    'intervention': next(t for t in off['trace'] if t['selected']),
    'historical_anchor_max_probability_error': anchor_error}, ensure_ascii=False), flush=True)
