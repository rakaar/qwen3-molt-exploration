"""Pinned experiment driver on the explicitly authorized CUDA worker only.

The CPU controller prepares this source but never executes/imports it. Runtime
setup and generation are separate; the original 120-answer file is unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from export_artifacts import export

EXPERIMENT = Path('/workspace/em-screen-task/experiment')
OUTPUT = Path('/workspace/em-screen-results')
EXPORT = Path('/var/log/em-screen-artifacts')
COMMIT = '21da619829635b97b1797600e6b3e5bd3ef5e468'
EXPECTED_INSTANCE = 56373998


def require_worker():
    if int(os.environ.get('CONTAINER_ID', '-1')) != EXPECTED_INSTANCE:
        raise RuntimeError('This driver is restricted to the authorized GPU instance')
    if 'SPAR_VAST_API_KEY' in os.environ:
        raise RuntimeError('Controller management credential must never reach the worker')
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=True)


def write(name, value):
    path = OUTPUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def snapshot():
    proof = Path('/workspace/em-screen-results/native-deadline-status.json')
    # The optional native proof is already a sanitized task file. No system
    # logs, environment files, or platform credentials are copied.
    return export(OUTPUT, EXPERIMENT, EXPORT)


def smoke():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required; CPU fallback is prohibited')
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split('.')[:2])) < (12, 8):
        raise RuntimeError('Pinned CUDA 12.8 runtime required')
    expected = {'torch': '2.8.0', 'transformers': '4.57.1', 'accelerate': '1.10.1',
                'huggingface-hub': '0.35.3', 'safetensors': '0.6.2'}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if any(versions[name].split('+')[0] != version for name, version in expected.items()):
        raise RuntimeError('Pinned worker package versions do not match')
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 24_000_000_000 or properties.major < 8:
        raise RuntimeError('One Ampere-or-newer GPU with at least 24 GB is required')
    free_disk = shutil.disk_usage(OUTPUT).free
    if free_disk < 26_000_000_000:
        raise RuntimeError('Insufficient free disk for three pinned checkpoints and margin')
    value = (torch.tensor([2.0], device='cuda') * 3).item()
    torch.cuda.synchronize()
    if value != 6.0:
        raise RuntimeError('Actual CUDA tensor operation failed')
    result = {'instance_id': EXPECTED_INSTANCE, 'gpu': properties.name,
              'total_vram_bytes': properties.total_memory, 'free_vram_bytes': torch.cuda.mem_get_info()[0],
              'free_disk_bytes': free_disk, 'torch_cuda': torch.version.cuda,
              'packages': versions, 'cuda_tensor_value': value,
              'cuda_operation_verified': True, 'model_inference_performed': False,
              'source_commit': COMMIT, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    write('smoke.json', result)
    write('task-status.json', {'phase': 'hardware-smoke-complete', 'instance_id': EXPECTED_INSTANCE})
    snapshot()


def model_smoke():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU model inference')
    info = next(m for m in json.loads((EXPERIMENT / 'models.json').read_text()) if m['id'] == 'baseline')
    tokenizer = AutoTokenizer.from_pretrained(info['repo_id'], revision=info['revision'], token=False,
                                               trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(info['repo_id'], revision=info['revision'], token=False,
                torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa',
                trust_remote_code=False).eval().requires_grad_(False)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Reply with OK.'}],
                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda:0')
    with torch.inference_mode():
        tokens = model.generate(**encoded, do_sample=False, max_new_tokens=4,
                                 pad_token_id=tokenizer.eos_token_id)
    result = json.loads((OUTPUT / 'smoke.json').read_text())
    result['model_smoke'] = {'repo': info['repo_id'], 'revision': info['revision'],
                            'device': str(model.device), 'generated_tokens': tokens.shape[-1] - encoded.input_ids.shape[-1],
                            'outside_pilot': True, 'sample_saved_as_pilot_response': False}
    write('smoke.json', result)
    snapshot()


def run(samples, models, seconds):
    command = [os.sys.executable, str(EXPERIMENT / 'run.py'), '--output', str(OUTPUT),
               '--samples', str(samples), '--max-seconds', str(seconds)]
    if models:
        command += ['--models', *models]
    env = os.environ.copy()
    for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN'):
        env.pop(name, None)
    log = OUTPUT / 'worker.log'
    with log.open('ab') as stream:
        child = subprocess.Popen(command, cwd=EXPERIMENT, env=env, stdout=stream,
                                 stderr=stream, stdin=subprocess.DEVNULL, start_new_session=True)
        while child.poll() is None:
            write('task-status.json', {'phase': 'generation', 'samples': samples,
                                      'pid': child.pid, 'instance_id': EXPECTED_INSTANCE,
                                      'updated_at_utc': datetime.now(timezone.utc).isoformat()})
            try:
                snapshot()
            except (ValueError, OSError):
                pass  # A writer snapshot race fails closed; retry on the next poll.
            time.sleep(10)
    (OUTPUT / 'worker-exit-code.txt').write_text(str(child.returncode) + '\n')
    subprocess.run([os.sys.executable, str(EXPERIMENT / 'summarize.py'), '--output', str(OUTPUT)],
                   cwd=EXPERIMENT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    write('task-status.json', {'phase': 'generation-exited', 'samples': samples,
                              'exit_code': child.returncode, 'instance_id': EXPECTED_INSTANCE})
    snapshot()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'model-smoke', 'run', 'export'))
    parser.add_argument('--samples', type=int, choices=(5, 10), default=5)
    parser.add_argument('--models', nargs='*', choices=('baseline', 'misaligned-v1', 'misaligned-v2'))
    parser.add_argument('--max-seconds', type=int, default=10800)
    args = parser.parse_args()
    require_worker()
    if args.mode == 'smoke':smoke()
    elif args.mode == 'model-smoke':model_smoke()
    elif args.mode == 'export':snapshot()
    else:run(args.samples, args.models, min(10800, args.max_seconds))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never render arbitrary exception text, environment, subprocess args,
        # startup logs, auth headers, or tokens into the task export.
        OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=True)
        write('task-status.json', {'phase': 'failed', 'error_type': type(error).__name__,
                                  'instance_id': EXPECTED_INSTANCE})
        try:snapshot()
        except Exception:pass
        raise SystemExit(1)
