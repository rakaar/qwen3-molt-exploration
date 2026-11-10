"""Recheck both descriptions and both highlight scopes on three fixed extended examples."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for name in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'VAST_API_KEY', 'VAST_AI_KEY']:
    os.environ.pop(name, None)
os.environ.update(VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false', VLLM_USE_FLASHINFER_SAMPLER='0')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'audit.jsonl').exists(), 'Never overwrite prior responses'
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert sha(ROOT / 'plan.json') == protocol['plan_sha256']
    assert sha(ROOT / 'expected-active-fuzz-runtime.json') == protocol['expected_runtime_sha256']
    assert sha(ROOT / 'expected-judge-verification.json') == protocol['expected_weights_verification_sha256']
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert len(plan['tasks']) == protocol['requests'] == 12
    assert hashlib.sha256(plan['candidate_explanation'].encode()).hexdigest() == protocol['source_candidate_sha256']
    assert hashlib.sha256(plan['baseline_explanation'].encode()).hexdigest() == protocol['source_baseline_sha256']
    judge_root = Path(protocol['existing_judge_root'])
    expected = json.loads((ROOT / 'expected-active-fuzz-runtime.json').read_text())
    verification = json.loads((ROOT / 'expected-judge-verification.json').read_text())
    assert verification['repo'] == protocol['judge_repo'] and verification['revision'] == protocol['judge_revision']
    for row in verification['files']:
        path = judge_root / 'judge' / row['file']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    save(OUT / 'cached-judge-reverification.json', verification)
    tokenizer = AutoTokenizer.from_pretrained(judge_root / 'judge', local_files_only=True)
    tasks = []
    for source in plan['tasks']:
        task = dict(source)
        ids = tokenizer.apply_chat_template(task['messages'], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=False)
        if hasattr(ids, 'input_ids'):
            ids = ids.input_ids
        if ids and isinstance(ids[0], list):
            assert len(ids) == 1
            ids = ids[0]
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert len(ids) + 32 <= 4096
        request = dict(messages=task['messages'], model=protocol['judge_repo'], revision=protocol['judge_revision'],
                       temperature=0, max_tokens=32, enable_thinking=False)
        task.update(prompt_token_ids=ids,
                    request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        tasks.append(task)
    save(OUT / 'prepared-inputs.json', dict(tasks=tasks, protocol=protocol))
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, vllm=vllm.__version__,
                   gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
                   judge_repo=protocol['judge_repo'], judge_revision=protocol['judge_revision'],
                   native_NVFP4_arithmetic=False, backend='Marlin NVFP4 weights with BF16 activations on Ampere',
                   max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=.95, enforce_eager=True,
                   language_model_only=True, temperature=0, max_tokens=32, enable_thinking=False,
                   flashinfer_sampler=False, started_utc=datetime.now(timezone.utc).isoformat(),
                   source_sha256=sha(ROOT / 'run.py'), longest_prompt_tokens=max(len(t['prompt_token_ids']) for t in tasks))
    fields = ['torch', 'transformers', 'vllm', 'gpu', 'compute_capability', 'judge_repo', 'judge_revision',
              'native_NVFP4_arithmetic', 'backend', 'max_model_len', 'max_num_seqs', 'gpu_memory_utilization',
              'enforce_eager', 'language_model_only', 'temperature', 'max_tokens', 'enable_thinking', 'flashinfer_sampler']
    assert all(runtime[k] == expected[k] for k in fields), 'Judge runtime differs from the original comparison'
    save(OUT / 'runtime.json', runtime)
    save(OUT / 'judge-settings-comparison.json', dict(all_match=True, fields=fields))
    print(json.dumps(dict(stage='loading_judge', tasks=len(tasks))), flush=True)
    llm = LLM(model=str(judge_root / 'judge'), tokenizer=str(judge_root / 'judge'), dtype='bfloat16',
              trust_remote_code=False, max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=4096,
              gpu_memory_utilization=.95, enforce_eager=True, enable_prefix_caching=False,
              language_model_only=True, seed=42)
    params = SamplingParams(temperature=0, max_tokens=32)
    rows = []
    started = time.monotonic()
    for start in range(0, len(tasks), 10):
        assert time.monotonic() - started < protocol['maximum_runtime_minutes'] * 60
        batch = tasks[start:start + 10]
        outputs = llm.generate([dict(prompt_token_ids=t['prompt_token_ids']) for t in batch], params, use_tqdm=False)
        assert len(outputs) == len(batch)
        for task, result in zip(batch, outputs):
            answer = result.outputs[0]
            stripped = answer.text.strip()
            valid = stripped in ['[0]', '[1]'] and answer.finish_reason == 'stop'
            row = dict(**task, response=answer.text, response_token_ids=list(answer.token_ids),
                       finish_reason=answer.finish_reason, prediction=(stripped == '[1]') if valid else None,
                       at_utc=datetime.now(timezone.utc).isoformat())
            with (OUT / 'audit.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row)
        save(OUT / 'progress.json', dict(completed=len(rows), total=len(tasks)))
        print(json.dumps(dict(completed=len(rows), total=len(tasks))), flush=True)
    controls = [r for r in rows if r['cached_original_response'] is not None]
    save(OUT / 'summary.json', dict(status='complete' if all(r['prediction'] is not None for r in rows) else 'invalid_response',
                                   requests=len(rows), repeated_baselines_match=all(r['response'].strip() == r['cached_original_response'] for r in controls),
                                   repeated_controls=len(controls), elapsed_seconds=time.monotonic() - started,
                                   posthoc=True, description_refinement_performed=False,
                                   interpretation='Same three recorded extended windows; only description and highlight scope vary. No subject-model generation or gate intervention.'))
    files = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()]
    save(OUT / 'transfer-manifest.json', dict(files=files))
    print((OUT / 'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
