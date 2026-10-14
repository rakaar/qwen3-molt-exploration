"""Clean native-Qwen baseline; no hooks, MOLT weights, or activation intervention."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts'
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT/'protocol.json').read_text())
D = json.loads((ROOT/'cases.json').read_text())
START = time.monotonic()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(name, obj):
    temporary = OUT/(name+'.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(OUT/name)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--model-dir', type=Path, required=True)
    args.add_argument('--base-verification', type=Path, required=True)
    options = args.parse_args()
    verification = json.loads(options.base_verification.read_text())
    assert verification['repo'] == P['base_repo']
    assert verification['revision'] == P['base_revision']
    print('Verifying cached base files against prior pinned download manifest...', flush=True)
    for record in verification['files']:
        path = options.model_dir/record['file']
        assert path.stat().st_size == record['bytes'], str(path)
        assert sha(path) == record['sha256'], str(path)
    save('base-verification.json', verification)
    torch.manual_seed(P['seed'])
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(options.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(options.model_dir, local_files_only=True,
              dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert model.config.model_type == 'qwen3'
    passages = {p['id']:p['text'] for p in D['passages']}
    result = {'status':'running', 'started_at_utc':datetime.now(timezone.utc).isoformat(),
              'protocol':P, 'source_sha256':{n:sha(ROOT/n) for n in ['run.py','protocol.json','cases.json']},
              'runtime':{'torch':torch.__version__, 'transformers':transformers.__version__,
                         'python':platform.python_version(), 'gpu':torch.cuda.get_device_name(),
                         'cuda':torch.version.cuda}, 'rows':[]}
    assert torch.__version__ == '2.14.0+cu126'
    assert transformers.__version__ == '4.57.6'
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    def generate(case):
        prompt = P['prompt_template'].format(context=passages[case['passage_id']], question=case['question'])
        chat = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        batch = tokenizer(chat, add_special_tokens=False, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=False,
                        max_new_tokens=P['decoding']['max_new_tokens'], use_cache=True,
                        pad_token_id=tokenizer.eos_token_id)
        prompt_ids = batch['input_ids'][0].tolist()
        output_ids = generated[0,len(prompt_ids):].tolist()
        return {**case, 'prompt':prompt, 'chat':chat, 'prompt_token_ids':prompt_ids,
                'output_token_ids':output_ids,
                'answer':tokenizer.decode(output_ids, skip_special_tokens=True),
                'output_tokens':[tokenizer.decode([i]) for i in output_ids],
                'stop_reason':'eos' if output_ids and output_ids[-1] in eos else 'token_limit'}
    for case in D['cases']:
        assert time.monotonic()-START < P['max_runtime_minutes']*60
        row = generate(case)
        result['rows'].append(row)
        save('generations.json',result)
        print(case['id'], row['answer'], 'STOP='+row['stop_reason'], flush=True)
    repeat = generate(D['cases'][0])
    assert repeat['output_token_ids'] == result['rows'][0]['output_token_ids']
    result['clean_repeat']={'id':repeat['id'],'identical_output_token_ids':True}
    result['status']='complete'
    result['elapsed_seconds']=time.monotonic()-START
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    save('generations.json',result)
    print('Complete: 20 clean generations; first-case repeat token-identical.', flush=True)

if __name__ == '__main__':
    main()
