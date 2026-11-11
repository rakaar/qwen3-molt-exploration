"""Remove only the L29-T2375 contribution at every active processed token."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import time

for key in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'VAST_API_KEY', 'VAST_AI_KEY']:
    os.environ.pop(key, None)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--molt-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    verification = json.loads((HERE / 'expected-base-verification.json').read_text())
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'results.json').exists(), 'Never overwrite prior captures'
    for row in verification['files']:
        f = args.model_dir / row['file']
        assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
    assert verification['revision'] == protocol['base_revision']
    assert sha(args.molt_checkpoint) == protocol['molt_checkpoint_sha256']
    assert sha(HERE / 'weights/L29-T2375.npz') == protocol['feature_sha256']
    assert torch.__version__ == '2.14.0+cu126' and transformers.__version__ == '4.57.6'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
    assert len(model.model.layers) == 36 and model.config.hidden_size == 2560
    with np.load(HERE / 'weights/L29-T2375.npz', allow_pickle=False) as f:
        w = {k: torch.from_numpy(f[k].copy()).float().cuda() for k in f.files}
    assert w['V'].shape == (2560, 32) and w['U'].shape == (32, 2560)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    assert all(isinstance(i, int) for i in eos)
    special = set(tok.all_special_ids)
    started = time.monotonic()

    def hooks():
        return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())

    def decode(ident):
        return tok.decode([ident], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def encode(case):
        text = case['prompt'] if case['prompt_format'] == 'raw' else tok.apply_chat_template([{'role':'user', 'content':case['prompt']}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert text == case['formatted_prompt'] and enc['input_ids'] == case['input_token_ids']
        start = text.index(case['prompt'])
        end = start + len(case['prompt'])
        meta = []
        for position, (ident, (a, b)) in enumerate(zip(enc['input_ids'], enc['offset_mapping'])):
            region = 'instruction'
            if case['prompt_format'] == 'raw':
                region = 'calibration_prompt'
            elif b <= start:
                region = 'template_before'
            elif a >= end:
                region = 'answer_prefix'
            elif b > start + case['list_span'][0] and a < start + case['list_span'][1]:
                region = 'list'
            elif b > start + case['question_span'][0] and a < start + case['question_span'][1]:
                region = 'question'
            meta.append(dict(position=position, token_id=ident, token=decode(ident), phase='input', region=region, special=ident in special, final_input=position == len(enc['input_ids']) - 1))
        return text, enc['input_ids'], meta

    @torch.inference_mode()
    def run(case, condition, candidates, scope='all'):
        assert hooks() == 0
        text, prompt_ids, meta = encode(case)
        alpha = 1.0 if condition == 'off' else 0.0
        state = dict(current=meta, context='generation', terminal=False)
        trace, arrays, handles = [], {}, []

        def observe(_module, inputs):
            x = inputs[0].detach().float()
            assert x.shape == (1, len(state['current']), 2560)
            z = (x - w['input_mean']) / w['input_std']
            raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
            independent = (z * w['e']).sum(-1) + w['bias'][0]
            assert torch.allclose(raw, independent, atol=3e-5, rtol=3e-5)
            gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, 0.0)
            operator = (z @ w['V'] @ w['U']) * w['output_std']
            state.update(x=x, raw=raw, gate=gate, operator=operator)

        def edit(_module, _inputs, output):
            selected = torch.tensor([[scope == 'all' or m['final_input'] for m in state['current']]], device='cuda')
            mask = (state['gate'] > 0) & selected
            delta = -alpha * state['gate'].unsqueeze(-1) * state['operator'] * selected.unsqueeze(-1)
            changed = (output.float() + delta).to(output.dtype)
            assert torch.isfinite(changed).all()
            assert torch.equal(changed[~mask], output[~mask])
            if alpha == 0:
                assert torch.equal(changed, output)
            actual = changed.float() - output.float()
            values = torch.stack([state['raw'], state['gate'], state['operator'].norm(dim=-1), delta.norm(dim=-1), actual.norm(dim=-1), output.float().norm(dim=-1)], dim=-1)[0].cpu().tolist()
            for i, (m, v) in enumerate(zip(state['current'], values)):
                eligible = bool(mask[0, i])
                trace.append(dict(**m, context=state['context'], trace_only_terminal=state['terminal'], raw=v[0], natural_gate=v[1], effective_gate=0.0 if alpha == 1 and eligible else v[1], operator_norm=v[2], intended_delta_norm=v[3], actual_delta_norm=v[4], native_mlp_norm=v[5], eligible=eligible, removed=bool(alpha == 1 and eligible)))
            if condition == 'off' and state['context'] == 'generation' and state['current'][0]['phase'] == 'input':
                positions = torch.nonzero(mask[0], as_tuple=False).flatten()
                arrays.update(input_active_positions=positions.cpu().numpy(), residual_input=state['x'][0, positions].cpu().numpy(), gate_before=state['gate'][0, positions].cpu().numpy(), operator=state['operator'][0, positions].cpu().numpy(), native_mlp=output[0, positions].float().cpu().numpy(), edited_mlp=changed[0, positions].float().cpu().numpy(), intended_delta=delta[0, positions].cpu().numpy(), actual_delta=actual[0, positions].cpu().numpy())
            return changed if alpha == 1 and bool(mask.any()) else None

        if condition != 'native':
            handles.append(model.model.layers[29].post_attention_layernorm.register_forward_pre_hook(observe))
            handles.append(model.model.layers[29].mlp.register_forward_hook(edit))

        def output_meta(index, ident, candidate=False):
            return [dict(position=len(prompt_ids) + index, token_id=ident, token=decode(ident), phase='candidate' if candidate else 'response', region='candidate' if candidate else 'response', special=ident in special, final_input=False)]

        def forward(ids, mask, cache):
            assert time.monotonic() - started < protocol['maximum_runtime_minutes'] * 60
            value = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
            logits = value.logits[0, -1].float()
            assert torch.isfinite(logits).all()
            return logits, value.past_key_values

        candidate_scores = {}
        try:
            ids = torch.tensor([prompt_ids], device='cuda')
            mask, cache, generated, steps = torch.ones_like(ids), None, [], []
            for step in range(protocol['max_new_tokens']):
                logits, cache = forward(ids, mask, cache)
                probs = logits.softmax(-1)
                chosen = int(logits.argmax())
                entry = dict(token_id=chosen, token=decode(chosen), probability=float(probs[chosen]), logits_sha256=hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest())
                if step == 0:
                    vals, inds = probs.topk(5)
                    entry['top5'] = [dict(token_id=int(i), token=decode(int(i)), probability=float(q)) for q, i in zip(vals, inds)]
                    first_logits = logits.clone()
                    cache0 = copy.deepcopy(cache)
                    arrays['first_logits'] = logits.cpu().numpy()
                steps.append(entry)
                generated.append(chosen)
                if chosen in eos:
                    break
                ids = torch.tensor([[chosen]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
                state['current'] = output_meta(len(generated) - 1, chosen)
            # Same read-only terminal pass as the frozen baseline, without another decoded token.
            state.update(current=output_meta(len(generated) - 1, generated[-1]), terminal=True)
            tail, _ = forward(torch.tensor([[generated[-1]]], device='cuda'), torch.ones((1, len(prompt_ids) + len(generated)), device='cuda', dtype=torch.long), cache)
            native = dict(output_token_ids=generated, response=tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False), stop_reason='eos' if generated[-1] in eos else 'token_limit', steps=steps, unused_terminal_logits_sha256=hashlib.sha256(tail.cpu().numpy().tobytes()).hexdigest())
            for name, word in candidates.items():
                tokens = tok.encode(word, add_special_tokens=False)
                assert tokens and tok.decode(tokens, clean_up_tokenization_spaces=False) == word
                assert tok.encode(text + word, add_special_tokens=False) == prompt_ids + tokens
                state.update(context='score-' + name, terminal=False)
                score_cache = copy.deepcopy(cache0)
                conditional = []
                score_logits = first_logits
                score_mask = torch.ones((1, len(prompt_ids)), device='cuda', dtype=torch.long)
                for index, ident in enumerate(tokens):
                    prob = score_logits.softmax(-1)
                    conditional.append(dict(token_id=ident, token=decode(ident), probability=float(prob[ident]), logit=float(score_logits[ident]), logsumexp=float(score_logits.logsumexp(-1))))
                    state['current'] = output_meta(index, ident, candidate=True)
                    score_ids = torch.tensor([[ident]], device='cuda')
                    score_mask = torch.cat([score_mask, torch.ones_like(score_ids)], dim=1)
                    score_logits, score_cache = forward(score_ids, score_mask, score_cache)
                end_prob = float(score_logits.softmax(-1)[sorted(eos)].sum())
                probability = math.prod(t['probability'] for t in conditional)
                candidate_scores[name] = dict(text=word, token_ids=tokens, conditional=conditional, word_probability=probability, eos_probability_given_word=end_prob, exact_word_then_eos_probability=probability * end_prob)
        finally:
            for handle in handles:
                handle.remove()
        assert hooks() == 0
        if condition != 'native':
            generation_trace = [t for t in trace if t['context'] == 'generation']
            assert [t['token_id'] for t in generation_trace] == prompt_ids + generated
            assert sum(t['trace_only_terminal'] for t in generation_trace) == 1
            if condition == 'off':
                assert all(t['effective_gate'] == 0 for t in trace if scope == 'all' or t['final_input'])
                assert all(t['intended_delta_norm'] == 0 for t in trace if not t['removed'])
        path = OUT / f"{case['id']}--{condition}.npz"
        np.savez_compressed(path, **arrays)
        return dict(condition=condition, scope=scope, alpha=alpha, native=native, candidates=candidate_scores, trace=trace, arrays=dict(file=path.name, sha256=sha(path)))

    result = dict(status='running', protocol=protocol, protocol_sha256=sha(HERE / 'protocol.json'), source_sha256=sha(HERE / 'run.py'), runtime=dict(torch=torch.__version__, transformers=transformers.__version__, native_dtype='BF16', observer_dtype='FP32', attention='eager', TF32=False, threshold=float(w['theta'][0]), eos_ids=sorted(eos), model_checkpoint_verified=True, full_molt_checkpoint_verified=True), calibration=None, cases=[])
    save(OUT / 'results.json', result)
    calibration = protocol['calibration']
    calibration_native = run(calibration, 'native', dict(eleventh=' november', twelfth=' december'), scope='final_input')
    assert calibration_native['native'] == calibration['native']
    calibration_zero = run(calibration, 'zero', dict(eleventh=' november', twelfth=' december'), scope='final_input')
    assert calibration_zero['native'] == calibration_native['native'] and calibration_zero['candidates'] == calibration_native['candidates']
    calibration_off = run(calibration, 'off', dict(eleventh=' november', twelfth=' december'), scope='final_input')
    result['calibration'] = dict(native=calibration_native, zero=calibration_zero, off=calibration_off)
    save(OUT / 'results.json', result)
    for case in protocol['cases']:
        candidates = dict(eleventh=case['eleventh_word'], twelfth=case['twelfth_word'])
        native = run(case, 'native', candidates)
        assert native['native'] == case['native'], 'Native baseline changed; do not interpret the ablation'
        zero = run(case, 'zero', candidates)
        assert zero['native'] == native['native'] and zero['candidates'] == native['candidates']
        active = [t['position'] for t in zero['trace'] if t['context'] == 'generation' and t['phase'] == 'input' and t['natural_gate'] > 0]
        assert active == case['native_input_active_positions']
        off = run(case, 'off', candidates)
        row = dict(id=case['id'], list_id=case['list_id'], format=case['format'], eleventh_word=case['eleventh_word'], twelfth_word=case['twelfth_word'], native_twelfth_question_control=case['native_twelfth_question_control'], native=native, zero=zero, off=off, native_reference_exact=True, zero_strength_exact=True)
        result['cases'].append(row)
        save(OUT / 'results.json', result)
        save(OUT / 'progress.json', dict(completed=len(result['cases']), total=7))
        print(json.dumps(dict(case=case['id'], native=native['native']['response'], off=off['native']['response'], p11=off['candidates']['eleventh']['word_probability'], p12=off['candidates']['twelfth']['word_probability'])), flush=True)
    result.update(status='complete', hooks_remaining=hooks(), runtime_seconds=time.monotonic() - started, finished_at_utc=datetime.now(timezone.utc).isoformat())
    save(OUT / 'results.json', result)
    files = [dict(path=f.name, bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(OUT.iterdir()) if f.is_file()]
    save(OUT / 'transfer-manifest.json', dict(files=files))
    print(json.dumps(dict(status='complete', cases=len(result['cases']), runtime_seconds=result['runtime_seconds'])), flush=True)

if __name__ == '__main__':
    main()
