"""Observe water/ice activation timing, freeze native controls, then remove natural contributions at input scopes."""
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
        if 'formatted_prompt' in case:
            assert text == case['formatted_prompt'] and enc['input_ids'] == case['input_token_ids']
        else:
            case.update(formatted_prompt=text, input_token_ids=enc['input_ids'])
        start = text.index(case['prompt'])
        end = start + len(case['prompt'])
        query_final_positions = [i for i,(a,b) in enumerate(enc['offset_mapping']) if b == end and a < end]
        assert len(query_final_positions) == 1
        case['query_final_position'] = query_final_positions[0]
        meta = []
        for position, (ident, (a, b)) in enumerate(zip(enc['input_ids'], enc['offset_mapping'])):
            region = 'instruction'
            if b <= start:
                region = 'template_before'
            elif a >= end:
                region = 'answer_prefix'
            elif b > start + case['codebook_span'][0] and a < start + case['codebook_span'][1]:
                region = 'codebook'
            elif b > start + case['query_span'][0] and a < start + case['query_span'][1]:
                region = 'query'
            meta.append(dict(position=position, token_id=ident, token=decode(ident), phase='input', region=region, special=ident in special, final_input=position == len(enc['input_ids']) - 1))
        return text, enc['input_ids'], meta

    @torch.inference_mode()
    def run(case, condition, candidates):
        assert hooks() == 0
        text, prompt_ids, meta = encode(case)
        alpha = 1.0 if condition.startswith('off-') else 0.0
        target_value = 0.0
        if condition == 'off-all':
            selected_positions = set(range(len(prompt_ids)))
        elif condition == 'off-last':
            selected_positions = {len(prompt_ids)-1}
        else:
            selected_positions = {case['query_final_position']}
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
            selected = torch.tensor([[m['phase'] == 'input' and m['position'] in selected_positions for m in state['current']]], device='cuda')
            delta = alpha * (target_value - state['gate']).unsqueeze(-1) * state['operator'] * selected.unsqueeze(-1)
            changed = (output.float() + delta).to(output.dtype)
            assert torch.isfinite(changed).all()
            assert torch.equal(changed[~selected], output[~selected])
            if alpha == 0:
                assert torch.equal(changed, output)
            actual = changed.float() - output.float()
            values = torch.stack([state['raw'], state['gate'], state['operator'].norm(dim=-1), delta.norm(dim=-1), actual.norm(dim=-1), output.float().norm(dim=-1)], dim=-1)[0].cpu().tolist()
            for i, (m, v) in enumerate(zip(state['current'], values)):
                eligible = bool(selected[0, i])
                trace.append(dict(**m, context=state['context'], trace_only_terminal=state['terminal'], raw=v[0], natural_gate=v[1], effective_gate=target_value if alpha == 1 and eligible else v[1], target_gate=target_value if eligible else None, operator_norm=v[2], intended_delta_norm=v[3], actual_delta_norm=v[4], native_mlp_norm=v[5], selected=eligible, edited=bool(alpha == 1 and eligible and v[1] > 0)))
            if alpha == 1 and state['context'] == 'generation' and state['current'][0]['phase'] == 'input':
                positions = torch.nonzero(selected[0], as_tuple=False).flatten()
                assert positions.cpu().tolist() == sorted(selected_positions)
                audit_positions = sorted({case['query_final_position'],len(prompt_ids)-1} | set(torch.nonzero((selected & (state['gate']>0))[0],as_tuple=False).flatten().cpu().tolist()))
                audit = torch.tensor(audit_positions, device='cuda')
                arrays.update(selected_positions=positions.cpu().numpy(), all_gate_before=state['gate'][0].cpu().numpy(), all_gate_after=torch.where(selected,state['gate']*0,state['gate'])[0].cpu().numpy(), all_intended_delta_norm=delta[0].norm(dim=-1).cpu().numpy(), all_actual_delta_norm=actual[0].norm(dim=-1).cpu().numpy(), all_quantization_error_norm=(actual[0]-delta[0]).norm(dim=-1).cpu().numpy(), audit_positions=audit.cpu().numpy(), residual_input=state['x'][0, audit].cpu().numpy(), gate_before=state['gate'][0, audit].cpu().numpy(), gate_after=torch.where(selected,state['gate']*0,state['gate'])[0,audit].cpu().numpy(), operator=state['operator'][0, audit].cpu().numpy(), native_mlp=output[0, audit].float().cpu().numpy(), edited_mlp=changed[0, audit].float().cpu().numpy(), intended_delta=delta[0, audit].cpu().numpy(), actual_delta=actual[0, audit].cpu().numpy())
            return changed if alpha == 1 and bool(selected.any()) else None

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
                    probe_ids = sorted({int(i) for i in inds} | {tok.encode(word,add_special_tokens=False)[0] for word in candidates.values()})
                    arrays.update(first_probe_ids=np.array(probe_ids,dtype=np.int64), first_probe_logits=logits[probe_ids].cpu().numpy(), first_logsumexp=np.array([float(logits.logsumexp(-1))],dtype=np.float32))
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
                # Candidates are output-token continuations; the supplied prompt is never retokenized.
                state.update(context='score-' + name, terminal=False)
                score_cache = copy.deepcopy(cache0)
                conditional = []
                score_logits = first_logits
                score_mask = torch.ones((1, len(prompt_ids)), device='cuda', dtype=torch.long)
                for index, ident in enumerate(tokens):
                    prob = score_logits.softmax(-1)
                    conditional.append(dict(token_id=ident, token=decode(ident), probability=float(prob[ident]), log_probability=float(score_logits.log_softmax(-1)[ident]), logit=float(score_logits[ident]), logsumexp=float(score_logits.logsumexp(-1))))
                    state['current'] = output_meta(index, ident, candidate=True)
                    score_ids = torch.tensor([[ident]], device='cuda')
                    score_mask = torch.cat([score_mask, torch.ones_like(score_ids)], dim=1)
                    score_logits, score_cache = forward(score_ids, score_mask, score_cache)
                end_log_prob = min(0., float(torch.logsumexp(score_logits[sorted(eos)],dim=0)-torch.logsumexp(score_logits,dim=0)))
                end_prob = math.exp(end_log_prob)
                word_log_prob = sum(t['log_probability'] for t in conditional)
                probability = math.exp(word_log_prob)
                candidate_scores[name] = dict(text=word, token_ids=tokens, conditional=conditional, word_probability=probability, word_log_probability=word_log_prob, float32_probability_product=math.prod(t['probability'] for t in conditional), eos_probability_given_word=end_prob, exact_word_then_eos_probability=probability * end_prob)
        finally:
            for handle in handles:
                handle.remove()
        assert hooks() == 0
        if condition != 'native':
            generation_trace = [t for t in trace if t['context'] == 'generation']
            assert [t['token_id'] for t in generation_trace] == prompt_ids + generated
            assert sum(t['trace_only_terminal'] for t in generation_trace) == 1
            if alpha == 1:
                selected_trace = [t for t in trace if t['edited']]
                assert {t['position'] for t in selected_trace} == {t['position'] for t in generation_trace if t['phase']=='input' and t['position'] in selected_positions and t['natural_gate']>0}
                assert all(t['context'] == 'generation' and t['phase'] == 'input' and t['effective_gate'] == target_value for t in selected_trace)
            assert all(t['actual_delta_norm'] == t['intended_delta_norm'] == 0 for t in trace if not t['edited'])
        path = OUT / f"{case['id']}--{condition}.npz"
        np.savez_compressed(path, **arrays)
        return dict(condition=condition, scope=condition, target_gate=0.0 if alpha else None, alpha=alpha, native=native, candidates=candidate_scores, trace=trace, arrays=dict(file=path.name, sha256=sha(path)))

    result=dict(status='running',phase='baseline',protocol=protocol,protocol_sha256=sha(HERE/'protocol.json'),source_sha256=sha(HERE/'run.py'),runtime=dict(torch=torch.__version__,transformers=transformers.__version__,native_dtype='BF16',observer_dtype='FP32',attention='eager',TF32=False,threshold=float(w['theta'][0]),eos_ids=sorted(eos),model_checkpoint_verified=True,full_molt_checkpoint_verified=True),cases=[])
    def normalize(s):
        return s.strip().strip('`*_"\'“”‘’').rstrip('.!?').strip().casefold()
    def correct(case,response):
        if case['prompt_format']=='chat':return normalize(response['response'])==case['expected'].casefold() and response['stop_reason']=='eos'
        import re
        return re.match(r'^\s*'+re.escape(case['expected'])+r'(?![A-Za-z])',response['response'],re.I) is not None
    def candidates(case):
        targets={'expected':case['expected'],'water':'water','ice':'ice'}
        if case['calibration']:targets={'month11':'November','month12':'December'}
        values={};seen=set()
        for name,word in targets.items():
            for i,variant in enumerate(dict.fromkeys([word.lower(),word.title(),' '+word.lower(),' '+word.title()])):
                ids=tuple(tok.encode(variant,add_special_tokens=False))
                if ids in seen:continue
                seen.add(ids);values[name+'-'+str(i)]=variant
        return values
    for case in copy.deepcopy(protocol['cases']):
        native=run(case,'native',candidates(case));zero=run(case,'zero',candidates(case))
        assert native['native']==zero['native'] and native['candidates']==zero['candidates']
        row=dict(id=case['id'],case=case,native=native,zero=zero,native_correct=correct(case,native['native']),runs=[])
        result['cases'].append(row);save(OUT/'results.json',result)
        print(json.dumps(dict(phase='baseline',id=case['id'],response=native['native']['response'],correct=row['native_correct'])),flush=True)
    assert all(c['native_correct'] for c in result['cases'] if c['case']['calibration'])
    save(OUT/'baseline-results.json',result)
    result['baseline_lock']=dict(file='baseline-results.json',sha256=sha(OUT/'baseline-results.json'),frozen_at_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'baseline-lock.json',result['baseline_lock']);result['phase']='interventions';save(OUT/'results.json',result)
    for row in result['cases']:
        if not row['native_correct']:continue
        case=row['case']
        for scope in protocol['scopes']:
            v=run(case,scope,candidates(case));row['runs'].append(v)
            if case['id']=='month-calibration-11' and scope=='off-last':assert v['native']['response'].lstrip().casefold().startswith('december')
            save(OUT/'results.json',result)
            print(json.dumps(dict(phase='intervention',id=case['id'],scope=scope,response=v['native']['response'],edited_input_tokens=sum(t['edited'] for t in v['trace']))),flush=True)
    result.update(status='complete',hooks_remaining=hooks(),runtime_seconds=time.monotonic()-started,finished_at_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'results.json',result)
    save(OUT/'transfer-manifest.json',dict(files=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.iterdir()) if f.is_file()]))
    print(json.dumps(dict(status='complete',cases=len(result['cases']),runtime_seconds=result['runtime_seconds'])),flush=True)

if __name__=='__main__':main()
