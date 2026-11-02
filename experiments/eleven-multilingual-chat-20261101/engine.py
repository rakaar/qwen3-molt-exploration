"""Selected-position fixed gate replacement; uses the already-loaded native model."""
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def assessment(case, answer, stop):
    found = []
    if case['kind'] == 'number':
        found.extend((m.start(), -len(m.group()), m.group()) for m in re.finditer(r'(?<![A-Za-z0-9])[-+]?\d+(?![A-Za-z0-9])', answer))
    for canonical, aliases in case['aliases'].items():
        for alias in aliases:
            pattern = r'(?<![A-Za-z0-9])' + re.escape(alias) + r'(?![A-Za-z0-9])'
            found.extend((m.start(), -len(m.group()), canonical) for m in re.finditer(pattern, answer, re.I))
    found.sort()
    first = found[0][2] if found else None
    return {'first_recognized_answer': first, 'first_matches_expected': first == case['original_expected'],
            'first_matches_target': first == case['target_expected'],
            'target_mentioned': any(t[2] == case['target_expected'] for t in found),
            'truncated': stop == 'token_limit'}


class Selective:
    def __init__(self, model, tokenizer):
        OUT.mkdir(exist_ok=True)
        self.model, self.tok = model, tokenizer
        self.protocol = json.loads((ROOT / 'protocol.json').read_text())
        self.plan = json.loads((ROOT / 'plan.json').read_text())
        self.cases = json.loads((ROOT / 'cases.json').read_text())
        manifest = json.loads((ROOT / 'weights/manifest.json').read_text())
        self.feature = manifest['features'][0]
        assert (self.feature['layer'], self.feature['transform']) == (29, 2375)
        assert manifest['revision'] == self.protocol['molt_revision']
        file = ROOT / 'weights' / self.feature['file']
        assert sha(file) == self.feature['sha256']
        with np.load(file, allow_pickle=False) as a:
            self.w = {k: torch.from_numpy(a[k].copy()).float().cuda() for k in a.files}
        assert self.w['U'].shape == (32, 2560) and self.w['V'].shape == (2560, 32)
        assert torch.__version__ == '2.14.0+cu126' and transformers.__version__ == '4.57.6'
        assert not model.training and next(model.parameters()).dtype == torch.bfloat16
        assert self.hooks() == 0
        self.attention = [b.self_attn.forward for b in model.model.layers]
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.started = time.monotonic()

    def hooks(self):
        return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in self.model.modules())

    @torch.inference_mode()
    def run(self, case, prompt, name, spec=None, target=None, alpha=1., replay=None, audit=False):
        tok, model, w = self.tok, self.model, self.w
        if self.protocol.get('prompt_format') == 'raw':
            chat = prompt
        else:
            chat = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=self.protocol['enable_thinking'])
        prompt_ids = tok.encode(chat, add_special_tokens=False)
        state = {'step': 0, 'prefix_ids': list(prompt_ids)}
        traces, vectors, scores, generated, handles = [], [], [], [], []
        probe_texts = list(dict.fromkeys([text for aliases in case['aliases'].values() for text in aliases]))
        probe_texts = list(dict.fromkeys(probe_texts + [s.casefold() for s in probe_texts]))
        probe_ids = sorted(set(sum([tok.encode(t, add_special_tokens=False) for s in probe_texts for t in [s, ' ' + s]], []) + list(self.eos)))


        def observe(_module, inputs):
            z = (inputs[0].detach().float() - w['input_mean']) / w['input_std']
            raw = F.linear(z, w['e'].reshape(1, -1), w['bias']).squeeze(-1)
            gate = torch.where((raw > w['theta'][0]) & (raw > 0), raw, 0.)
            state.update(z=z, raw=raw, gate=gate, operator=(z @ w['V'] @ w['U']) * w['output_std'])

        def edit(_module, _inputs, output):
            positions = list(range(len(prompt_ids))) if state['step'] == 0 else [len(prompt_ids) + state['step'] - 1]
            token_ids = prompt_ids if state['step'] == 0 else [state['prefix_ids'][-1]]
            phase = 'input' if state['step'] == 0 else 'output'
            chosen = [spec['kind'] == 'all' or spec['kind'] == phase
                      or (spec['kind'] == 'positions' and pos in spec['positions']) for pos in positions]
            selected = torch.tensor([chosen], device='cuda', dtype=torch.bool)
            fixed_values = [float(replay[str(p)]) if replay and str(p) in replay else float(target) for p in positions]
            fixed = torch.tensor([fixed_values], device='cuda', dtype=torch.float32)
            delta = torch.where(selected.unsqueeze(-1),
                                float(alpha) * (fixed - state['gate']).unsqueeze(-1) * state['operator'], 0.)
            assert torch.isfinite(delta).all()
            changed = (output.float() + delta).to(output.dtype)
            assert torch.isfinite(changed).all()
            assert torch.equal(changed[~selected], output[~selected]), 'Unselected local MLP output changed'
            if alpha == 0.:
                assert torch.equal(changed, output)
            actual = changed.float() - output.float()
            for j, (pos, token) in enumerate(zip(positions, token_ids)):
                traces.append({'position': pos, 'phase': phase, 'token_id': token, 'token': tok.decode([token]),
                               'selected': chosen[j], 'raw': float(state['raw'][0, j]),
                               'natural_gate': float(state['gate'][0, j]), 'requested_gate': fixed_values[j], 'alpha': alpha,
                               'operator_norm': float(state['operator'][0, j].norm()),
                               'intended_delta_norm': float(delta[0, j].norm()),
                               'actual_delta_norm': float(actual[0, j].norm()), 'unselected_unchanged': not chosen[j]})
            if audit and state['step'] in self.protocol['audit_steps']:
                js = sorted(set([0, len(positions) - 1] + [j for j, flag in enumerate(chosen) if flag][-3:]))
                file = f"{case['id']}--{name}--step-{state['step']}.npz"
                np.savez_compressed(OUT / file, positions=np.array([positions[j] for j in js]),
                                    selected=selected[:, js].cpu().numpy(), fixed=fixed[:, js].cpu().numpy(),
                                    z=state['z'][:, js].cpu().numpy(), raw=state['raw'][:, js].cpu().numpy(),
                                    gate=state['gate'][:, js].cpu().numpy(), operator=state['operator'][:, js].cpu().numpy(),
                                    native_mlp=output[:, js].float().cpu().numpy(), edited_mlp=changed[:, js].float().cpu().numpy(),
                                    intended_delta=delta[:, js].cpu().numpy(), actual_delta=actual[:, js].cpu().numpy())
                vectors.append({'file': file, 'sha256': sha(OUT / file), 'alpha': alpha})
            return changed

        if spec is not None:
            assert target is not None
            block = model.model.layers[29]
            handles = [block.post_attention_layernorm.register_forward_pre_hook(observe), block.mlp.register_forward_hook(edit)]
        ids = torch.tensor([prompt_ids], device='cuda')
        mask, cache, stop = torch.ones_like(ids), None, 'token_limit'
        try:
            for step in range(self.protocol['max_new_tokens']):
                assert time.monotonic() - self.started < self.protocol['max_runtime_minutes'] * 60
                state['step'] = step
                pred = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
                cache = pred.past_key_values
                logits = pred.logits[0, -1].float()
                assert torch.isfinite(logits).all()
                lp = logits.log_softmax(-1)
                greedy = int(logits.argmax())
                values, indices = lp.exp().topk(6)
                scores.append({'position': step, 'token_id': greedy, 'token': tok.decode([greedy]),
                               'selected_probability': float(lp[greedy].exp()), 'selected_logit': float(logits[greedy]),
                               'logsumexp': float(logits.logsumexp(-1)),
                               'probe_probabilities': {str(i): float(lp[i].exp()) for i in probe_ids},
                               'top_tokens': [{'token_id': int(i), 'token': tok.decode([int(i)]), 'probability': float(p)}
                                              for i, p in zip(indices, values)]})
                generated.append(greedy)
                state['prefix_ids'].append(greedy)
                if greedy in self.eos:
                    stop = 'eos'
                    break
                ids = torch.tensor([[greedy]], device='cuda')
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            assert all(b.self_attn.forward == f for b, f in zip(model.model.layers, self.attention))
        finally:
            for handle in handles:
                handle.remove()
        assert self.hooks() == 0
        if spec is not None:
            assert len(traces) == len(prompt_ids) + len(generated) - 1
            assert [t['position'] for t in traces] == list(range(len(traces)))
        answer = tok.decode(generated, skip_special_tokens=True)
        return {'case_id': case['id'], 'name': name, 'prompt': prompt, 'chat': chat, 'spec': spec,
                'prompt_token_ids': prompt_ids, 'output_token_ids': generated, 'target': target, 'alpha': alpha,
                'replay': replay, 'answer': answer, 'stop_reason': stop, 'trace': traces, 'vectors': vectors,
                'steps': scores, 'assessment': assessment(case, answer, stop), 'hooks_remaining': 0}

