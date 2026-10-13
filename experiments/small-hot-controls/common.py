"""Original-model, single-position MOLT controls; no cloud authentication."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
P = json.loads((ROOT / 'protocol.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, obj):
    temp = OUT / (name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(OUT / name)


class Experiment:
    def __init__(self):
        torch.manual_seed(0)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen3-4b', local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            ROOT / 'models/qwen3-4b', local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='eager').cuda().eval().requires_grad_(False)
        self.model.config.use_cache = False
        self.attention_methods = [b.self_attn.forward for b in self.model.model.layers]
        path = ROOT / 'weights/molt-22-selected.safetensors'
        assert sha(path) == P['selected_asset_sha256']
        self.w = {k: v.cuda() for k, v in load_file(path).items()}
        self.receiver, self.receiver_tokens = self.encode(P['receiver'], 'small')
        self.donor, self.donor_tokens = self.encode(P['donor'], 'hot')
        assert self.tokenizer.encode('cold', add_special_tokens=False) == [87072]
        assert self.tokenizer.encode('large', add_special_tokens=False) == [16767]
        self.clean_reference = None

    def encode(self, prompt, operand):
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                               return_tensors='pt')
        offsets = batch.pop('offset_mapping')[0].tolist()
        start = text.index(operand, text.index(prompt))
        positions = [i for i, (a, b) in enumerate(offsets)
                     if b > a and b > start and a < start + len(operand)]
        assert positions == [P['position']] == [9]
        ids = batch['input_ids'].cuda()
        assert ids.shape == (1, P['sequence_length']) == (1, 29)
        tokens = [{'position': i, 'id': int(v), 'text': self.tokenizer.decode([int(v)])}
                  for i, v in enumerate(ids[0])]
        return ids, tokens

    def summary(self, logits):
        probability = logits.float().softmax(-1)
        values, ids = probability.topk(8)
        return {'P_cold': float(probability[87072]), 'P_large': float(probability[16767]),
                'top_token': self.tokenizer.decode([int(ids[0])]),
                'top_tokens': [{'id': int(i), 'text': self.tokenizer.decode([int(i)]),
                                'probability': float(v)} for i, v in zip(ids, values)]}

    @torch.inference_mode()
    def forward(self, donor=False, edits=None, attention=False):
        """Always use eight identical prompts; edits cannot mix attention between rows."""
        size, pos = P['batch_size'], P['position']
        ids = (self.donor if donor else self.receiver).repeat(size, 1)
        assert all(b.self_attn.forward == a for b, a in
                   zip(self.model.model.layers, self.attention_methods))
        assert all(not b.mlp._forward_hooks for b in self.model.model.layers)
        c = SimpleNamespace()
        block = self.model.model.layers[P['layer']]

        def pre(_module, args):
            c.prenorm = args[0].detach().clone()

        def mlp(_module, args, output):
            c.mlp_input = args[0].detach().clone()
            c.before = output.detach().clone()
            if self.clean_reference is not None and not donor:
                assert torch.equal(c.prenorm, self.clean_reference.prenorm)
                assert torch.equal(c.mlp_input, self.clean_reference.mlp_input)
                assert torch.equal(c.before, self.clean_reference.before)
            changed = output.clone()
            for row, edit in enumerate(edits or []):
                if edit['mode'] == 'replace':
                    changed[row, pos] = edit['vector'].to(output.dtype)
                    assert torch.equal(changed[row, pos], edit['vector'].to(output.dtype))
                else:
                    assert edit['mode'] == 'add'
                    # Preserve the original matched experiment's BF16 write ordering.
                    changed[row, pos] = output[row, pos] + edit['vector'].to(output.dtype)
            mask = torch.ones(output.shape[1], device=output.device, dtype=torch.bool)
            mask[pos] = False
            assert torch.equal(changed[:, mask], output[:, mask])
            c.after = changed.detach().clone()
            return changed

        def residual(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            c.residual = hidden[:, pos].detach().clone()

        handles = [block.post_attention_layernorm.register_forward_pre_hook(pre),
                   block.mlp.register_forward_hook(mlp), block.register_forward_hook(residual)]
        try:
            output = self.model(ids, use_cache=False, output_attentions=attention)
            c.logits = output.logits[:, -1].detach().clone()
            c.attention = [a.detach().cpu() for a in output.attentions] if attention else None
        finally:
            for handle in handles:
                handle.remove()
        assert torch.equal(c.residual, c.prenorm[:, pos] + c.after[:, pos])
        return c

    def gates(self, c):
        x = (c.prenorm[0, P['position']].float() - self.w['input_standardizer.mean'][22]) / self.w['input_standardizer.std'][22]
        raw = F.linear(x, self.w['e.weight'], self.w['e.bias'])
        threshold = self.w['nonlinearity.theta'].reshape(-1)
        gate = raw.reshape(-1) * ((raw.reshape(-1) > threshold) & (raw.reshape(-1) > 0))
        assert gate.shape == (2480,) and torch.isfinite(gate).all()
        return x, gate

    def parts(self, x, gate, feature, selected=None):
        weights = selected or self.w
        if gate[feature] == 0:
            return torch.zeros(2560, device='cuda', dtype=torch.float32)
        lowrank = (x @ weights[f'V_{feature}']) @ weights[f'U_{feature}']
        return gate[feature] * lowrank * self.w['output_standardizer.std'][22]


def lossless_logits(logits):
    array = logits.float().cpu().numpy()
    smaller = array.astype(np.float16)
    assert np.array_equal(smaller.astype(np.float32), array), 'Logit storage is not lossless'
    return smaller
