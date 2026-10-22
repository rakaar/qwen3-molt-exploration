"""Frozen medical-adapter MOLT interventions; no downloads or pod lifecycle code.

Run locally on the authorized GPU host: python runner.py [--audit-only].
Protocol contract is documented in validate_protocol(). Outputs are atomic JSON.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DEADLINE = None

CONDITIONS = ('native', 'subtract_response', 'subtract_query_response',
              'replace_response', 'replace_query_response',
              'replace_zero_response', 'replace_zero_query_response')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def finite(tensor, name):
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f'Nonfinite {name}')


def norm(tensor):
    return float(tensor.detach().double().norm().item())


def validate_protocol(p):
    """Required keys: base_revision, adapter_sha256, molt_revision, cases.

    Each case: id, text, feature={layer,transform,label}, seeds=[10 integers].
    Top-level seeds may supply the case seeds. Optional input_ids must exactly
    match fresh nonthinking, single-user-message chat-template tokenization.
    paths: assets_root (default /root/reconstruction), base_manifest and
    molt_manifest (defaults inside assets_root), model, adapter, molts.
    All manifest hashes are verified before any model execution. Cases can
    alternatively be stored in cases_file (JSON array or JSONL, relative ROOT).
    """
    for key in ('base_revision', 'adapter_sha256', 'molt_revision'):
        assert isinstance(p[key], str) and p[key]
    assert re.fullmatch(r'[0-9a-f]{64}', p['adapter_sha256'])
    assert int(p.get('max_new_tokens', 600)) == 600
    assert float(p.get('temperature', 1.0)) == 1.0
    assert float(p.get('top_p', 1.0)) == 1.0
    assert int(p.get('top_k', 0)) == 0
    generation = p.get('generation', {})
    for key, expected in {'temperature':1.0,'top_p':1.0,'top_k':0,
                          'max_new_tokens':600,'repetition_penalty':1.0,
                          'enable_thinking':False,'system_message':None}.items():
        assert generation.get(key, expected) == expected, f'Unsupported generation option {key}'
    if 'conditions' in p:
        assert tuple(p['conditions']) == CONDITIONS
    cases = p.get('cases')
    if cases is None:
        source = ROOT / p['cases_file']
        content = source.read_text()
        cases = json.loads(content) if source.suffix == '.json' else [json.loads(x) for x in content.splitlines() if x.strip()]
    assert cases and len({x['id'] for x in cases}) == len(cases)
    for case in cases:
        assert re.fullmatch(r'[A-Za-z0-9_.-]+', case['id']), 'Unsafe case ID'
        assert isinstance(case['text'], str) and case['text']
        feature = case['feature']
        assert 0 <= feature['layer'] < 36 and 0 <= feature['transform'] < 2480
        assert isinstance(feature['label'], str)
        case['seeds'] = case.get('seeds', p.get('seeds'))
        assert len(case['seeds']) == 10 and len(set(case['seeds'])) == 10
        assert all(type(s) is int and 0 <= s < 2**63 for s in case['seeds'])
    return cases


def resolve_paths(p):
    paths = p.get('paths', {})
    assets = Path(paths.get('assets_root', '/root/reconstruction'))
    def get(key, default):
        result = Path(paths.get(key, default))
        return result if result.is_absolute() else ROOT / result
    return dict(assets=assets, model=get('model', assets/'model'),
                adapter=get('adapter', assets/'adapter'), molts=get('molts', assets/'molts'),
                base_manifest=get('base_manifest', assets/'expected-base-verification.json'),
                molt_manifest=get('molt_manifest', assets/'molt-manifest.json'))


def verify_assets(p, paths, layers):
    base_manifest = json.loads(paths['base_manifest'].read_text())
    assert base_manifest['revision'] == p['base_revision']
    assert base_manifest['files']
    verified_base = []
    for row in base_manifest['files']:
        path = paths['model'] / row['file']
        assert sha(path) == row['sha256'], f'Base hash mismatch: {path}'
        verified_base.append(dict(file=str(path), sha256=row['sha256']))
    adapter = paths['adapter'] / 'adapter_model.safetensors'
    assert sha(adapter) == p['adapter_sha256'], 'Adapter hash mismatch'
    manifest = json.loads(paths['molt_manifest'].read_text())
    assert manifest['revision'] == p['molt_revision']
    assets = {}
    for layer in layers:
        rows = [r for r in manifest['files'] if r.get('layer') == layer and str(r['file']).endswith('.safetensors')]
        assert len(rows) == 1, f'Need one pinned MOLT weight hash for layer {layer}'
        row = rows[0]
        path = paths['molts'] / row['file']
        # Remote deployments may rename checkpoint.safetensors to model.safetensors;
        # acceptance still requires the pinned checkpoint's exact SHA-256.
        if not path.exists():
            alternative = path.parent/'model.safetensors'
            assert alternative.exists(), str(path)
            path = alternative
        assert sha(path) == row['sha256'], f'MOLT hash mismatch: {path}'
        config = path.parent/'config.json'
        assert config.exists(), f'Missing MOLT config {config}'
        configs = [r for r in manifest['files'] if r.get('layer') == layer and str(r['file']).endswith('config.json')]
        if configs:
            assert len(configs) == 1 and sha(config) == configs[0]['sha256']
        assets[layer] = dict(path=path, config=config, sha256=row['sha256'], config_sha256=sha(config))
    return assets, dict(base_files=verified_base, adapter_sha256=sha(adapter),
                        base_manifest_sha256=sha(paths['base_manifest']),
                        molt_manifest_sha256=sha(paths['molt_manifest']))


def load_model(paths):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    tok = AutoTokenizer.from_pretrained(paths['model'], local_files_only=True, use_fast=True)
    assert tok.is_fast, 'Offset mapping requires fast tokenizer'
    base = AutoModelForCausalLM.from_pretrained(paths['model'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0').eval().requires_grad_(False)
    model = PeftModel.from_pretrained(base, paths['adapter'], is_trainable=False).eval().requires_grad_(False)
    saved = load_file(str(paths['adapter']/'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model)
    assert len(saved) == 504 and saved.keys() == actual.keys(), 'Adapter tensor inventory mismatch'
    for key in saved:
        assert torch.equal(saved[key], actual[key].detach().cpu()), f'Adapter tensor mismatch {key}'
    del saved, actual
    assert not any(p.requires_grad for p in model.parameters())
    blocks = model.get_base_model().model.layers
    assert len(blocks) == 36
    assert not getattr(model, '_disable_adapters', False)
    return model, tok, blocks


def prompt(tok, case):
    messages = [{'role': 'user', 'content': case['text']}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    marker = '__MOLT_CONTENT_BOUNDARY_873849753__'
    assert marker not in case['text']
    template = tok.apply_chat_template([{'role':'user','content':marker}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    assert template.count(marker) == 1
    prefix, suffix = template.split(marker)
    assert rendered == prefix + case['text'] + suffix, 'Template altered content unexpectedly'
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    template_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert ids == template_ids
    if 'input_ids' in case:
        assert ids == case['input_ids'], f'Saved question IDs differ for {case["id"]}'
    start, end = len(prefix), len(prefix)+len(case['text'])
    # Mixed boundary tokens are rejected: only exclusively user-content tokens
    # may be edited. This prevents accidentally editing template positions.
    query = [i for i,(a,b) in enumerate(offsets) if b>a and a>=start and b<=end]
    assert query and len(ids)-1 not in query
    mixed = [i for i,(a,b) in enumerate(offsets) if b>start and a<end and i not in query]
    assert not mixed, f'User/template token boundary overlap: {mixed}'
    return dict(input_ids=ids, input_ids_sha256=digest(ids), rendered_prompt_sha256=digest(rendered),
                query_content_indices=query, response_prefill_index=len(ids)-1,
                content_character_span=[start,end], matched_prompt_input_ids=('input_ids' in case),
                system_message=False, enable_thinking=False)


class Intervention:
    """Observe the residual before RMSNorm; edit only the chosen raw MLP output."""
    def __init__(self, block, molt, feature, info, condition, *, alpha=1., force_gate_zero=False):
        assert condition in CONDITIONS
        self.block, self.molt, self.feature, self.info = block, molt, feature, info
        self.condition, self.alpha, self.force_gate_zero = condition, alpha, force_gate_zero
        self.handles, self.traces, self.calls = [], [], []
        self.processed = 0
        self.x = None
        self.prefill_x = None
        self.audit_points = []

    def __enter__(self):
        self.handles = [self.block.post_attention_layernorm.register_forward_pre_hook(self.pre),
                        self.block.mlp.register_forward_hook(self.post)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def pre(self, module, args):
        assert self.x is None, 'Unconsumed residual capture'
        assert args[0].shape[0] == 1 and args[0].dtype == torch.bfloat16
        self.x = args[0].detach().clone()
        if self.processed == 0:
            self.prefill_x = self.x.clone()  # Entire prefill, including excluded positions.
        finite(self.x, 'pre-normalization residual')

    def contribution(self, x):
        flat = x.float().reshape(-1, self.molt.width)
        with torch.autocast(device_type='cuda', enabled=False):
            z = (flat-self.molt.input_mean)/self.molt.input_std
            feature = self.feature
            raw_gate = (z @ self.molt.encoder[feature:feature+1].T).flatten()+self.molt.bias[feature]
            finite(z, 'standardized residual')
            finite(raw_gate, 'selected gate preactivation')
            group = next(g for g,(lo,hi,_) in enumerate(self.molt.groups) if lo<=feature<hi)
            local = feature-self.molt.groups[group][0]
            gate = torch.where((raw_gate>self.molt.theta[feature]) & (raw_gate>0),raw_gate,0.)
            if self.force_gate_zero:
                gate = torch.zeros_like(gate)
            vector = ((z @ self.molt.Vs[group][local]) @ self.molt.Us[group][local])
            contribution = vector * gate[:,None] * self.molt.output_std
        finite(contribution, 'feature contribution')
        return contribution, gate

    def post(self, module, args, output):
        assert self.x is not None
        x, self.x = self.x, None
        assert output.shape == x.shape and output.dtype == torch.bfloat16
        finite(output, 'native raw MLP output')
        count = x.shape[1]
        if self.processed == 0:
            assert count == len(self.info['input_ids'])
        else:
            assert count == 1, 'Cached decoding must process exactly one token'
        full_positions = list(range(self.processed, self.processed+count))
        query_scope = 'query_response' in self.condition
        eligible = [j for j,pos in enumerate(full_positions)
                    if pos >= self.info['response_prefill_index'] or
                    (query_scope and pos in self.info['query_content_indices'])]
        self.calls.append(dict(call=len(self.calls), cache_phase='prefill' if self.processed==0 else 'decode',
                               full_positions=full_positions, eligible_local_indices=eligible,
                               captured_positions=count))
        self.processed += count
        if not eligible:
            return None
        selected = torch.tensor(eligible, device=x.device)
        residual = x[0].index_select(0, selected)
        native = output[0].index_select(0, selected)
        contribution, gates = self.contribution(residual)
        raw = None
        if self.condition.startswith('replace'):
            raw, _, _ = self.molt(residual)
            target = raw-contribution if self.condition.startswith('replace_zero') else raw
            intended = target-native.float()
            edited = target.to(dtype=torch.bfloat16)
            assert torch.equal(edited, target.to(torch.bfloat16)), 'Replacement conversion mismatch'
        elif self.condition.startswith('subtract') and self.alpha != 0:
            intended = -self.alpha*contribution
            edited = (native.float()+intended).to(torch.bfloat16)
            inactive = gates == 0
            assert torch.equal(edited[inactive], native[inactive]), 'Inactive gate changed native output'
        else:
            intended = torch.zeros_like(contribution)
            edited = native
        finite(edited, 'edited MLP output')
        expected = edited.clone()
        result = output.clone()
        result[0, selected] = edited
        assert torch.equal(result[0].index_select(0, selected), expected), 'Edited output did not match intended BF16 output'
        excluded = [j for j in range(count) if j not in eligible]
        if excluded:
            assert torch.equal(result[:,excluded], output[:,excluded]), 'Excluded template output changed'
        for k,j in enumerate(eligible):
            pos = full_positions[j]
            native_norm = norm(native[k])
            actual_norm = norm(edited[k].float()-native[k].float())
            gate = float(gates[k])
            self.traces.append(dict(full_position=pos,
                processed_input_token=(self.info['input_ids'][pos] if pos<len(self.info['input_ids']) else None),
                phase='query' if pos<self.info['response_prefill_index'] else 'response',
                predicted_answer_index=pos-self.info['response_prefill_index'] if pos>=self.info['response_prefill_index'] else None,
                natural_gate=gate, contribution_l2_norm=norm(contribution[k]),
                intended_delta_l2_norm=norm(intended[k]), actual_bf16_delta_l2_norm=actual_norm,
                native_mlp_l2_norm=native_norm, effect_ratio=actual_norm/native_norm if native_norm else None,
                gate_inactive_bit_exact=(torch.equal(edited[k],native[k]) if gate==0 and self.condition.startswith('subtract') else None)))
        if len(self.audit_points)<3:
            take = min(3-len(self.audit_points),len(residual))
            self.audit_points.extend([residual[j].detach().clone() for j in range(take)])
        return result

    def validate_calls(self, generated):
        assert len(self.calls) == len(generated)
        assert self.processed == len(self.info['input_ids'])+len(generated)-1
        for call in self.calls:
            expected = [j for j,pos in enumerate(call['full_positions'])
                        if pos>=self.info['response_prefill_index'] or
                        ('query_response' in self.condition and pos in self.info['query_content_indices'])]
            assert call['eligible_local_indices'] == expected
        for trace in self.traces:
            position = trace['full_position']
            if position>=len(self.info['input_ids']):
                trace['processed_input_token'] = generated[position-len(self.info['input_ids'])]
        assert self.prefill_x is not None and self.prefill_x.shape[1] == len(self.info['input_ids'])


@torch.inference_mode()
def decode(model, tok, info, seed, *, intervention=None, max_tokens=600, collect_logits=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ids = torch.tensor([info['input_ids']],dtype=torch.long,device='cuda')
    generated, logits_digests = [], []
    first_logits = None
    past = None
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos or [])
    started = time.time()
    manager = intervention if intervention is not None else contextlib.nullcontext()
    with manager:
        for step in range(max_tokens):
            if DEADLINE is not None and time.monotonic() >= DEADLINE:
                raise TimeoutError('Protocol max_runtime_minutes reached; completed samples retained; GPU left running')
            attention = torch.ones((1,len(info['input_ids'])+step),dtype=torch.long,device='cuda')
            result = model(input_ids=ids, attention_mask=attention, past_key_values=past,
                           use_cache=True, logits_to_keep=1, return_dict=True)
            logits = result.logits[0,-1]
            finite(logits,'sampling logits')
            assert logits.dtype == torch.bfloat16
            if collect_logits:
                logits_digests.append(hashlib.sha256(logits.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                if first_logits is None:
                    first_logits = logits.clone()
            probabilities = torch.softmax(logits.float(), dim=-1)
            finite(probabilities,'sampling probabilities')
            token = int(torch.multinomial(probabilities,1).item())
            generated.append(token)
            past = result.past_key_values
            del result
            if token in eos:
                reason = 'eos'
                break
            ids = torch.tensor([[token]],dtype=torch.long,device='cuda')
        else:
            reason = 'max_new_tokens'
    if intervention is not None:
        intervention.validate_calls(generated)
    torch.cuda.synchronize()
    elapsed = time.time()-started
    del past
    return dict(generated_token_ids=generated, generated_token_ids_sha256=digest(generated),
                text=tok.decode(generated,skip_special_tokens=True),
                answer=tok.decode(generated,skip_special_tokens=True),
                text_sha256=hashlib.sha256(tok.decode(generated,skip_special_tokens=True).encode()).hexdigest(),
                answer_sha256=hashlib.sha256(tok.decode(generated,skip_special_tokens=True).encode()).hexdigest(),
                text_with_special_tokens=tok.decode(generated,skip_special_tokens=False),
                stop_reason=reason, elapsed_seconds=elapsed, logits_sha256=logits_digests), first_logits


def trace_summary(traces):
    rows = {}
    for phase in ('query','response','all'):
        selected = [t for t in traces if phase=='all' or t['phase']==phase]
        if not selected:
            continue
        entry = dict(positions=len(selected), active_positions=sum(t['natural_gate']>0 for t in selected),
                     changed_positions=sum(t['actual_bf16_delta_l2_norm']>0 for t in selected))
        for name in ('natural_gate','contribution_l2_norm','intended_delta_l2_norm',
                     'actual_bf16_delta_l2_norm','native_mlp_l2_norm','effect_ratio'):
            values = [t[name] for t in selected if t[name] is not None]
            entry[name] = dict(mean=sum(values)/len(values),max=max(values),sum=sum(values)) if values else None
        rows[phase] = entry
    return rows


@torch.inference_mode()
def audit(model, tok, block, molt, case, info, output):
    """Assertions are mandatory; no fallback tolerances or bypass switches."""
    seed, feature = case['seeds'][0], case['feature']['transform']
    records = dict(layer=case['feature']['layer'], case_id=case['id'], seed=seed)
    steps = 4
    plain, plain_logits = decode(model,tok,info,seed,max_tokens=steps,collect_logits=True)
    passive_hook = Intervention(block,molt,feature,info,'native')
    passive, passive_logits = decode(model,tok,info,seed,intervention=passive_hook,max_tokens=steps,collect_logits=True)
    assert torch.equal(plain_logits,passive_logits), 'Passive hooks changed first logits'
    assert plain['logits_sha256']==passive['logits_sha256'], 'Passive hooks changed cached logits'
    assert plain['generated_token_ids']==passive['generated_token_ids'], 'Passive hooks changed same-seed sequence'
    records['passive'] = dict(bit_identical_first_logits=True,bit_identical_cached_logits=True,
                              same_seed_tokens=True,token_ids=plain['generated_token_ids'])
    for name,kwargs in (('alpha_zero',dict(alpha=0.)), ('inactive_gate',dict(force_gate_zero=True))):
        hook = Intervention(block,molt,feature,info,'subtract_query_response',**kwargs)
        sample,_ = decode(model,tok,info,seed,intervention=hook,max_tokens=steps,collect_logits=True)
        assert sample['logits_sha256']==plain['logits_sha256'], f'{name} altered logits'
        assert sample['generated_token_ids']==plain['generated_token_ids'], f'{name} altered same-seed output'
        assert all(t['actual_bf16_delta_l2_norm']==0 for t in hook.traces)
        records[name] = dict(bit_identical_all_logits=True,same_seed_tokens=True,
                             tested_positions=len(hook.traces),hook_calls=hook.calls)
    # Real, captured prefill residuals independently exercise complete gate-zero
    # reconstruction, not a fixed vector intervention during generation.
    x = passive_hook.prefill_x[0]
    chosen = [info['query_content_indices'][0], info['response_prefill_index']]
    x = x[chosen].float()
    # Pin a genuine naturally active historical input solely for numerical
    # mechanics. No reference activation, vector, or gate enters generation.
    reference = case['audit_reference']
    reference_path = Path(reference['path'])
    assert sha(reference_path) == reference['sha256'], 'Audit reference hash mismatch'
    payload = torch.load(reference_path, map_location='cpu', weights_only=True)
    reference_x = payload[reference['key']]
    assert reference['key'] == f'x{case["feature"]["layer"]}'
    assert reference_x.ndim == 2 and reference_x.shape[-1] == molt.width
    finite(reference_x, 'pinned numerical-audit residuals')
    _, reference_gates = passive_hook.contribution(reference_x.to(device='cuda',dtype=torch.float32))
    active = (reference_gates>0).nonzero(as_tuple=False).flatten()
    assert len(active)>0, 'Pinned audit reference has no naturally active selected feature'
    active_position = int(active[reference_gates[active].argmax()])
    active_x = reference_x[active_position:active_position+1].to(device='cuda',dtype=torch.float32)
    x = torch.cat((x,active_x),dim=0)
    records['active_numerical_audit_reference'] = dict(path=str(reference_path),
        sha256=reference['sha256'],key=reference['key'],position=active_position,
        natural_gate=float(reference_gates[active_position]),
        source_active_positions=len(active),used_for_generation=False)
    del payload, reference_x, reference_gates, active_x
    contribution, gates = passive_hook.contribution(x)
    assert float(gates[-1])>0, 'Real active numerical-audit point became inactive'
    z, allgates = molt._encode(x)
    full,_,_ = molt(x)
    full_gate = allgates[:,feature]
    assert torch.equal(gates>0,full_gate>0), 'Selected-only and full encoder gate masks differ'
    assert bool(((gates-full_gate).abs() <= 1e-5*full_gate.abs()).all()), 'Selected-only/full gate values differ'
    records['selected_encoder_matches_full'] = dict(natural_gates=gates.cpu().tolist(),
        full_encoder_gates=full_gate.cpu().tolist(),gate_mask_identical=True,relative_tolerance=1e-5)
    zero = allgates.clone()
    zero[:,feature] = 0
    explicit = molt._decode(molt._sum_sparse(z,zero,molt.method))
    algebra = full-contribution
    difference = (explicit-algebra).double().norm(dim=-1)
    denominator = explicit.double().norm(dim=-1)
    assert bool((difference<=1e-5*denominator).all()), 'Gate-zero reconstruction algebra failed'
    natural_inactive = allgates[:,feature]==0
    if bool(natural_inactive.any()):
        assert torch.equal(contribution[natural_inactive],torch.zeros_like(contribution[natural_inactive]))
    records['gate_zero_algebra'] = dict(prompt_positions=chosen,reference_position=active_position,
        required_real_active_point_passed=True,norm_rtol=1e-5,
        relative_l2_error=(difference/denominator).cpu().tolist(),natural_gates=gates.cpu().tolist(),
        natural_inactive_positions=int(natural_inactive.sum()))
    records['conditions'] = {}
    for condition in CONDITIONS[1:]:
        hook = Intervention(block,molt,feature,info,condition)
        result,_ = decode(model,tok,info,seed,intervention=hook,max_tokens=steps)
        records['conditions'][condition] = dict(hook_calls=hook.calls,trace_summary=trace_summary(hook.traces),
            token_ids=result['generated_token_ids'],eligible_masks_asserted=True,
            replacement_exact_bf16_asserted=condition.startswith('replace'),
            excluded_positions_bit_exact_asserted=True)
    # A final unhooked pass proves removable hooks did not mutate the model.
    after,_ = decode(model,tok,info,seed,max_tokens=steps,collect_logits=True)
    assert after['logits_sha256']==plain['logits_sha256']
    assert after['generated_token_ids']==plain['generated_token_ids']
    records['baseline_stable_after_all_hooks'] = True
    records['passed'] = True
    write(output,records)
    return records


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output')
    parser.add_argument('--audit-only',action='store_true')
    args = parser.parse_args()
    out = args.output
    def progress(phase,**kwargs):
        row = dict(phase=phase,timestamp=time.time(),**kwargs)
        write(out/'status.json',row)
        print(json.dumps(row),flush=True)
    try:
        protocol = json.loads(args.protocol.read_text())
        cases = validate_protocol(protocol)
        runtime_minutes = float(protocol.get('max_runtime_minutes',60))
        assert 0 < runtime_minutes <= 60
        DEADLINE = time.monotonic()+runtime_minutes*60
        layers = sorted({c['feature']['layer'] for c in cases})
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        paths = resolve_paths(protocol)
        progress('verifying_assets',layers=layers)
        assets, verification = verify_assets(protocol,paths,layers)
        model,tok,blocks = load_model(paths)
        source_hashes = {name:sha(ROOT/name) for name in ('runner.py','reconstruct.py','protocol.json') if (ROOT/name).exists()}
        source_hashes['selected_protocol'] = sha(args.protocol)
        runtime = dict(packages={name:importlib.metadata.version(name) for name in
                       ('torch','transformers','peft','numpy','safetensors')},
                       gpu=torch.cuda.get_device_name(),native_dtype='bfloat16',molt_dtype='float32',
                       attention='sdpa',tf32=False,adapter_enabled=True,adapter_verified_tensors=504,
                       no_training=True,source_hashes=source_hashes,protocol_sha256=sha(args.protocol),
                       base_revision=protocol['base_revision'],adapter_sha256=protocol['adapter_sha256'],
                       adapter_revision=protocol.get('adapter_revision'),molt_revision=protocol['molt_revision'],
                       asset_verification=verification)
        write(out/'runtime.json',runtime)
        prompts = {c['id']:prompt(tok,c) for c in cases}
        write(out/'prompts.json',prompts)
        from reconstruct import FrozenMolt
        summaries = []
        expected = len(cases)*len(CONDITIONS)*10
        for layer in layers:
            asset = assets[layer]
            progress('loading_molt',layer=layer)
            molt = FrozenMolt(asset['path'],layer,'cuda',config_path=asset['config'])
            layer_cases = [c for c in cases if c['feature']['layer']==layer]
            for case in layer_cases:
                # Audit every chosen feature and prompt, before its bulk samples.
                progress('mechanics_audit',layer=layer,case_id=case['id'])
                audit(model,tok,blocks[layer],molt,case,prompts[case['id']],out/'audits'/f'{case["id"]}.json')
                if args.audit_only:
                    continue
                for seed in case['seeds']:
                    for condition in CONDITIONS:
                        destination = out/'samples'/case['id']/condition/f'seed-{seed}.json'
                        info = prompts[case['id']]
                        identity = dict(case_id=case['id'],question=case['text'],feature=case['feature'],
                                        condition=condition,seed=seed,protocol_sha256=runtime['protocol_sha256'],
                                        source_hashes=source_hashes,input_ids_sha256=info['input_ids_sha256'])
                        if destination.exists():
                            old = json.loads(destination.read_text())
                            assert all(old[k]==v for k,v in identity.items()), f'Resume identity mismatch {destination}'
                            assert old['generated_token_ids_sha256']==digest(old['generated_token_ids'])
                            sample = old
                        else:
                            hook = Intervention(blocks[layer],molt,case['feature']['transform'],info,condition)
                            result,_ = decode(model,tok,info,seed,intervention=hook)
                            sample = dict(**identity,**result,input_ids=info['input_ids'],
                                query_content_indices=info['query_content_indices'],
                                response_prefill_index=info['response_prefill_index'],
                                matched_prompt_input_ids=info['matched_prompt_input_ids'],
                                rendered_prompt_sha256=info['rendered_prompt_sha256'],
                                sampling=dict(temperature=1.,top_p=1.,top_k=0,max_new_tokens=600),
                                native_dtype='bfloat16',molt_dtype='float32',adapter_enabled=True,
                                molt_sha256=asset['sha256'],molt_config_sha256=asset['config_sha256'],
                                traces=hook.traces,trace_summary=trace_summary(hook.traces),
                                hook_calls=hook.calls,mechanics_audit_passed=True)
                            write(destination,sample)
                            hook = None  # Release this layer before loading the next full MOLT.
                        summaries.append(dict(case_id=case['id'],feature=case['feature'],condition=condition,seed=seed,
                            file=str(destination.relative_to(out)),sha256=sha(destination),
                            tokens=len(sample['generated_token_ids']),stop_reason=sample['stop_reason'],
                            elapsed_seconds=sample['elapsed_seconds'],trace_summary=sample['trace_summary']))
                        write(out/'summary.json',dict(complete=False,done=len(summaries),expected=expected,samples=summaries))
                        progress('sampling',done=len(summaries),expected=expected,case_id=case['id'],condition=condition,seed=seed)
            del molt
            torch.cuda.empty_cache()
        if args.audit_only:
            progress('audit_complete',cases=len(cases),layers=layers)
        else:
            assert len(summaries)==expected
            write(out/'summary.json',dict(complete=True,done=len(summaries),expected=expected,samples=summaries))
            manifest_files = sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ('complete.json','status.json') and not p.name.endswith('.tmp'))
            write(out/'complete.json',dict(complete=True,expected_samples=expected,samples=len(summaries),
                protocol_sha256=runtime['protocol_sha256'],source_hashes=source_hashes,
                files={str(p.relative_to(out)):sha(p) for p in manifest_files}))
            progress('complete',done=len(summaries),expected=expected)
    except BaseException as error:
        progress('failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    with torch.inference_mode():
        main()
