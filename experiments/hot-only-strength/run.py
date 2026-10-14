"""Fixed-source hot-only and independently scaled small/hot MOLT sweeps."""
import hashlib
import json
import platform
import time

import numpy as np
import torch

from common import Experiment, OUT, P, ROOT, save, sha, transformers


def bits(tensor):
    assert tensor.dtype == torch.bfloat16
    return tensor.contiguous().view(torch.uint16).cpu().numpy().copy()


@torch.inference_mode()
def main():
    started = time.monotonic()
    e = Experiment()
    receiver, donor = e.forward(), e.forward(donor=True)
    e.clean_reference = receiver
    rx, rg = e.gates(receiver)
    dx, dg = e.gates(donor)
    small, hot = e.parts(rx, rg, 2339), e.parts(dx, dg, 941)
    assert rg[2339] > 0 and dg[941] > 0 and rg[941] == 0 and dg[2339] == 0
    zero = torch.zeros_like(small)
    np.savez_compressed(OUT / 'contributions.npz', small=small.cpu().numpy(), hot=hot.cpu().numpy(),
                        receiver_input=rx.cpu().numpy(), donor_input=dx.cpu().numpy(),
                        receiver_gates=rg.cpu().numpy(), donor_gates=dg.cpu().numpy())
    R = {'protocol': P, 'status': 'running', 'rows': [], 'batch_clean_checks': [],
         'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
                     'model_dtype': 'BF16', 'molt_dtype': 'FP32', 'attention': 'unfrozen eager'},
         'source_sha256': {n: sha(ROOT / n) for n in ('run.py', 'common.py', 'protocol.json', 'download_base.py')},
         'receiver_tokens': e.receiver_tokens, 'donor_tokens': e.donor_tokens,
         'source_gates': {'small_receiver': float(rg[2339]), 'hot_donor': float(dg[941])},
         'source_norms': {'small': float(small.norm()), 'hot': float(hot.norm()),
                          'native_receiver_mlp': float(receiver.before[0, 9].float().norm())}}
    logits, intended, before, after = [], [], [], []
    unique = {}

    def vector(alpha, beta):
        # Match the prior pair's sum/write ordering exactly, including FP32 source math.
        return (zero + (-beta * small)) + alpha * hot

    def spec(alpha, beta, group='coarse', suffix=''):
        return {'id': f'{group}_a{alpha:g}_b{beta:g}{suffix}', 'group': group,
                'alpha': float(alpha), 'beta': float(beta), 'mode': 'add',
                'vector': vector(alpha, beta)}

    def append(row, logit, delta, pre, post):
        encoded = bits(logit)
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        if digest not in unique:
            unique[digest] = len(logits)
            logits.append(encoded)
        row.update(logit_row=unique[digest], logit_sha256=digest, vector_row=len(intended))
        row.update(e.summary(logit))
        row['intended_delta_norm'] = float(delta.norm())
        row['actual_delta_norm'] = float((post.float() - pre.float()).norm())
        row['actual_delta_over_native_mlp_norm'] = row['actual_delta_norm'] / R['source_norms']['native_receiver_mlp']
        intended.append(delta.cpu().numpy())
        before.append(pre.float().cpu().numpy())
        after.append(post.float().cpu().numpy())
        R['rows'].append(row)
        return row

    append({'id': 'clean_receiver', 'group': 'baseline', 'alpha': 0., 'beta': 0.},
           receiver.logits[0], zero, receiver.before[0, 9], receiver.before[0, 9])
    append({'id': 'clean_donor', 'group': 'baseline_donor', 'alpha': 0., 'beta': 0.},
           donor.logits[0], zero, donor.before[0, 9], donor.before[0, 9])
    save('baseline.json', {'receiver': R['rows'][0], 'donor': R['rows'][1]})

    def execute(specs):
        added = []
        for offset in range(0, len(specs), P['batch_size'] - 1):
            assert time.monotonic() - started < P['runtime_limit_minutes'] * 60
            batch = specs[offset:offset + P['batch_size'] - 1]
            caps = e.forward(edits=batch)
            for i in range(len(batch), P['batch_size']):
                assert torch.equal(caps.logits[i], receiver.logits[i])
                assert torch.equal(caps.after[i], receiver.before[i])
            R['batch_clean_checks'].append({'conditions': [s['id'] for s in batch],
                                           'clean_rows': P['batch_size'] - len(batch), 'bitwise_equal': True})
            for i, s in enumerate(batch):
                row = {k: v for k, v in s.items() if k not in ('vector', 'mode')}
                added.append(append(row, caps.logits[i], s['vector'], caps.before[i, 9], caps.after[i, 9]))
                print('ROW', row['id'], 'P(cold)', row['P_cold'], 'top', repr(row['top_token']), flush=True)
            save('results.json', R)
        return added

    coarse = [spec(a, b) for b in P['small_subtractions'] for a in P['hot_strengths']]
    coarse += [spec(0, b, 'small_only') for b in P['small_only_strengths']]
    save('conditions-before-execution.json', [{k: v for k, v in s.items() if k != 'vector'} for s in coarse])
    execute(coarse)
    brackets = []
    for beta in P['small_subtractions']:
        rows = sorted((r for r in R['rows'] if r['group'] == 'coarse' and r['beta'] == beta), key=lambda r: r['alpha'])
        crossing = next((i for i, r in enumerate(rows) if r['P_cold'] >= .99), None)
        if crossing is None:
            brackets.append({'beta': beta, 'reached_99': False, 'largest_tested_alpha': rows[-1]['alpha']})
            continue
        if crossing == 0:
            brackets.append({'beta': beta, 'reached_99': True, 'lower_alpha': 0, 'upper_alpha': 0})
            continue
        lo, hi = rows[crossing - 1], rows[crossing]
        coarse_bracket = [lo['alpha'], hi['alpha']]
        for step in range(8):
            mid = execute([spec((lo['alpha'] + hi['alpha']) / 2, beta, 'refinement', f'_step{step}')])[0]
            if mid['P_cold'] >= .99:
                hi = mid
            else:
                lo = mid
        brackets.append({'beta': beta, 'reached_99': True, 'coarse_bracket': coarse_bracket,
                         'lower_alpha': lo['alpha'], 'lower_P_cold': lo['P_cold'],
                         'upper_alpha': hi['alpha'], 'upper_P_cold': hi['P_cold'],
                         'interpretation': 'First observed coarse-grid crossing, locally refined; not a proven global minimum.'})
    R['threshold_brackets'] = brackets

    by_id = {r['id']: r for r in R['rows']}
    originals = [by_id['coarse_a4.5_b4.5']]
    for beta in (0, .5):
        originals.append(max((r for r in R['rows'] if r['beta'] == beta and r['group'] == 'coarse'), key=lambda r: r['P_cold']))
    repeated = execute([spec(r['alpha'], r['beta'], 'repeat', f'_{i}') for i, r in enumerate(originals)])
    for original, repeat in zip(originals, repeated):
        assert original['logit_sha256'] == repeat['logit_sha256']
    execute([spec(0, 0, 'restoration')])
    assert R['rows'][-1]['logit_sha256'] == R['rows'][0]['logit_sha256']

    # Verify attention is truly recomputed: only downstream layers can change.
    base_attn = e.forward(attention=True)
    selected_attn = e.forward(edits=[spec(4.5, 4.5)] * P['batch_size'], attention=True)
    changes = [{'layer': i, 'max_absolute_change': float((a[0].float() - b[0].float()).abs().max())}
               for i, (a, b) in enumerate(zip(base_attn.attention, selected_attn.attention))]
    assert all(r['max_absolute_change'] == 0 for r in changes[:23])
    assert any(r['max_absolute_change'] > 0 for r in changes[23:])
    R['attention_changes_selected_pair'] = changes

    # Full-prefix greedy continuations. The ONLY directly edited position is prompt token9.
    cases = [by_id['clean_receiver'], by_id['coarse_a4.5_b4.5'],
             by_id['coarse_a4.5_b0'], by_id['coarse_a20_b0'], by_id['coarse_a512_b0']]
    for beta in (0, .5):
        candidates = [r for r in R['rows'] if r['beta'] == beta and r['group'] in ('coarse', 'refinement')]
        cases.append(max(candidates, key=lambda r: r['P_cold']))
        hits = [r for r in candidates if r['P_cold'] >= .99]
        if hits:
            cases.append(min(hits, key=lambda r: r['alpha']))
    eos = e.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generations, seen = [], set()
    for case in cases:
        key = (case['alpha'], case['beta'])
        if key in seen:
            continue
        seen.add(key)
        ids = e.receiver.repeat(P['batch_size'], 1)
        delta = vector(*key)
        produced, ended = [], False
        first_logits = None

        def hook(_module, _args, output):
            result = output.clone()
            result[:, 9] = output[:, 9] + delta.to(output.dtype)
            assert torch.equal(result[:, :9], output[:, :9])
            assert torch.equal(result[:, 10:], output[:, 10:])
            return result

        handle = e.model.model.layers[22].mlp.register_forward_hook(hook)
        try:
            for step in range(24):
                output = e.model(ids, use_cache=False)
                next_logits = output.logits[0, -1]
                if step == 0:
                    assert hashlib.sha256(bits(next_logits).tobytes()).hexdigest() == case['logit_sha256']
                    first_logits = e.summary(next_logits)
                next_id = int(next_logits.argmax())
                produced.append(next_id)
                if next_id in eos:
                    ended = True
                    break
                ids = torch.cat([ids, torch.full((P['batch_size'], 1), next_id, device='cuda', dtype=torch.long)], dim=1)
        finally:
            handle.remove()
        generations.append({'source_condition': case['id'], 'alpha': key[0], 'beta': key[1],
                            'token_ids': produced, 'text': e.tokenizer.decode(produced, skip_special_tokens=True),
                            'ended_with_eos': ended, 'token_limit': 24,
                            'first_logits_match_sweep_bitwise': True, 'first_token': first_logits})
        print('GENERATION', key, repr(generations[-1]['text']), 'EOS', ended, flush=True)
    save('generations.json', generations)
    np.savez_compressed(OUT / 'next-token-logits.npz', logits_bf16_bits=np.stack(logits))
    np.savez_compressed(OUT / 'direct-write.npz', intended_delta=np.stack(intended), before=np.stack(before), after=np.stack(after))
    R.update(status='complete', seconds=time.monotonic() - started,
             peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
             checks={'fixed_clean_source_vectors': True, 'unchanged_upstream_input_every_condition': True,
                     'native_mlp_retained': True, 'only_layer22_prompt_token9_directly_edited': True,
                     'repeat_and_restoration_bitwise_equal': True, 'attention_unfrozen_verified': True,
                     'full_vocabulary_bf16_logits_lossless': True, 'generation_initial_logits_match': True})
    save('results.json', R)
    save('export-checksums.json', {str(path.relative_to(ROOT)): sha(path) for path in sorted(OUT.iterdir())
                                  if path.is_file() and path.name != 'export-checksums.json'})
    print('COMPLETE', len(R['rows']), 'conditions', flush=True)


if __name__ == '__main__':
    main()
