"""Uniform and active random MOLT controls plus exact native MLP-output replacement."""
import json
import hashlib
import platform
import time

import numpy as np
import torch
from safetensors.torch import load_file

from common import Experiment, OUT, P, ROOT, save, sha, transformers


def logit_bits(tensor):
    assert tensor.dtype == torch.bfloat16
    return tensor.contiguous().view(torch.uint16).cpu().numpy().copy()


def decode_bits(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


@torch.inference_mode()
def main():
    start = time.monotonic()
    prepared = json.loads((OUT / 'preparation-completion.json').read_text())
    assert prepared['status'] == 'complete'
    assert prepared['selection_sha256'] == sha(OUT / 'selection.json')
    assert prepared['clean_state_sha256'] == sha(OUT / 'clean-state.pt')
    selection = json.loads((OUT / 'selection.json').read_text())
    provenance = json.loads((OUT / 'slice-provenance.json').read_text())
    assert sha(ROOT / 'weights/control-uv.safetensors') == provenance['file_sha256']
    e = Experiment()
    receiver, donor = e.forward(), e.forward(donor=True)
    prior_state = torch.load(OUT / 'clean-state.pt', map_location='cpu', weights_only=True)
    for name, caps in [('receiver', receiver), ('donor', donor)]:
        for field, value in [('prenorm', caps.prenorm), ('mlp_input', caps.mlp_input),
                             ('mlp_output', caps.before), ('logits', caps.logits)]:
            assert torch.equal(value[0].cpu(), prior_state[name + '_' + field])
    e.clean_reference = receiver
    rec_x, rec_gate = e.gates(receiver)
    don_x, don_gate = e.gates(donor)
    assert rec_gate.cpu().tolist() == selection['all_receiver_gates']
    assert don_gate.cpu().tolist() == selection['all_donor_gates']
    uv = {k: v.cuda() for k, v in load_file(ROOT / 'weights/control-uv.safetensors').items()}
    ids = selection['needed_ids']
    rec = {f: e.parts(rec_x, rec_gate, f, uv) for f in ids}
    don = {f: e.parts(don_x, don_gate, f, uv) for f in ids}
    for f in (2339, 941):
        assert torch.equal(rec[f], e.parts(rec_x, rec_gate, f))
        assert torch.equal(don[f], e.parts(don_x, don_gate, f))
    np.savez_compressed(OUT / 'contributions.npz', ids=np.array(ids),
        receiver=np.stack([rec[f].cpu().numpy() for f in ids]),
        donor=np.stack([don[f].cpu().numpy() for f in ids]),
        receiver_gates=rec_gate.cpu().numpy(), donor_gates=don_gate.cpu().numpy(),
        receiver_standardized_input=rec_x.cpu().numpy(), donor_standardized_input=don_x.cpu().numpy())
    del uv
    specs = []
    strength = P['strength']

    def add(name, group, terms, sample_index=None, multiplier=1):
        # Terms identify the feature, clean source prompt, and signed coefficient.
        vector = sum((coefficient * (rec if source == 'receiver' else don)[f]
                      for source, f, coefficient in terms),
                     torch.zeros(2560, device='cuda', dtype=torch.float32))
        specs.append({'id': name, 'group': group, 'terms': terms, 'mode': 'add',
                      'vector': vector, 'sample_index': sample_index, 'multiplier': multiplier})

    add('selected_pair', 'selected_pair', [('receiver', 2339, -strength), ('donor', 941, strength)])
    add('selected_hot_only', 'selected_single_positive', [('donor', 941, strength)])
    add('selected_small_only', 'selected_single_negative', [('receiver', 2339, -strength)])
    add('selected_pair_reverse', 'selected_pair_reverse', [('receiver', 2339, strength), ('donor', 941, -strength)], multiplier=-1)
    specs.append({'id': 'whole_mlp_donor_replace', 'group': 'whole_mlp', 'terms': [], 'mode': 'replace',
                  'vector': donor.before[0, 9], 'sample_index': None, 'multiplier': 1})
    specs.append({'id': 'self_mlp_replace', 'group': 'identity', 'terms': [], 'mode': 'replace',
                  'vector': receiver.before[0, 9], 'sample_index': None, 'multiplier': 1})
    add('zero_delta', 'identity', [])
    for index, f in enumerate(selection['uniform_ids']):
        add(f'uniform_positive_{f}', 'uniform_single_positive', [('donor', f, strength)], index)
        add(f'uniform_negative_{f}', 'uniform_single_negative', [('receiver', f, -strength)], index)
    for index, (a, b) in enumerate(selection['uniform_pairs']):
        add(f'uniform_pair_{a}_{b}', 'uniform_pair', [('receiver', a, -strength), ('donor', b, strength)], index)
        add(f'uniform_pair_reverse_{a}_{b}', 'uniform_pair_reverse', [('receiver', a, strength), ('donor', b, -strength)], index, -1)
    for index, f in enumerate(selection['donor_active_ids']):
        add(f'active_positive_{f}', 'active_single_positive', [('donor', f, strength)], index)
    for index, f in enumerate(selection['receiver_active_ids']):
        add(f'active_negative_{f}', 'active_single_negative', [('receiver', f, -strength)], index)
    for index, (a, b) in enumerate(selection['active_pairs']):
        add(f'active_pair_{a}_{b}', 'active_pair', [('receiver', a, -strength), ('donor', b, strength)], index)
    add('selected_pair_repeat', 'repeat', [('receiver', 2339, -strength), ('donor', 941, strength)])
    add('restored_receiver', 'identity', [])
    metadata = [{k: v for k, v in spec.items() if k != 'vector'} for spec in specs]
    save('conditions-before-execution.json', metadata)

    R = {'protocol': P, 'status': 'running', 'selection': selection,
         'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
                     'model_dtype': 'BF16', 'molt_dtype': 'FP32', 'attention': 'unfrozen eager',
                     'batch_size': P['batch_size']},
         'rows': [], 'checks': {}, 'source_sha256': {n: sha(ROOT / n) for n in ('run.py', 'common.py', 'prepare.py')},
         'selected_asset_sha256': P['selected_asset_sha256'], 'original_mlp_norm': float(receiver.before[0, 9].float().norm()),
         'prior_reference': json.loads((ROOT / 'prior-reference.json').read_text()), 'batch_clean_checks': []}
    logits, intended, before, after = [], [], [], []
    logit_indices = {}

    def append(row, logit, delta, pre, post):
        encoded = logit_bits(logit)
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        if digest not in logit_indices:
            logit_indices[digest] = len(logits)
            logits.append(encoded)
        row['logit_row'] = logit_indices[digest]
        assert np.array_equal(encoded, logits[row['logit_row']])
        row['logit_sha256'] = digest
        row['vector_row'] = len(intended)
        row.update(e.summary(logit))
        row['intended_delta_norm'] = float(delta.float().norm())
        row['actual_delta_norm'] = float((post.float() - pre.float()).norm())
        row['actual_delta_over_original_mlp_norm'] = row['actual_delta_norm'] / R['original_mlp_norm']
        row['nonzero_edit'] = row['actual_delta_norm'] > 0
        intended.append(delta.float().cpu().numpy())
        before.append(pre.float().cpu().numpy()); after.append(post.float().cpu().numpy())
        R['rows'].append(row)

    zero = torch.zeros(2560, device='cuda')
    append({'id': 'clean_receiver', 'group': 'baseline', 'terms': [], 'mode': 'none'}, receiver.logits[0], zero, receiver.before[0, 9], receiver.before[0, 9])
    append({'id': 'clean_donor', 'group': 'baseline_donor', 'terms': [], 'mode': 'none'}, donor.logits[0], zero, donor.before[0, 9], donor.before[0, 9])
    size = P['batch_size']
    for start_index in range(0, len(specs), size - 1):
        assert time.monotonic() - start < P['runtime_limit_minutes'] * 60
        group = specs[start_index:start_index + size - 1]
        padding = [{'mode': 'add', 'vector': zero}] * (size - len(group))
        caps = e.forward(edits=group + padding)
        # A clean row accompanies every batch, including the final partially filled batch.
        for index in range(len(group), size):
            assert torch.equal(caps.logits[index], receiver.logits[index])
            assert torch.equal(caps.after[index], receiver.before[index])
        R['batch_clean_checks'].append({'start_index': start_index, 'clean_rows': size - len(group),
                                       'bitwise_equal': True})
        for index, spec in enumerate(group):
            row = {k: v for k, v in spec.items() if k != 'vector'}
            row['source_gates'] = [{'source': source, 'feature': f,
                                   'gate': float((rec_gate if source == 'receiver' else don_gate)[f])}
                                  for source, f, coefficient in spec['terms']]
            delta = spec['vector'].float() if spec['mode'] == 'add' else spec['vector'].float() - caps.before[index, 9].float()
            append(row, caps.logits[index], delta, caps.before[index, 9], caps.after[index, 9])
            if row['id'] in ('selected_pair', 'whole_mlp_donor_replace'):
                print('REFERENCE', row['id'], 'P_cold', R['rows'][-1]['P_cold'],
                      'P_large', R['rows'][-1]['P_large'], 'top', R['rows'][-1]['top_token'], flush=True)
            if not R['rows'][-1]['nonzero_edit']:
                assert torch.equal(caps.logits[index], receiver.logits[index]), 'Inactive edit changed logits'
        save('results.json', R)
        print('BATCH', start_index // (size - 1) + 1, 'completed', min(start_index + len(group), len(specs)),
              'of', len(specs), flush=True)
    np.savez_compressed(OUT / 'next-token-logits.npz', logits_bf16_bits=np.stack(logits))
    np.savez_compressed(OUT / 'direct-write.npz', intended_delta=np.stack(intended),
                        before=np.stack(before), after=np.stack(after))
    by_id = {row['id']: row for row in R['rows']}
    selected, repeat = by_id['selected_pair'], by_id['selected_pair_repeat']
    assert np.array_equal(logits[selected['logit_row']], logits[repeat['logit_row']])
    for name in ('self_mlp_replace', 'zero_delta', 'restored_receiver'):
        assert np.array_equal(logits[by_id[name]['logit_row']], logits[0])
    whole = by_id['whole_mlp_donor_replace']
    assert np.array_equal(after[whole['vector_row']], donor.before[0, 9].float().cpu().numpy())
    native_attention = e.forward(attention=True)
    pair_spec = specs[0]
    changed_attention = e.forward(edits=[pair_spec] * size, attention=True)
    assert torch.equal(changed_attention.logits[0].cpu().float(), torch.from_numpy(decode_bits(logits[selected['logit_row']])))
    changes = [{'layer': i, 'max_absolute_change': float((a[0].float() - b[0].float()).abs().max())}
               for i, (a, b) in enumerate(zip(native_attention.attention, changed_attention.attention))]
    assert all(row['max_absolute_change'] == 0 for row in changes[:23])
    assert any(row['max_absolute_change'] > 0 for row in changes[23:])
    R['attention_changes_selected'] = changes
    R['checks'].update({'clean_state_matches_preparation_bitwise': True,
                       'unchanged_upstream_prenorm_and_mlp_inputs_every_condition': True,
                       'only_layer22_token9_directly_edited': True,
                       'exact_donor_mlp_output_replacement_verified': True,
                       'all_inactive_edits_exactly_recover_clean_logits': True,
                       'self_zero_restoration_bitwise_equal': True,
                       'selected_repeat_bitwise_equal': True,
                       'all_attention_methods_unchanged': True,
                       'attention_changes_only_downstream_of_layer22': True,
                       'lossless_full_vocabulary_logits_stored': True})
    old = R['prior_reference']['conditions'][0]
    R['prior_selected_comparison'] = {'prior_P_cold': old['P_cold'], 'current_P_cold': selected['P_cold'],
                                     'absolute_difference': abs(old['P_cold'] - selected['P_cold']),
                                     'note': 'CUDA build and batch shape differ from the October10 run; all current controls use the same eight-row batch shape.'}
    R.update({'status': 'complete', 'seconds': time.monotonic() - start,
              'logit_storage': 'Deduplicated full-vocabulary BF16 bit patterns stored as uint16; exactly lossless.',
              'unique_logit_rows': len(logits),
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
    save('results.json', R)
    save('export-checksums.json', {str(path.relative_to(ROOT)): sha(path)
                                  for path in sorted(OUT.iterdir()) if path.is_file() and path.name != 'export-checksums.json'})
    print('COMPLETE', len(R['rows']), 'conditions', flush=True)


if __name__ == '__main__':
    main()
