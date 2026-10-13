"""Select controls from clean gates, then fetch only required public U/V slices."""
import concurrent.futures
import json
import platform
import random
import struct
import time

import requests
import torch
from safetensors.torch import save_file

from common import Experiment, OUT, P, ROOT, save, sha, transformers


def fetch(url, start, end):
    for attempt in range(4):
        try:
            with requests.get(url + f'?control_range={start}-{end}',
                              headers={'Range': f'bytes={start}-{end}'}, timeout=60, stream=True) as r:
                r.raise_for_status()
                assert r.status_code == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                blob = r.raw.read(end - start + 2)
                assert len(blob) == end - start + 1
                return blob
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


if __name__ == '__main__':
    start = time.monotonic()
    e = Experiment()
    receiver = e.forward(attention=False)
    donor = e.forward(donor=True, attention=False)
    for c in (receiver, donor):
        assert all(torch.equal(c.logits[0], l) for l in c.logits)
    rec_x, rec_gate = e.gates(receiver)
    don_x, don_gate = e.gates(donor)
    assert e.summary(receiver.logits[0])['top_token'] == 'large'
    assert e.summary(donor.logits[0])['top_token'] == 'cold'
    assert float(rec_gate[941]) == float(don_gate[2339]) == 0
    assert rec_gate[2339] > 0 and don_gate[941] > 0
    target_ids = {2339, 941}
    pool = [i for i in range(2480) if i not in target_ids]
    rng = random.Random(P['random_seed'])
    uniform = rng.sample(pool, P['random_count'])
    permutation = uniform.copy()
    while True:
        rng.shuffle(permutation)
        if all(a != b for a, b in zip(uniform, permutation)):
            break
    rec_active = [i for i in pool if rec_gate[i] > 0]
    don_active = [i for i in pool if don_gate[i] > 0]
    rec_sample = rng.sample(rec_active, min(100, len(rec_active)))
    don_sample = rng.sample(don_active, min(100, len(don_active)))
    pair_pool = [(a, b) for a in rec_active for b in don_active if a != b]
    pairs = rng.sample(pair_pool, min(100, len(pair_pool)))
    needed = set(uniform + rec_sample + don_sample + [i for pair in pairs for i in pair]) | target_ids
    uv_ids = sorted(i for i in needed if rec_gate[i] > 0 or don_gate[i] > 0)
    selection = {'seed': P['random_seed'], 'uniform_ids': uniform,
                 'uniform_pairs': list(map(list, zip(uniform, permutation))),
                 'receiver_active_pool': rec_active, 'donor_active_pool': don_active,
                 'receiver_active_ids': rec_sample, 'donor_active_ids': don_sample,
                 'active_pairs': pairs, 'needed_ids': sorted(needed), 'uv_ids': uv_ids,
                 'all_receiver_gates': rec_gate.cpu().tolist(), 'all_donor_gates': don_gate.cpu().tolist(),
                 'selection_basis': 'Fixed seed and clean gates only; no intervention outcomes seen.',
                 'saved_before_interventions': True}
    save('selection.json', selection)
    baseline = {'receiver': e.summary(receiver.logits[0]), 'donor': e.summary(donor.logits[0]),
                'receiver_tokens': e.receiver_tokens, 'donor_tokens': e.donor_tokens,
                'original_mlp_norm': float(receiver.before[0, 9].float().norm()),
                'whole_mlp_delta_norm': float((donor.before[0, 9].float() - receiver.before[0, 9].float()).norm()),
                'batch_size': P['batch_size'], 'all_identical_clean_rows_bitwise_equal': True,
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                            'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name()}}
    save('baseline.json', baseline)
    torch.save({'receiver_prenorm': receiver.prenorm[0].cpu(), 'donor_prenorm': donor.prenorm[0].cpu(),
                'receiver_mlp_input': receiver.mlp_input[0].cpu(), 'donor_mlp_input': donor.mlp_input[0].cpu(),
                'receiver_mlp_output': receiver.before[0].cpu(), 'donor_mlp_output': donor.before[0].cpu(),
                'receiver_logits': receiver.logits[0].cpu(), 'donor_logits': donor.logits[0].cpu()},
               OUT / 'clean-state.pt')
    print('CLEAN_READY', json.dumps(baseline['receiver']), flush=True)
    print('CONTROL_SELECTION', json.dumps({'uniform': len(uniform), 'receiver_active': len(rec_active),
          'donor_active': len(don_active), 'active_pairs': len(pairs), 'uv_slices_needed': len(uv_ids)}), flush=True)

    prefix = f"https://huggingface.co/{P['molt_repo']}/resolve/{P['molt_revision']}/layer-22"
    config = requests.get(prefix + '/config.json', timeout=40); config.raise_for_status()
    cfg = config.json()
    assert cfg['layer'] == 22 and cfg['input_location'] == 'pre_norm' and cfg['output_location'] == 'raw'
    save('verified-layer22-config.json', cfg)
    url = prefix + '/checkpoint.safetensors'
    first = fetch(url, 0, 65535)
    length = struct.unpack('<Q', first[:8])[0]
    assert length < 65528
    header = json.loads(first[8:8+length])
    assert header == json.loads((ROOT / 'weights/molt-22-header.json').read_text())
    groups = [(0, 80, 512), (80, 240, 256), (240, 560, 128), (560, 1200, 64), (1200, 2480, 32)]

    def download_feature(feature):
        group = next(j for j, (lo, hi, rank) in enumerate(groups) if lo <= feature < hi)
        offset = feature - groups[group][0]
        tensors, records = {}, []
        for role in ('U', 'V'):
            key = f'{role}s.{group}'
            spec = header[key]
            begin, end = [8 + length + i for i in spec['data_offsets']]
            size = (end - begin) // spec['shape'][0]
            begin += size * offset
            end = begin + size
            blob = fetch(url, begin, end - 1)
            assert spec['dtype'] == 'F32'
            tensor = torch.frombuffer(bytearray(blob), dtype=torch.float32).reshape(spec['shape'][1:]).clone()
            tensors[f'{role}_{feature}'] = tensor
            if feature in target_ids:
                assert torch.equal(tensor, e.w[f'{role}_{feature}'].cpu()), 'Original target slice differs'
            import hashlib
            records.append({'feature': feature, 'source_tensor': key, 'index': offset,
                            'rank': groups[group][2], 'range': [begin, end - 1],
                            'shape': list(tensor.shape), 'bytes': len(blob),
                            'sha256': hashlib.sha256(blob).hexdigest()})
        print('VERIFIED_SLICES', feature, groups[group][2], flush=True)
        return tensors, records

    tensors, records = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for weights, rows in pool.map(download_feature, uv_ids):
            tensors.update(weights); records.extend(rows)
    target = ROOT / 'weights/control-uv.safetensors'
    save_file(tensors, target)
    save('slice-provenance.json', {'repo': P['molt_repo'], 'revision': P['molt_revision'],
                                  'tensor_records': records, 'file': target.name,
                                  'file_sha256': sha(target), 'bytes': target.stat().st_size,
                                  'original_target_slices_bitwise_equal': True,
                                  'inactive_both_prompts': 'No U/V slice required: gate times any output is zero.'})
    save('preparation-completion.json', {'status': 'complete', 'seconds': time.monotonic() - start,
                                       'source_sha256': {n: sha(ROOT / n) for n in ('common.py', 'prepare.py')},
                                       'selected_asset_sha256': sha(ROOT / 'weights/molt-22-selected.safetensors'),
                                       'selection_sha256': sha(OUT / 'selection.json'),
                                       'clean_state_sha256': sha(OUT / 'clean-state.pt')})
    print('PREPARATION_COMPLETE', target.stat().st_size, flush=True)
