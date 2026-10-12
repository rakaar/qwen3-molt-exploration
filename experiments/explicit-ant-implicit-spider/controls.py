"""Prespecified strength-32 replay, donor controls, and ten equal-norm directions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import run as experiment


@torch.inference_mode()
def main():
    r = json.loads((experiment.OUT / 'results.json').read_text())
    assert r['status'] == 'complete'
    p = experiment.P
    pair = next(x for x in p['pairs'] if x['id'] == p['primary_pair'])
    l, f, s = 3, 2309, p['random_controls']['strength']
    receiver = experiment.encode(pair['receiver'], pair['receiver_cue'])
    donor = experiment.encode(pair['donor'], pair['donor_cue'])
    clean, donor_caps = experiment.forward(receiver), experiment.forward(donor)
    a, b = experiment.parts(clean, l, f), experiment.parts(donor_caps, l, f)
    rp, dp = receiver['sites']['cue_end'], donor['sites']['cue_end']
    delta = s * (b['contributions'][dp] - a['contributions'][rp])
    norm = float(delta.norm())
    assert norm > 0 and rp == 12 and dp == 8
    rows, logits = [], []
    data = {'selection_rule': 'Fixed strength 32, layer 3, transform 2309, cue_end; prespecified before observing outcomes.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'random_seeds': p['random_controls']['seeds'], 'trials': rows}

    def evaluate(kind, vector, details):
        caps = experiment.forward(receiver, {'layer': l, 'position': rp, 'delta': vector})
        assert all(torch.equal(caps.residuals[k], clean.residuals[k]) for k in range(l))
        row = {'pair': pair['id'], 'kind': kind, 'layer': l, 'feature': f,
               'receiver_position': rp, 'strength': s, 'intended_delta_norm': float(vector.norm()),
               'logits_index': len(logits), **details, **experiment.summarize(caps.logits)}
        rows.append(row)
        logits.append(caps.logits.cpu().float().numpy())
        return row

    repeat = evaluate('repeat_full_contribution', delta, {'donor': pair['donor'], 'donor_position': dp})
    expected = next(t for t in r['trials'] if t['pair'] == pair['id'] and t['method'] == 'full_contribution' and t['strength'] == s)
    assert repeat['P6'] == expected['P6'] and repeat['P8'] == expected['P8']
    data['strength32_generation'] = experiment.generation(receiver, {'layer': l, 'position': rp, 'delta': delta})
    for seed in data['random_seeds']:
        rng = torch.Generator(device='cuda').manual_seed(seed)
        vector = torch.randn(2560, device='cuda', generator=rng)
        evaluate('norm_matched_gaussian', vector / vector.norm() * norm, {'seed': seed})
    alternatives = [{'id': x['id'], 'text': x['text'], 'cue': x['cue'], 'expected': x['expected']} for x in p['controls']]
    explicit = next(x for x in p['pairs'] if x['id'] == 'explicit_spider_ant')
    alternatives.append({'id': 'explicit_spider', 'text': explicit['receiver'], 'cue': 'spider', 'expected': '8'})
    for alt in alternatives:
        record = experiment.encode(alt['text'], alt['cue'])
        part = experiment.parts(experiment.forward(record), l, f)
        pos = record['sites']['cue_end']
        vector = s * (part['contributions'][pos] - a['contributions'][rp])
        evaluate('alternative_donor_same_strength', vector, {'donor_id': alt['id'], 'donor_text': alt['text'],
            'donor_expected': alt['expected'], 'donor_position': pos, 'donor_gate': float(part['gates'][pos])})
    np.savez_compressed(experiment.OUT / 'control-logits.npz', logits=np.stack(logits))
    experiment.save('random-control-results.json', data)
    manifest = json.loads((experiment.OUT / 'export-checksums.json').read_text())
    for name in ['random-control-results.json', 'control-logits.npz']:
        with (experiment.OUT / name).open('rb') as stream:
            manifest[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    experiment.save('export-checksums.json', manifest)
    print('CONTROLS_COMPLETE', len(rows), 'strength32 generation', data['strength32_generation'], flush=True)


if __name__ == '__main__':
    main()
