"""Matched-strength and norm controls for the primary semantic clue-token result."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import run as experiment

ROOT, OUT = experiment.ROOT, experiment.OUT


@torch.inference_mode()
def main():
    r = json.loads((OUT/'results.json').read_text())
    assert r['status']=='complete'
    candidates = [x for x in r['trials'] if x['pair']=='web_anthill' and x['site']=='cue_end' and x['method']=='full_contribution' and not x['role'].startswith('lexical')]
    chosen=max(candidates,key=lambda x:x['P6'])
    l,f,s=chosen['layer'],chosen['feature'],chosen['strength']
    assert s>0
    pairs={p['id']:p for p in r['protocol']['pairs']}
    control_prompts={x['id']:x for x in r['protocol']['controls']}
    rows,logits=[],[]
    data={'primary_selected_trial':chosen,'selection_rule':'Highest primary clue-token P6 among semantic candidates; controls selected after initial sweep and reported separately.',
          'random_seeds':list(range(20)),'trials':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for pair_id in ['web_anthill','silk_colony']:
        pair=pairs[pair_id]
        receiver=experiment.encode(pair['receiver'],pair['receiver_cue'])
        donor=experiment.encode(pair['donor'],pair['donor_cue'])
        clean=experiment.forward(receiver)
        original_donor=experiment.forward(donor)
        a=experiment.parts(clean,l,f);b=experiment.parts(original_donor,l,f)
        rp,dp=receiver['sites']['cue_end'],donor['sites']['cue_end']
        original_delta=s*(b['contributions'][dp]-a['contributions'][rp])
        norms=float(original_delta.norm())

        def evaluate(kind,delta,details):
            caps=experiment.forward(receiver,{'layer':l,'position':rp,'delta':delta})
            assert all(torch.equal(caps.residuals[k],clean.residuals[k]) for k in range(l))
            row={'pair':pair_id,'kind':kind,'layer':l,'feature':f,'receiver_position':rp,'strength':s,
                 'intended_delta_norm':float(delta.norm()),'logits_index':len(logits),**details,**experiment.summarize(caps.logits)}
            rows.append(row);logits.append(caps.logits.cpu().float().numpy())
            return row

        repeat=evaluate('repeat_full_contribution',original_delta,{'donor':pair['donor']})
        assert repeat['P6']==next(x['P6'] for x in r['trials'] if x['pair']==pair_id and x['layer']==l and x['feature']==f and x['site']=='cue_end' and x['method']=='full_contribution' and x['strength']==s)
        gate_delta=s*(b['gates'][dp]-a['gates'][rp])*a['lowrank'][rp]
        gate=evaluate('matched_strength_gate_only',gate_delta,{'receiver_gate':float(a['gates'][rp]),'donor_gate':float(b['gates'][dp])})
        evaluate('matched_strength_ablation',-s*a['contributions'][rp],{})
        for seed in data['random_seeds']:
            control_rng=torch.Generator(device='cuda').manual_seed(seed)
            random=torch.randn(2560,device='cuda',generator=control_rng)
            delta=random/random.norm()*norms
            evaluate('norm_matched_gaussian',delta,{'seed':seed})
        for name in ['bee','cow','scorpion','antique','antioxidant']:
            prompt=control_prompts[name]
            alternative=experiment.encode(prompt['text'],prompt['cue'])
            caps=experiment.forward(alternative)
            part=experiment.parts(caps,l,f)
            pos=alternative['sites']['cue_end']
            delta=s*(part['contributions'][pos]-a['contributions'][rp])
            evaluate('alternative_donor_same_strength',delta,{'donor_id':name,'donor_text':prompt['text'],'donor_expected':prompt['expected'],'donor_gate':float(part['gates'][pos])})
        for side in ['receiver','donor']:
            explicit=pairs['explicit_spider_ant']
            alt=experiment.encode(explicit[side],explicit[side+'_cue'])
            caps=experiment.forward(alt);part=experiment.parts(caps,l,f);pos=alt['sites']['cue_end']
            delta=s*(part['contributions'][pos]-a['contributions'][rp])
            evaluate('alternative_donor_same_strength',delta,{'donor_id':'explicit_'+side,'donor_text':explicit[side],'donor_expected':'8' if side=='receiver' else '6','donor_gate':float(part['gates'][pos])})
        print('CONTROLS',pair_id,'repeat P6',repeat['P6'],'matched gate-only P6',gate['P6'],flush=True)
    np.savez_compressed(OUT/'control-logits.npz',logits=np.stack(logits))
    experiment.save('random-control-results.json',data)
    manifest=json.loads((OUT/'export-checksums.json').read_text())
    for name in ['random-control-results.json','control-logits.npz']:
        with (OUT/name).open('rb') as stream:
            manifest[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    experiment.save('export-checksums.json',manifest)
    print('CONTROLS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    main()
