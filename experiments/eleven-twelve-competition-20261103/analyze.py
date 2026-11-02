"""Saved-logit diagnostic: output competition versus an internal ordinal shift."""
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'comparisons/2026-11-02-eleven-is-off/remote-export/eleven-is-off-20261102/artifacts/results.json'
OUT=ROOT/'results/2026-11-03-eleven-twelve-competition'
OUT.mkdir(exist_ok=True)
d=json.loads(SOURCE.read_text());assert d['status']=='complete' and d['hooks_remaining']==0
assert d['prompt']=='the eleventh month of the year is'
native=d['native'];off=d['gate_off']
assert off['spec']['positions']==[7]
assert native['output_token_ids'][0]==84223 and off['output_token_ids'][0]==89011
states={}
for label,run in [('native',native),('gate_off',off)]:
    step=run['steps'][0];p=step['probe_probabilities']
    assert abs(math.log(p[str(step['token_id'])])+step['logsumexp']-step['selected_logit'])<1e-5
    p11=sum(p[str(i)] for i in [84223,6702])
    p12=sum(p[str(i)] for i in [89011,6652])
    states[label]={'p_november_variants':p11,'p_december_variants':p12,
                   'p_december_conditioned_on_not_two_november_variants':p12/(1-p11),
                   'lowercase_november_logit':math.log(p['84223'])+step['logsumexp'],
                   'lowercase_december_logit':math.log(p['89011'])+step['logsumexp']}
a,b=states['native'],states['gate_off']
counterfactual=a['p_december_variants']/(1-a['p_november_variants'])*(1-b['p_november_variants'])
other_scale=(1-b['p_november_variants'])/(1-a['p_november_variants'])
counterfactual_lowercase=d['native']['steps'][0]['probe_probabilities']['89011']*other_scale
counterfactual_titlecase=d['native']['steps'][0]['probe_probabilities']['6652']*other_scale
counterfactual_other_mass=(1-a['p_november_variants']-a['p_december_variants'])*other_scale
assert a['p_november_variants']+a['p_december_variants']<1
assert a['p_december_variants']>1-a['p_november_variants']-a['p_december_variants']
assert counterfactual>1-b['p_november_variants']-counterfactual and counterfactual>b['p_november_variants']
assert counterfactual_lowercase>max(counterfactual_titlecase,counterfactual_other_mass,b['p_november_variants'])
out={'prompt':d['prompt'],'feature':'L29-T2375','edited_input_position':7,'edited_token':' is',
     'scope':'Only final input position; no generated positions edited.',
     'answer_events':{'November':[' november',' November'],'December':[' december',' December']},
     'states':states,'output_only_counterfactual':{
         'definition':'Change only the combined probability mass of the two leading-space November tokens to the observed gate-off total; preserve relative probabilities of every other token.',
         'p_december_variants':counterfactual,
         'p_lowercase_december_token':counterfactual_lowercase,
         'p_titlecase_december_token':counterfactual_titlecase,
         'actual_december_probability_minus_counterfactual':b['p_december_variants']-counterfactual,
         'december_wins_even_under_this_counterfactual':True},
     'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
     'source_path':str(SOURCE.relative_to(ROOT)),
     'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'new_inference':False,'limits':[
         'One exact raw prefix. Events sum lowercase/title-case leading-space whole-word tokens, excluding other renderings and fragments.',
         'Counterfactual is an analytic output-distribution control, not an internal-network intervention.',
         'Raw logit changes include changed final normalization and downstream computation; they do not identify the causal path.',
         'A greedy twelve answer can arise by weakening eleven; the flip alone does not establish an internal ordinal increment.']}
(OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Why eleven-off produces twelve: saved output competition','',
       'Prompt: `the eleventh month of the year is`. L29–T2375 removed only at final input ` is`, position 7. No new inference.',
       '','| Answer event | Native | Gate off |','|---|---:|---:|',
       f'| November variants | {100*a["p_november_variants"]:.4f}% | {100*b["p_november_variants"]:.4f}% |',
       f'| December variants | {100*a["p_december_variants"]:.4f}% | {100*b["p_december_variants"]:.4f}% |',
       '',f'If only the two November tokens lose combined probability mass to their observed gate-off level, with all other relative preferences fixed, December reaches {100*counterfactual:.4f}%. It already becomes the greedy answer under this analytic control. Actual ablation gives {100*b["p_december_variants"]:.4f}%, a further {100*(b["p_december_variants"]-counterfactual):.4f} percentage points.',
       '',f'December as a fraction of probability outside those two November tokens rises from {100*a["p_december_conditioned_on_not_two_november_variants"]:.4f}% to {100*b["p_december_conditioned_on_not_two_november_variants"]:.4f}%. Thus the actual edit changes relative preferences among the remaining tokens as well.',
       '',f'Lowercase November raw logit: {a["lowercase_november_logit"]:.6f} to {b["lowercase_november_logit"]:.6f}. Lowercase December: {a["lowercase_december_logit"]:.6f} to {b["lowercase_december_logit"]:.6f}. These include downstream computation and changed final normalization.',
       '','This supports separating two questions: why December is the fallback competitor, and why ablation further favors it. It does not show an internal plus-one computation.',
       '','Probabilities sum exact leading-space lowercase/title-case one-token forms. Other spellings/fragments and following punctuation/EOS are excluded. The control is an output-distribution calculation, not a run in which internal MOLT activity was preserved and logits were edited.',
       '','Reproduce: `python3 experiments/eleven-twelve-competition-20261103/analyze.py`. Exact values, source hash and limitations are in summary.json.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(out,indent=2))
