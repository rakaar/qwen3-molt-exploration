"""Gate comparison with the successful reproduction's model and steering setup."""
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from small_hot_upstream import FeatureIntervention, _steer_base_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/small-hot-reproduction'
recipe = json.loads((OUT/'recipe.json').read_text())
manifest = json.loads((OUT/'weights-manifest.json').read_text())
torch.manual_seed(0)
torch.set_num_threads(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SelectedFeatures:
    """Only selected entries are accessible; missing features raise instead of faking zeros."""
    def __init__(self, feature_ids, values):
        self.index = {f:i for i,f in enumerate(feature_ids)}
        self.values = values
    def dim(self): return 2
    def __getitem__(self, key):
        position, feature = key
        return self.values[position, self.index[feature]]


class SelectedLayer:
    def __init__(self, spec):
        blob = (OUT/spec['file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == spec['sha256']
        self.ids = [f['feature'] for f in spec['features']]
        self.index = {f:i for i,f in enumerate(self.ids)}
        enc, dec, bias = [], [], []
        for i in range(len(self.ids)):
            row = torch.frombuffer(bytearray(blob[i*10242:(i+1)*10242]), dtype=torch.bfloat16)
            enc.append(row[:2560]);dec.append(row[2560:5120]);bias.append(row[5120])
        self.enc = torch.stack(enc).cuda()
        self.dec = torch.stack(dec).cuda()
        self.bias = torch.stack(bias).cuda()
    def encode_selected(self, x):
        return SelectedFeatures(self.ids, F.relu(F.linear(x, self.enc, self.bias)))
    def _get_decoder_vectors(self, indices):
        return self.dec[[self.index[int(i)] for i in indices]]


class SelectedSet:
    def __init__(self):
        self.transcoders = {s['layer']:SelectedLayer(s) for s in manifest['layers']}
    def __len__(self): return 36


tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/qwen3-4b', local_files_only=True)
def tokenize(prompt):
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],
        return_tensors='pt',add_generation_prompt=True,enable_thinking=False).cuda()
ids = tokenize(recipe['recipient_user_prompt'])
donor_ids = tokenize(recipe['donor_user_prompt'])
assert ids.shape == donor_ids.shape == (1,29)
assert tokenizer.decode([int(ids[0,9])]) == 'small'
assert tokenizer.decode([int(donor_ids[0,9])]) == 'hot'
model = AutoModelForCausalLM.from_pretrained(ROOT/'models/qwen3-4b',dtype=torch.bfloat16,
    local_files_only=True,attn_implementation='eager').cuda().eval().requires_grad_(False)
model.config.use_cache = False
tc = SelectedSet()


def capture(input_ids):
    caps = SimpleNamespace(attn_weights={}, mlp_outputs={}, rmsnorm_scales={})
    mlp_inputs, hooks = {}, []
    for i,layer in enumerate(model.model.layers):
        def attn_hook(_m,_a,out,i=i): caps.attn_weights[i] = out[1].detach()
        def mlp_hook(_m,args,out,i=i):
            caps.mlp_outputs[i] = out.detach()
            if i in tc.transcoders: mlp_inputs[i] = args[0][0].detach()
        hooks.append(layer.self_attn.register_forward_hook(attn_hook))
        hooks.append(layer.mlp.register_forward_hook(mlp_hook))
    try:
        with torch.inference_mode():
            caps.original_logits = model(input_ids,output_attentions=True,use_cache=False).logits.detach()
    finally:
        for hook in hooks: hook.remove()
    assert len(caps.attn_weights) == len(caps.mlp_outputs) == 36
    base = SimpleNamespace(features={i:tc.transcoders[i].encode_selected(x) for i,x in mlp_inputs.items()})
    return caps,base


def summarize(logits):
    p = logits[-1].float().softmax(-1)
    vals,idx = p.topk(8)
    return {'P_large':float(p[16767]),'P_cold':float(p[87072]),
            'top_token':tokenizer.decode([int(idx[0])]),
            'top_tokens':[{'id':int(i),'text':tokenizer.decode([int(i)]),'probability':float(v)} for i,v in zip(idx,vals)]}


import sys
from safetensors.torch import load_file
sys.path.insert(0, str(ROOT/'crosslayer-transcoder'))
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer

RESULTS = ROOT/'results/small-hot-molt-gates-21-22'
RESULTS.mkdir(parents=True, exist_ok=True)
LAYERS = [21,22]
STRENGTH = 4.5
inputs, gate_hooks = {}, []
current = None
for layer in LAYERS:
    def observe(_module, args, layer=layer):
        inputs.setdefault(current,{})[layer] = args[0][0].detach().cpu().clone()
    gate_hooks.append(model.model.layers[layer].post_attention_layernorm.register_forward_pre_hook(observe))

def steer(ivs):
    return _steer_base_model(model,tc,ids,ivs,base,caps,l_max=22,patch_end_layer=22,
        freeze_attention=True,n_bos_tokens=1,mlp_name_template='model.layers.{layer}.mlp',
        attn_name_template='model.layers.{layer}.self_attn',
        layernorm_templates=['model.layers.{layer}.input_layernorm','model.layers.{layer}.post_attention_layernorm'],
        final_norm_name='model.norm',readout_layers=None)

report = {'layers_zero_based':LAYERS,'strength':STRENGTH,
    'recipient_prompt':recipe['recipient_user_prompt'],'donor_prompt':recipe['donor_user_prompt'],
    'input_location':'Before post_attention_layernorm, as required by Georg MOLT checkpoints',
    'protocol':'Published decoder deltas on original MLP outputs; attention patterns frozen; MLP outputs 0..22 pinned clean plus deltas. MOLTs are passive gate readouts.',
    'tokens':[{'position':p,'id':int(i),'text':tokenizer.decode([int(i)])} for p,i in enumerate(ids[0])],
    'readouts':{},'layers':[],
    'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()}}
with torch.inference_mode():
    try:
        current='baseline'
        caps,base=capture(ids)
        report['readouts'][current]=summarize(caps.original_logits[0])
        current='natural_hot'
        donor_caps,donor_base=capture(donor_ids)
        report['readouts'][current]=summarize(donor_caps.original_logits[0])
        small_ivs=[FeatureIntervention(f['layer'],f['feature'],position=9,m=-STRENGTH)
                   for f in recipe['selection']['small (multilingual)']]
        hot_ivs=[FeatureIntervention(f['layer'],f['feature'],position=9,value=STRENGTH*f['acts']['hot_en'])
                 for f in recipe['selection']['hot (multilingual)']]
        for current,ivs in [('zero_delta',[FeatureIntervention(22,113890,position=9,m=0)]),
                            ('suppression_only',small_ivs),('addition_only',hot_ivs),('combined',small_ivs+hot_ivs)]:
            result=steer(ivs)
            report['readouts'][current]=summarize(result.ablated_logits)
            print('READOUT',current,report['readouts'][current]['top_token'],
                  'P_COLD',report['readouts'][current]['P_cold'],flush=True)
            if current=='zero_delta':
                assert torch.equal(result.ablated_logits,result.baseline_logits)
                for layer in LAYERS: assert torch.equal(inputs[current][layer],inputs['baseline'][layer])
        reference=next(x for x in json.loads((OUT/'reproduction.json').read_text())['runs'] if x['strength']==STRENGTH)
        assert abs(report['readouts']['combined']['P_cold']-reference['P_cold'])<1e-7
        current='restored'
        restored=model(ids).logits[0]
        assert torch.equal(restored,caps.original_logits[0])
        for layer in LAYERS: assert torch.equal(inputs[current][layer],inputs['baseline'][layer])
    finally:
        for h in gate_hooks: h.remove()
    report['controls']={'zero_delta_logits_and_inputs_exact':True,'restored_logits_and_inputs_exact':True,
                        'combined_matches_prior_reproduction':True}
    torch.save(inputs,RESULTS/'captured-prenorm-inputs.pt')
    weight_manifest=json.loads((RESULTS/'gate-weights-manifest.json').read_text())
    report['molt_revision']=weight_manifest['revision']
    report['download_bytes']=sum(x['download_bytes'] for x in weight_manifest['layers'])
    full_gate_tensors={}
    for layer in LAYERS:
        spec=next(x for x in weight_manifest['layers'] if x['layer']==layer)
        path=RESULTS/spec['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
        weights=load_file(path)
        config=json.loads((RESULTS/f'layer-{layer}-config.json').read_text())
        standardizer=DimensionwiseInputStandardizer(config['standardizer_n_layers'],config['d_acts'])
        encoder=torch.nn.Linear(config['d_acts'],config['n_features'])
        nonlinearity=JumpReLU(theta=0.,bandwidth=1.,n_layers=1,d_features=config['n_features'])
        for module,prefix in [(standardizer,'input_standardizer'),(encoder,'e'),(nonlinearity,'nonlinearity')]:
            module.load_state_dict({k.split('.',1)[1]:v for k,v in weights.items() if k.startswith(prefix+'.')},strict=True)
            module.cuda().eval().requires_grad_(False)
        standardizer.is_initialized=True
        assert bool((standardizer.std[layer]>0).all())
        gates,bf_gates={},{}
        for condition in inputs:
            x=inputs[condition][layer].cuda()
            gates[condition]=nonlinearity(encoder(standardizer(x.float(),layer))).cpu()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                bf_gates[condition]=nonlinearity(encoder(standardizer(x,layer))).cpu()
            assert bool(torch.isfinite(gates[condition]).all())
        assert torch.equal(gates['baseline'],gates['zero_delta'])
        assert torch.equal(gates['baseline'],gates['restored'])
        entry={'layer':layer,'total_gates':config['n_features'],
               'precision':'FP32 gate computation on BF16 residuals; BF16 autocast sensitivity check',
               'conditions':{},'comparisons':{}}
        for condition,g in gates.items():
            full_gate_tensors[f'layer_{layer}_{condition}']=g
            entry['conditions'][condition]={
                'counts_by_token':(g>0).sum(-1).tolist(),
                'mean_active_count':float((g>0).sum(-1).float().mean()),
                'bf16_counts_by_token':(bf_gates[condition]>0).sum(-1).tolist(),
                'bf16_active_set_disagreements_by_token':((g>0)!=(bf_gates[condition]>0)).sum(-1).tolist(),
                'active_by_token':[[{'id':int(i),'gate':float(g[p,i]),'bf16_gate':float(bf_gates[condition][p,i])}
                                   for i in torch.where(g[p]>0)[0]] for p in range(len(g))]}
        baseline=gates['baseline']
        for condition in ['suppression_only','addition_only','combined','natural_hot']:
            g=gates[condition]
            comparisons=[]
            for pos in range(len(g)):
                before=set(torch.where(baseline[pos]>0)[0].tolist());after=set(torch.where(g[pos]>0)[0].tolist())
                changes=[{'id':i,'before':float(baseline[pos,i]),'after':float(g[pos,i]),
                          'delta':float(g[pos,i]-baseline[pos,i])} for i in before|after]
                changes.sort(key=lambda x:-abs(x['delta']))
                dx=inputs[condition][layer][pos].float()-inputs['baseline'][layer][pos].float()
                comparisons.append({'position':pos,'before_count':len(before),'after_count':len(after),
                    'turned_on':sorted(after-before),'turned_off':sorted(before-after),'shared':sorted(before&after),
                    'gate_changes':changes,'input_delta_l2':float(dx.norm()),
                    'relative_input_delta_l2':float(dx.norm()/inputs['baseline'][layer][pos].float().norm().clamp_min(1e-12))})
            entry['comparisons'][condition]=comparisons
        report['layers'].append(entry)
        for pos in [9,28]:
            print('GATE_COMPARISON',layer,pos,json.dumps(entry['comparisons']['combined'][pos]),flush=True)
        (RESULTS/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(full_gate_tensors,RESULTS/'all-gates.pt')
print('MOLT_GATE_COMPARISON_OK',flush=True)

