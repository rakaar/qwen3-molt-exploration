"""Published 12-feature swap with selected-weight adapters and upstream steering."""
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


with torch.inference_mode():
    caps,base = capture(ids)
    donor_caps,donor_base = capture(donor_ids)
    baseline = summarize(caps.original_logits[0])
    donor = summarize(donor_caps.original_logits[0])
    print('BASELINE',baseline,flush=True);print('DONOR',donor,flush=True)
    assert baseline['top_token']=='large' and donor['top_token']=='cold'
    feature_checks=[]
    for group,fs in recipe['selection'].items():
        for f in fs:
            layer,index=f['layer'],f['feature']
            feature_checks.append({'group':group,'layer':layer,'feature':index,
                'recipient_activation':float(base.features[layer][9,index]),
                'donor_activation':float(donor_base.features[layer][9,index]),
                'published_reference_activation':f['act']})
    report={'protocol':recipe,'transcoder_revision':manifest['revision'],
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                   'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()},
        'implementation':'Verbatim upstream _steer_base_model and frozen-attention helpers; adapters load only the 12 selected features. Baseline features encoded on captured real MLP inputs.',
        'recipient_token_ids':ids[0].tolist(),'donor_token_ids':donor_ids[0].tolist(),
        'baseline':baseline,'donor_baseline':donor,'feature_checks':feature_checks,'runs':[]}

    def run(ivs):
        return _steer_base_model(model,tc,ids,ivs,base,caps,l_max=22,patch_end_layer=22,
            freeze_attention=True,n_bos_tokens=1,mlp_name_template='model.layers.{layer}.mlp',
            attn_name_template='model.layers.{layer}.self_attn',
            layernorm_templates=['model.layers.{layer}.input_layernorm','model.layers.{layer}.post_attention_layernorm'],
            final_norm_name='model.norm',readout_layers=None)

    # Essential protocol control: frozen attention + clean MLP pinning, with zero delta.
    zero=[FeatureIntervention(22,113890,position=9,m=0)]
    control=run(zero)
    diff=float((control.ablated_logits-control.baseline_logits).abs().max())
    report['zero_delta_control']={**summarize(control.ablated_logits),'max_abs_logit_difference':diff}
    assert diff < 0.15, f'No-op control changed logits: {diff}'
    print('ZERO_DELTA_MAX_LOGIT_DIFF',diff,flush=True)

    for strength in [0,1.5,2.25,3,4.5]:
        started=time.monotonic()
        if strength==0:
            row={'strength':0,**baseline}
        else:
            ivs=[]
            for f in recipe['selection']['small (multilingual)']:
                ivs.append(FeatureIntervention(f['layer'],f['feature'],position=9,m=-strength))
            for f in recipe['selection']['hot (multilingual)']:
                ivs.append(FeatureIntervention(f['layer'],f['feature'],position=9,
                    value=strength*f['acts']['hot_en']))
            result=run(ivs)
            row={'strength':strength,**summarize(result.ablated_logits)}
        published=next(x for x in recipe['saved_results'] if x['strength']==strength)
        row.update({'published_P_cold':published['P_cold'],'published_P_large':published['P_large'],
                    'seconds':time.monotonic()-started})
        report['runs'].append(row)
        print('RUN',json.dumps(row),flush=True)
        (OUT/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')

    # Prove upstream's finally blocks restored hooks and attention forwards.
    restored=model(ids).logits[0]
    restore_diff=float((restored-caps.original_logits[0]).abs().max())
    assert restore_diff == 0
    report['restoration_max_abs_logit_difference']=restore_diff
    report['peak_gpu_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    (OUT/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
print('REPRODUCTION_OK',flush=True)
