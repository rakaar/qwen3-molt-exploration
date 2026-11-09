"""Freeze self-authored probes before any inference; preserve the revised text."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
cases=[]
def add(name,group,prompt,fmt='raw',**extra):
    cases.append(dict(id=name,group=group,prompt=prompt,prompt_format=fmt,**extra))

water=[
 ('ingredient','Ingredients: flour, sugar, water'),
 ('list','Essential supplies: food, shelter, water'),
 ('environment','Clean air and fresh water'),
 ('capitalized','Water is essential for life.'),
 ('sentence','Please pour the water into the glass.'),
 ('isolated','water'),
]
for name,text in water:add('water-'+name,'water',text)
for name,text in [
 ('ingredient','Ingredients: flour, sugar, milk'),
 ('list','Essential supplies: food, shelter, blankets'),
 ('environment','Clean air and fresh food'),
 ('capitalized','Food is essential for life.'),
 ('sentence','Please pour the milk into the glass.'),
 ('isolated','milk'),
]:add('control-'+name,'matched_nonwater',text)

domains={
 'month':'the {ordinal} month of the year is',
 'alphabet':'the {ordinal} letter of the english alphabet is',
 'roman':'the {ordinal} positive integer in roman numerals is',
 'integer':'the {ordinal} positive integer is',
}
words={10:'tenth',11:'eleventh',12:'twelfth'}
for domain,template in domains.items():
    for n in [10,11,12]:
        for form,ordinal in [('word',words[n]),('digit',str(n)+'th')]:
            add(f'{domain}-{n}-{form}','ordinal',template.format(ordinal=ordinal),domain=domain,ordinal=n,wording=form)
for n in [10,11,12]:
    add(f'recipe-boundary-{n}','list_boundary',f'{n-1}. Pour the remaining sauce over the pasta, then spread it out evenly.\n\n{n}. Sprinkle extra Parmesan cheese over the top.',ordinal=n)
for domain,stem in [
 ('month','month of the year'),('alphabet','letter of the English alphabet'),
 ('roman','positive integer in Roman numerals')]:
    for n in [10,11,12]:
        add(f'chat-{domain}-{n}','chat_ordinal',f'What is the {words[n]} {stem}? Answer briefly.','chat',domain=domain,ordinal=n)
for n,text in enumerate(['List three uses of water.','List three uses of milk.','What is the eleventh positive integer?'],1):
    add(f'chat-extra-{n}','chat_extra',text,'chat')

description=(PROJECT/'experiments/t2375-explanation-refinement-20261109/candidate-explanation.txt').read_text().strip()
protocol=dict(feature=2375,layer=29,date_local='2026-11-09',description=description,
    description_sha256=hashlib.sha256(description.encode()).hexdigest(),
    base_revision='1cfa9a7208912126459214e8b04321603b3df60c',
    molt_revision='081ce901b090b64aeecf9bc8709b57c10411e32a',
    tool='verify_hypothesis',cases=cases,maximum_runtime_minutes=20,
    intervention=None,heldout_benchmark=False,
    scope='Self-authored input-token probes derived from the frozen description; no judge and no generation.',
    design_notes=[
      'The tool accepts texts, not an explanation. The description is frozen metadata used to design probes.',
      'Preserve zero gates, exact token IDs, added/control tokens, and every input position.',
      'Use the unmodified upstream MCP verify_hypothesis registration and backing method.',
      'A self-authored-only session bypasses corpus initialization without fabricating corpus examples.',
      'The original SteeringEngine and MoltGates run native Qwen3; derived gate-only checkpoint contains exact encoder tensors.',
      'Compute the original dashboard BF16-autocast gate; FP32 gate values are an additional precision diagnostic.',
      'Neighboring ordinals are probes of specificity, not pre-labelled negatives; the revised description already allows wording dependence.',
      'No score of description quality is inferred from these hand-authored probes.',
    ])
path=ROOT/'protocol.json'
blob=json.dumps(protocol,indent=2,ensure_ascii=False)+'\n'
if path.exists():assert path.read_text()==blob,'Do not change frozen probes after looking at results'
else:path.write_text(blob)

source=PROJECT/'artifacts/molt-dashboard-vast-20261028/source'
names=['tools/molt_agentic_mcp.py','crosslayer_transcoder/__init__.py',
       'crosslayer_transcoder/model/__init__.py','crosslayer_transcoder/model/jumprelu.py',
       'crosslayer_transcoder/model/standardize.py','crosslayer_transcoder/model/serializable_module.py',
       'crosslayer_transcoder/dashboard/__init__.py','crosslayer_transcoder/dashboard/gates.py',
       'crosslayer_transcoder/dashboard/logits.py','crosslayer_transcoder/dashboard/token_display.py']
for name in names:
    dest=ROOT/'upstream'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/name,dest)
for name in ['expected-base-verification.json','weights/manifest.json','weights/L29-T2375.npz']:
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(PROJECT/'experiments/eleven-token-confound-20261108'/name,dest)
print(json.dumps(dict(cases=len(cases),protocol_sha256=hashlib.sha256(blob.encode()).hexdigest(),description_sha256=protocol['description_sha256'])))
