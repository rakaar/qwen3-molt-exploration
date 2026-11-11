from pathlib import Path
from datetime import datetime,timezone
import copy,json,shutil,hashlib
root=Path.cwd();src=root/'experiments/month-codewords-20261112';dest=src/'matched-controls';dest.mkdir(exist_ok=True);(dest/'weights').mkdir(exist_ok=True)
p=json.loads((src/'protocol.json').read_text());cases=[]
for m in ['mapping-2','mapping-4']:
 for n in [11,12]:
  c=copy.deepcopy(next(c for c in p['cases'] if c['mapping_id']==m and c['prompt_format']=='chat' and c['ordinal']==n))
  book=c['prompt'][c['codebook_span'][0]:c['codebook_span'][1]]
  c['prompt']=c['prompt'].replace('then reply with its codeword only. Do not reply with the month name or a number.','then reply with the month name only. Do not reply with a codeword or a number.')
  c.update(id=f'name-{m}-chat-{n}',mapping_id='name-'+m,expected='November' if n==11 else 'December',code_11='November',code_12='December',calibration=True,control_type='same_codebook_month_name')
  start=c['prompt'].index(c['query']);c['query_span']=[start,len(c['prompt'])];start=c['prompt'].index(book);c['codebook_span']=[start,start+len(book)];cases.append(c)
for n,ordinal in [(11,'eleventh'),(12,'twelfth')]:
 query=f'the {ordinal} month of the year is';prefix='Reply with the month name only.\n\n';cases.append(dict(id=f'plain-chat-{n}',mapping_id='plain-chat',prompt_format='chat',ordinal=n,prompt=prefix+query,query=query,query_span=[len(prefix),len(prefix+query)],codebook_span=[0,0],expected='November' if n==11 else 'December',code_11='November',code_12='December',calibration=True,control_type='plain_chat_month_name'))
p.update(experiment='Adaptive matched month-name controls after the codeword pilot',cases=cases,control_phase_frozen_at_utc=datetime.now(timezone.utc).isoformat(),control_phase_reason='The codeword pilot preserves both eligible answers. Add same-chat-format month-name controls using the same two codebooks, plus plain chat controls, to expose prompt-format/task confounds. Chosen after pilot results, before control outcomes; exploratory follow-up.',maximum_runtime_minutes=15)
(dest/'protocol.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
s=(src/'run.py').read_text().replace("    assert all(x['eligible'] for x in result['cases'] if x['case']['calibration']), 'Plain-month native positive control failed'", "    # Preserve native-failed matched controls rather than interpreting an ablation on them.")
(dest/'run.py').write_text(s)
for n in ['expected-base-verification.json','weights/L29-T2375.npz']:shutil.copy2(src/n,dest/n)
s=(root/'gpu-rentals/2026-11-11-layer29-setup/month_codewords_remote.py').read_text().replace("SRC=PROJECT/'experiments/month-codewords-20261112';ROOT='/workspace/month-codewords-20261112'", "SRC=PROJECT/'experiments/month-codewords-20261112/matched-controls';ROOT='/workspace/month-codewords-controls-20261112'")
s=s.replace('month-codewords-source','month-codewords-controls-source').replace('month-codewords-results','month-codewords-controls-results').replace('month-codewords-run-start','month-codewords-controls-run-start').replace('month-codewords-backup-verification','month-codewords-controls-backup-verification')
s=s.replace("arcname='month-codewords-20261112/", "arcname='month-codewords-controls-20261112/")
s=s.replace("len(r['cases'])==22", "len(r['cases'])==6")
s=s.replace("PROJECT/'results/2026-11-12-month-codewords'", "PROJECT/'results/2026-11-12-month-codewords/matched-controls'")
(root/'gpu-rentals/2026-11-11-layer29-setup/month_codewords_controls_remote.py').write_text(s)
print(json.dumps({'control_cases':len(cases),'source_sha256':hashlib.sha256((dest/'run.py').read_bytes()).hexdigest()}))
