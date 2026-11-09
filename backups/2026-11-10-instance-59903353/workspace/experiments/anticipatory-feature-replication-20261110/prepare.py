"""Prepare frozen wording x visible-context comparisons for L34-T965."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,importlib.util,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=HERE/'scoring'
OUT=ROOT/'results/2026-11-10-before-k-description';OUT.mkdir(exist_ok=True)
SUFFIX='\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def fp(r):return hashlib.sha256(json.dumps([r['str_tokens'],r['activations']],ensure_ascii=False).encode()).hexdigest()
freeze=json.loads((HERE/'protocol.json').read_text())
feature_path=ROOT/'comparisons/2026-11-10-next-anticipatory-feature/feature-965.json'
feature=json.loads(feature_path.read_text())
recovered=json.loads((OUT/'recovered-windows.json').read_text())
assert not recovered['unmatched'] and not recovered['nonunique_match_windows']
assert recovered['feature_sha256']==sha(feature_path)
old_plan=json.loads((ROOT/'results/2026-11-09-t2375-explanation-refinement/heldout-plan.json').read_text())
prefixes={m:next(c['baseline_messages'][:-1] for c in old_plan['cases'] if c['method']==m) for m in freeze['intended_methods']}
spec=importlib.util.spec_from_file_location('display',ROOT/'artifacts/molt-dashboard-vast-20261028/source/crosslayer_transcoder/dashboard/token_display.py')
display=importlib.util.module_from_spec(spec);spec.loader.exec_module(display);escape=display.escape_added_tokens
descriptions={a:(HERE/f'{a}-explanation.txt').read_text().strip() for a in ('baseline','candidate')}
assert descriptions['baseline']==feature['explanation']
for a,t in descriptions.items():
 assert hashlib.sha256(t.encode()).hexdigest()==freeze[a+'_sha256']
 shutil.copyfile(HERE/f'{a}-explanation.txt',RUN/f'{a}-explanation.txt')
save(RUN/'explanation-freeze.json',freeze)
cases=[];coverage=[]
for method in freeze['intended_methods']:
 for i,row in enumerate(feature['methods'][method]['rows']):
  if not row['activating'] and method!='active_fuzz':continue
  key=fp(row);window=recovered['windows'][key];match=window['matches'][0]
  last=next(x for x in match['active_positions'] if x['window_position']==len(row['str_tokens'])-1)
  assert len(last['following_tokens'])>=16
  latin_k=last['following_text'].lstrip().startswith(('K','k'))
  cyrillic_k=last['following_text'].lstrip().startswith(('К','к'))
  if row['activating']:
   coverage.append(dict(method=method,row_index=i,source_fingerprint=key,peak_token=last['token'],gate=last['gate'],
        strict_latin_k_peak=latin_k,cyrillic_k_peak=cyrillic_k,following_text=last['following_text'],
        pre_k_already_visible_in_prefix=any(a['following_text'].lstrip().startswith(('K','k')) and a['window_position']<len(row['str_tokens'])-1 for a in match['active_positions'])))
  shown=escape(row.get('shown_text',''.join(row['str_tokens'])))
  for context in ['prefix_only','plus_16_source_tokens']:
   extra='' if context=='prefix_only' else escape(''.join(last['following_tokens'][:16]))
   case=dict(method=method,row_index=i+(100 if context!='prefix_only' else 0),original_row_index=i,
     context=context,activating=row['activating'],role='positive' if row['activating'] else 'inactive_highlight_control',
     source_fingerprint=key,strict_latin_k_peak=latin_k,cyrillic_k_peak=cyrillic_k,
     historical_prediction=row['prediction'],highlighted_positions=row.get('highlighted_positions'),
     continuation_token_ids=last['following_token_ids'][:16])
   for arm,text in descriptions.items():
    messages=deepcopy(prefixes[method]);messages.append(dict(role='user',content='Latent explanation: '+escape(text)+'\n\nText examples:\n\nExample 0: '+shown+extra+SUFFIX))
    case[arm+'_messages']=messages
   cases.append(case)
assert len(cases)==96
save(OUT/'coverage.json',dict(feature='L34-T965',available_positive_contexts=12,rows=coverage,
 latin_k_peak_per_method={m:sum(x['strict_latin_k_peak'] for x in coverage if x['method']==m) for m in prefixes},
 cyrillic_k_peak_per_method={m:sum(x['cyrillic_k_peak'] for x in coverage if x['method']==m) for m in prefixes}))
plan=dict(feature='L34-T965',cases=cases,**{a+'_explanation':t for a,t in descriptions.items()})
save(RUN/'heldout-plan.json',plan)
for src,name in [('runtime.json','expected-active-fuzz-runtime.json'),('judge-verification.json','expected-judge-verification.json')]:
 shutil.copyfile(ROOT/'results/2026-11-09-t2375-active-fuzz-local-judge'/src,RUN/name)
protocol=dict(feature='L34-T965',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
 judge_repo=freeze['judge_repo'],judge_revision=freeze['judge_revision'],
 existing_judge_root='/workspace/t2375-active-fuzz-local-judge-20261109',
 methods=freeze['intended_methods'],cases_per_method={'detection':24,'fuzz':24,'active_fuzz':48},
 positive_cases_per_context_per_method=12,contexts=['prefix_only','plus_16_source_tokens'],requests=192,unique_request_bodies=144,
 maximum_runtime_minutes=12,temperature=0,max_tokens=32,enable_thinking=False,
 prompt_reconstruction='Original labeled feature rows with the verified shared Delphi system/few-shot prefixes from the previous experiment; last user message reconstructed with the same scoring template. Not claimed as a byte-verified historical request export.',
 descriptors_frozen_before_evaluation_inspection=True,labels_unchanged=True,highlights_unchanged=True,
 feature_source_sha256=sha(feature_path),explanation_freeze_sha256=sha(HERE/'protocol.json'),
 heldout_plan_sha256=sha(RUN/'heldout-plan.json'),
 expected_active_fuzz_runtime_sha256=sha(RUN/'expected-active-fuzz-runtime.json'),
 expected_judge_verification_sha256=sha(RUN/'expected-judge-verification.json'),
 **{a+'_sha256':freeze[a+'_sha256'] for a in descriptions})
save(RUN/'protocol.json',protocol)
print(json.dumps(dict(cases=len(cases),requests=192,strict_before_latin_K=sum(x['strict_latin_k_peak'] for x in coverage if x['method']=='fuzz'),before_cyrillic_K=sum(x['cyrillic_k_peak'] for x in coverage if x['method']=='fuzz'))))
