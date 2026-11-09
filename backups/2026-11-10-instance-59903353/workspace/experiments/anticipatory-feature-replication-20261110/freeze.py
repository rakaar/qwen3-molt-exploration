"""Freeze one candidate explanation before inspecting its evaluation passages."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
metadata=ROOT/'comparisons/2026-11-10-next-anticipatory-feature/layer-34.jsonl'
feature=next(json.loads(s) for s in metadata.read_text().splitlines() if json.loads(s)['feature_id']==965)
old=feature['explanation']
new=('Often activates on the token immediately before text beginning with K or k, including within words, names, code and quantities. '
     'Examples include DIS before K in DISK, 4 before K in 4K, and honky-ton before the final k. '
     'Activation belongs to the preceding word fragment, digit, space or punctuation, rather than necessarily to K itself. '
     'Some activations have non-K continuations, so this is a frequent pattern, not an exclusive rule.')
assert len(new.split())<=80
assert not (HERE/'protocol.json').exists(),'Do not overwrite the frozen candidate after inspection'
for arm,text in [('baseline',old),('candidate',new)]:
 (HERE/f'{arm}-explanation.txt').write_text(text+'\n')
protocol=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),feature='L34-T965',layer=34,feature_id=965,
 selection='Exploratory candidate from the user-linked report; selected for a clear before-K pattern and a description omitting K. Historical aggregate scores were visible, but evaluation passages have not been inspected for this candidate.',
 explanation_evidence='Original dashboard strongest training-partition examples: conversations 3332, 8714, 11097, 10751, 10038, 292, 1791, 3132. Conversation 4507 is an explicit non-K counterexample.',
 baseline_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
 descriptions_frozen=True,context_arms=['prefix_only','plus_16_source_tokens'],
 intended_methods=['detection','fuzz','active_fuzz'],judge_repo='vrfai/Qwen3.8-27B-NVFP4',judge_revision='1561c70b22784dba7c6943b3271fc5f185adda01',
 study_type='Exploratory matched diagnostic, not a preregistered or wholly unseen feature evaluation. Do not change wording after scoring feedback.',
 spending='Reuse existing authorized instance 59903353 within $10 cap; preserve its authorized 03:00 IST Nov10 stop. No new rental authorized by this script.')
(HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(protocol,indent=2))
