"""Select minimum tested successes and last coherent attempts before observed breakdown."""
import json
from pathlib import Path
S=Path(__file__).resolve().parent;W=S.parents[2];R=W/'results/2026-10-19-abstention-complete'
read=lambda p:json.loads(p.read_text());cases=read(S/'cases.json');rows=read(R/'measurements.json');P=read(S/'protocol.json')
byid={r['id']:r for r in rows};cells=[]
for c in cases:
 for scope in ['input','output','both']:
  xs=sorted([r for r in rows if r['case_id']==c['id'] and r['scope']==scope],key=lambda r:r['strength']);good=[r for r in xs if r['review']['success']];bad=[r for r in xs if not r['review']['coherent']];first=bad[0] if bad else None
  coherent=[r for r in xs if r['review']['coherent']];before=[r for r in coherent if first is None or r['strength']<first['strength']]
  selected=good[0] if good else before[-1] if before else None
  cells.append({'case_id':c['id'],'cohort':c['cohort'],'scope':scope,'success':bool(good),'minimum_successful_tested_strength':good[0]['strength'] if good else None,'successful_tested_strengths':[r['strength'] for r in good],'selected_id':selected['id'] if selected else None,'selected_strength':selected['strength'] if selected else 0,'first_incoherent_tested_strength':first['strength'] if first else None,'last_coherent_before_first_incoherent':before[-1]['strength'] if before else 0,'highest_coherent_tested_strength':coherent[-1]['strength'] if coherent else 0,'highest_tested_strength':xs[-1]['strength'] if xs else 0,'coherence_recovered_later':bool(first and any(r['strength']>first['strength'] for r in coherent)),'coherent_recovery_values':[r['strength'] for r in coherent if first and r['strength']>first['strength']],'tested_strengths':[r['strength'] for r in xs],'positive_trials':len(xs),'common_grid_complete':set(P['common_strengths'])<=set(r['strength'] for r in xs)})
fixed={scope:{cohort:{'clear_abstentions':sum(r['review']['success'] for r in rows if r['scope']==scope and r['cohort']==cohort and r['strength']==16),'coherent':sum(r['review']['coherent'] for r in rows if r['scope']==scope and r['cohort']==cohort and r['strength']==16),'answers_supplied':sum(r['review']['answer_present'] for r in rows if r['scope']==scope and r['cohort']==cohort and r['strength']==16)} for cohort in ['correct','made_up']} for scope in ['input','output','both']}
summary={'fixed_strength_16':fixed,'cases':20,'cells':cells,'unique_positive_trials':len(rows),'all_60_cells_common_grid_complete':all(x['common_grid_complete'] for x in cells),'common_grid':P['common_strengths'],'counts':{cohort:{scope:sum(x['success'] for x in cells if x['cohort']==cohort and x['scope']==scope) for scope in ['input','output','both']} for cohort in ['correct','made_up']},'selection_policy':P['display_policy']}
(R/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['cells','selection_policy']},indent=2))
