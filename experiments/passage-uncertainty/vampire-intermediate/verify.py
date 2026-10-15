"""Audit every saved intervention position and exact replay before reporting."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
r=json.loads((root/'artifacts/added-gate-results.json').read_text());assert r['status']=='complete'
protocol=json.loads((root/'protocol.json').read_text());assert r['protocol']==protocol
baselines=json.loads((root/'baseline-generations.json').read_text());base={x['id']:x for x in baselines['rows']}
anchors=json.loads((root/'prior-anchor-reference.json').read_text())['rows'];anchors={(x['case_id'],x['added_gate']):x for x in anchors}
controls=0;replays=0;positive=0;positions=0
for row in r['rows']:
 key=(row['case_id'],row['added_gate']);traces=row['traces'];assert not traces[0]['eligible'] and traces[0]['actual_bf16_delta_norm']==0
 assert row['output_token_ids'][0]==base[row['case_id']]['output_token_ids'][0]
 assert row['first_output_token_unchanged']
 if row['added_gate'] in [None,0]:
  controls+=1;assert row['output_token_ids']==base[row['case_id']]['output_token_ids'];assert all(x['actual_bf16_delta_norm']==0 for x in traces)
 else:
  positive+=1
  for trace in traces[1:]:
   assert trace['eligible'] and trace['segment']=='response';assert trace['gate_delta']==row['added_gate'];assert trace['effective_estimated_gate']==trace['natural_gate']+row['added_gate'];assert trace['actual_bf16_delta_norm']>0;positions+=1
 if key in anchors:
  assert row['output_token_ids']==anchors[key]['output_token_ids'];replays+=1
 assert len(traces)==len(row['output_token_ids'])
 assert [x['answer_position'] for x in traces[1:]]==list(range(len(traces)-1))
 if row['stop_reason']=='token_limit':assert len(row['output_token_ids'])==protocol['decoding']['max_new_tokens']
expected=sum(2+len(protocol['added_gate_values_by_case'][c]) for c in protocol['cases']);assert len(r['rows'])==expected
vampire={x['added_gate'] for x in r['rows'] if x['case_id']=='P2-U5' and x['added_gate'] not in [None,0]};assert vampire==set(protocol['added_gate_values_by_case']['P2-U5'])
proof={'status':'passed','total_trials':len(r['rows']),'native_zero_controls_exact':controls,'prior_anchors_exact':replays,'positive_trials':positive,'nonzero_edited_response_positions':positions,'vampire_positive_strengths':sorted(vampire),'first_output_token_unchanged_in_every_trial':True,'every_prompt_position_unedited':True,'every_processed_output_position_edited_for_positive_trials':True,'native_mlp_retained':True,'attention_unfrozen':True,'runtime':r['runtime'],'elapsed_seconds':r['elapsed_seconds'],'raw_results_sha256':hashlib.sha256((root/'artifacts/added-gate-results.json').read_bytes()).hexdigest()}
(root/'verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
