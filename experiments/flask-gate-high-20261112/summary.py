"""Combine both fixed-prompt strength ranges without hiding failed or incoherent outputs."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'results/2026-11-12-flask-gate-high'
s = json.loads((OUT/'analysis.json').read_text())
old = json.loads((ROOT/'results/2026-11-12-flask-gate-increase/analysis.json').read_text())
assert s['prompt'] == old['prompt']
refine = json.loads((ROOT/'results/2026-11-12-flask-gate-refine/analysis.json').read_text())
assert refine['prompt'] == s['prompt']
new_response_rows = s['rows'] + [v for v in refine['rows'] if v['target_gate'] is not None and v['target_gate'] > 64]
rows = old['rows'] + [v for v in new_response_rows if v['target_gate'] is not None and v['target_gate'] > 64]
rows.sort(key=lambda v: -1 if v['target_gate'] is None else v['target_gate'])
assert len({v['target_gate'] for v in rows}) == len(rows)
new_response_rows.sort(key=lambda v: -1 if v['target_gate'] is None else v['target_gate'])
combined = dict(prompt=s['prompt'],rows=rows,new_range_audit=s['audit'],refinement_audit=refine['audit'],refinement_protocol='results/2026-11-12-flask-gate-refine/protocol.json',previous_range='results/2026-11-12-flask-gate-increase',note='One fixed prompt; the native and strength-64 measurements are reproduced on the replacement host before interpreting the extended range.')
(OUT/'combined-analysis.json').write_text(json.dumps(combined,indent=2,ensure_ascii=False)+'\n')
lines = ['# All-input T2375 steering: extending the same oil-flask prompt above 64','','```text',s['prompt'],'```','','Both ranges use the exact same prompt and 39 supplied input positions, including all special/template and assistant-prefix tokens. Every supplied input position is selected without filtering for natural activation. Generated/candidate tokens remain unedited.','','| Fixed gate | Response preview | P(water) | P(oil) | Stop |','|---|---|---:|---:|---|']
for v in rows:
    name='Native' if v['target_gate'] is None else str(int(v['target_gate']))
    preview=v['response'].replace('\n',' ↵ ').replace('|','&#124;')
    if len(preview)>70:preview=preview[:67]+'...'
    lines.append(f"| {name} | {preview} | {100*v['p_water']:.6e}% | {100*v['p_oil']:.6f}% | {v['stop_reason']} |")
lines += ['','## Full new responses','']
for v in new_response_rows:
    if v['target_gate'] is None:continue
    lines += [f"### Gate {int(v['target_gate'])}",'','```text',v['response'],'```',f"Stop: {v['stop_reason']}. Strict single-word water answer: {v['clean_water_answer']}. All {v['edited_input_positions']} input positions edited; generated edits: {v['generated_positions_edited']}.",'']
clean=[int(v['target_gate']) for v in new_response_rows if v['target_gate'] is not None and v['clean_water_answer']]
lines += ['## Interpretation','',('A strict single-word water response occurred at these tested gate values: '+', '.join(map(str,clean))+'.') if clean else 'No strict single-word water response occurred at the tested higher values.','', 'These are the tested values, not a continuous search or a global threshold. A water prefix inside repeated/truncated text is reported separately from a strict single-word answer. This single-prompt, high-strength pilot does not establish a selective water-related mechanism; no unrelated norm-matched transform control was run.','','## Intervention and probability definition','','At layer 29, add `(target_gate - natural_gate) * local_operator` to native MLP output in FP32 followed by one BF16 cast. The local operator depends on the input. No full MLP replacement, prompt editing or generated-token steering. Later native computation remains active.','','P(water) and P(oil) sum four prescribed disjoint complete word-token prefixes: lowercase/titlecase with zero or one leading space. They exclude punctuation/EOS and arbitrary longer answer phrases; they are literal prefix events, not semantic-answer probabilities. Per-variant word-plus-EOS scores are in the raw captures.','','## Verification','','Pinned model and full L29 checkpoint hashes; BF16 native model, FP32 observer/edit, eager attention, TF32 off, greedy, thinking off, maximum 32 generated tokens. Native/zero equality is exact. Native baseline and strength 64 reproduce the previous host exactly. The baseline is hash-locked before the frozen 64,128,256,512,1024 sweep. After those outcomes, an explicitly exploratory refinement tested 80,96,112,144,160,192 and repeated64; its frozen protocol and raw captures are in the sibling flask-gate-refine directory. Independent CPU checks verify every selected position, scalar, operator, intended/actual vector edit and exact BF16 rounding. Raw outputs, top-five next-token scores, logs and setup verification are retained.','']
text='\n'.join(lines)
(HERE/'README.md').write_text(text)
(OUT/'README.md').write_text(text)
print(json.dumps({'combined_rows':len(rows),'new_clean_water_strengths':clean}))
