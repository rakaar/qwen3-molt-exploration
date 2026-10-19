"""Verify saved gate measurements and make local tables and a static figure."""
import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'remote-export'
D = json.loads((RUN / 'artifacts/measurements.json').read_text())
B = json.loads((RUN / 'artifacts/baseline.json').read_text())
P = json.loads((RUN / 'protocol.json').read_text())
F = ['L3-T2334', 'L3-T1372']
assert D['status'] == 'complete' and len(D['rows']) == 22
assert D['all_observation_hooks_return_none'] and D['native_attention_and_mlp_unmodified']
assert D['native_generation_replay_token_identical']
assert [r['id'] for r in D['rows']] == P['case_ids'] == [r['id'] for r in B['rows']]
for name, digest in D['source_sha256'].items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == digest
assets = {f"L3-T{a['transform']}": a for a in D['assets']['features']}
for a in assets.values():
    assert hashlib.sha256((RUN / 'weights' / a['file']).read_bytes()).hexdigest() == a['sha256']
for item in json.loads((RUN / 'download-manifest.json').read_text()):
    path = RUN / item['file']
    assert path.stat().st_size == item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['file']

summary = []
all_tokens = []
for row, baseline in zip(D['rows'], B['rows']):
    assert row['baseline_generation_no_hooks'] and row['all_native_greedy_predictions_verified']
    for key in ['answer', 'chat', 'prompt_token_ids', 'output_token_ids', 'target_positions']:
        assert row[key] == baseline[key]
    assert [t['token_id'] for t in row['prompt_tokens']] == row['prompt_token_ids']
    assert [t['full_chat_position'] for t in row['prompt_tokens']] == list(range(len(row['prompt_tokens'])))
    assert [t['token_id'] for t in row['answer_tokens']] == row['output_token_ids'][:len(row['answer_tokens'])]
    assert ''.join(t['token'] for t in row['answer_tokens']) == row['answer']
    source = [t for t in row['prompt_tokens'] if t['target_token']]
    assert [t['full_chat_position'] for t in source] == row['target_positions']
    response_text = ''.join(t['token'] for t in row['answer_tokens'])
    spans = [m.span() for m in re.finditer(r'(?i)(?<![a-z])ch(?![a-z])', response_text)]
    ch_tokens = []
    cursor = 0
    for token in row['answer_tokens']:
        end = cursor + len(token['token'])
        if any(end > a and cursor < b for a, b in spans):
            ch_tokens.append(token)
        cursor = end
    rec = {'id': row['id'], 'family': row['family'], 'target': row['target'],
        'answer': row['answer'], 'expected': row['expected'], 'source_tokens': [],
        'answer_CH_tokens': [], 'source_gate_max': {}, 'answer_CH_gate_max': {},
        'prompt_final_gate': {}, 'stop_reason': row['stop_reason']}
    for feature in F:
        rec['source_gate_max'][feature] = max(t['features'][feature]['gate'] for t in source)
        rec['answer_CH_gate_max'][feature] = max((t['features'][feature]['gate'] for t in ch_tokens), default=None)
        rec['prompt_final_gate'][feature] = row['prompt_tokens'][-1]['features'][feature]['gate']
    for field, tokens in [('source_tokens', source), ('answer_CH_tokens', ch_tokens)]:
        rec[field] = [{'token': t['token'], 'token_id': t['token_id'], 'position': t['full_chat_position'],
            'gates': {feature: t['features'][feature]['gate'] for feature in F}} for t in tokens]
    for token in row['prompt_tokens'] + row['answer_tokens']:
        for feature in F:
            value = token['features'][feature]
            assert all(math.isfinite(x) for x in value.values())
            expected = value['raw'] if value['raw'] > assets[feature]['theta'] and value['raw'] > 0 else 0
            assert value['gate'] == expected
            assert math.isclose(value['contribution_norm'], value['gate'] * value['lowrank_norm'], rel_tol=4e-6, abs_tol=1e-8)
            all_tokens.append({'case_id': row['id'], 'family': row['family'], 'target': row['target'],
                'segment': token['segment'], 'position': token['full_chat_position'], 'token_id': token['token_id'],
                'token': token['token'], 'target_token': token['target_token'], 'special': token['special'],
                'feature': feature, 'theta': assets[feature]['theta'], **value})
    summary.append(rec)

(ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
with (ROOT / 'all-token-gates.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(all_tokens[0])); writer.writeheader(); writer.writerows(all_tokens)
with (ROOT / 'summary.csv').open('w') as stream:
    fields = ['id', 'target', 'answer', 'source_CH_gate', 'source_Child_gate', 'answer_CH_gate', 'answer_Child_gate']
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in summary:
        writer.writerow(dict(zip(fields, [row['id'], row['target'], row['answer'],
            *row['source_gate_max'].values(), *row['answer_CH_gate_max'].values()])))

matrix = np.array([[*r['source_gate_max'].values(),
    *[np.nan if x is None else x for x in r['answer_CH_gate_max'].values()]] for r in summary])
fig, ax = plt.subplots(figsize=(12, 12.2))
fig.subplots_adjust(left=0.29, right=0.88, top=0.89, bottom=0.10)
cmap = plt.get_cmap('YlGnBu').copy(); cmap.set_bad('#dddddd')
im = ax.imshow(matrix, aspect='auto', vmin=0, vmax=float(np.nanmax(matrix)), cmap=cmap)
names = {'fewshot_upper':'Few-shot uppercase', 'fewshot_lower':'Few-shot lowercase',
         'direct_upper':'Direct uppercase', 'fewshot_upper_title':'Title case input', 'natural_context':'Natural context'}
ax.set_yticks(range(len(summary)), [names[r['family']] + ': ' + r['target'] +
    (' (surname)' if r['id']=='natural_surname' else ' (person)' if r['id']=='natural_person' else '') for r in summary], fontsize=10)
ax.set_xticks(range(4), ['Input word\nCH gate 2334', 'Input word\nChild gate 1372',
    'Emitted CH/ch\nCH gate 2334', 'Emitted CH/ch\nChild gate 1372'], fontsize=11)
ax.xaxis.tick_top(); ax.tick_params(axis='both', length=0)
for i in range(len(summary)):
    for j in range(4):
        x = matrix[i,j]
        ax.text(j, i, 'absent' if np.isnan(x) else f'{x:.2f}', ha='center', va='center', fontsize=10,
            color='white' if not np.isnan(x) and x > np.nanmax(matrix)*0.6 else '#17232c')
for y in [4.5, 9.5, 14.5, 19.5]:
    ax.axhline(y, color='white', linewidth=3)
ax.axvline(1.5, color='white', linewidth=4)
fig.suptitle('Layer 3 CH and Child gates on 22 unchanged model runs', fontsize=17)
fig.colorbar(im, ax=ax, label='Gate value (maximum over tokens in the marked span)', shrink=0.7)
fig.text(0.01, 0.025, 'Gray = the response contained no standalone CH/ch. Input values are span maxima; exact token values are in the report.\n'
    'Answer-token gates are measured while processing an already emitted token. Gate scales are not calibrated across transforms.', fontsize=10)
fig.savefig(ROOT / 'gate-activation.png', dpi=170, bbox_inches='tight')
fig.savefig(ROOT / 'gate-activation.svg', bbox_inches='tight')
plt.close(fig)

def esc(text):
    return html.escape(str(text))

def number(x):
    return 'absent' if x is None else f'{x:.4f}'

trs = []
details = []
for row, raw in zip(summary, D['rows']):
    values = [row['id'], row['target'], row['answer'], *[number(x) for x in row['source_gate_max'].values()],
        *[number(x) for x in row['answer_CH_gate_max'].values()]]
    trs.append('<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in values) + '</tr>')
    token_rows = []
    for t in raw['prompt_tokens'] + raw['answer_tokens']:
        cls = ' class="target"' if t['target_token'] else ''
        vals = [t['segment'], t['full_chat_position'], repr(t['token']),
                *[number(t['features'][f]['gate']) for f in F],
                *[number(t['features'][f]['raw']) for f in F]]
        token_rows.append('<tr' + cls + '>' + ''.join('<td>' + esc(v) + '</td>' for v in vals) + '</tr>')
    details.append('<details><summary>' + esc(row['id']) + '</summary><h3>Exact user prompt</h3><pre>' +
        esc(raw['user']) + '</pre><h3>Native answer</h3><pre>' + esc(row['answer']) +
        '</pre><table><thead><tr><th>Segment</th><th>Position</th><th>Token</th><th>CH gate</th><th>Child gate</th><th>CH raw</th><th>Child raw</th></tr></thead><tbody>' +
        ''.join(token_rows) + '</tbody></table></details>')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH and Child MOLT activation</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#182a35;margin:32px auto;max-width:1250px;padding:0 20px;background:#fafbfc}h1,h2,h3{line-height:1.2}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{text-align:left;border:1px solid #dce3e8;padding:8px;white-space:pre-wrap}th{background:#eaf1f5}pre{padding:14px;background:#eaf1f5;white-space:pre-wrap}details{margin:15px 0;padding:12px;border:1px solid #cdd9e0;background:white}summary{cursor:pointer;font-weight:650}.target{background:#fff1cd}img{max-width:100%;height:auto}.tablewrap{overflow-x:auto}.note{padding:14px;border-left:4px solid #497b96;background:#eef4f7}</style>
<h1>CH and Child MOLT activation</h1><p>Qwen3-4B · Layer 3 · transforms 2334 and 1372 · 19 October 2026</p>
<p>Twenty-two fixed prompts; native greedy answers followed by observation-only cached replay. Every next-token prediction matched. Both attention and the native MLP were unchanged.</p>
<div class="note">The Child gate activates on child/Child while the proposed broad CH gate is silent on those input words. The CH gate activates on several other CH words. This is an activation pattern consistent with an absorption candidate, not a causal demonstration of absorption.</div>
<h2>Input and answer token gates</h2><p>Numbers are maxima over the target word span or standalone CH/ch answer span. Gray means no standalone CH/ch occurred in the answer, not an inactive gate. In split words, inspect the exact token values below.</p>
<img src="gate-activation.png" alt="Gate values at input words and emitted CH or ch answer tokens">
<h2>All prompt summaries</h2><div class="tablewrap"><table><thead><tr><th>Prompt variant</th><th>Input</th><th>Actual answer</th><th>Input CH gate</th><th>Input Child gate</th><th>CH/ch answer CH gate</th><th>CH/ch answer Child gate</th></tr></thead><tbody>''' + ''.join(trs) + '''</tbody></table></div>
<h2>Timing and measurement</h2><p>A gate at an emitted CH token is measured when the model processes that already emitted token; it is not the activation that predicted CH. The prompt-final position predicts the first output token. Gate input is the residual after attention 3, before post-attention layer normalization, standardized by the released MOLT statistics. Gates use FP32 projection and the released JumpReLU threshold. Contributions are observed, never applied to model output. Transform IDs and token positions are zero-based.</p>
<p>CH gate threshold: ''' + str(assets[F[0]]['theta']) + '''; Child gate threshold: ''' + str(assets[F[1]]['theta']) + '''. Raw values below threshold produce exact zero gates. Native model uses BF16. Gate magnitudes are not calibrated between transforms.</p>
<h2>Exact prompts and complete token traces</h2>''' + ''.join(details) + '''
<h2>Files</h2><p><a href="summary.json">Summary JSON</a> · <a href="summary.csv">Summary CSV</a> · <a href="all-token-gates.csv">All token values</a> · <a href="verification.json">Verification</a> · <a href="remote-export/artifacts/measurements.json">Complete measurements and pinned provenance</a></p></html>'''
(ROOT / 'index.html').write_text(page)
verification = {'status':'passed','cases':len(summary),'features':F,'native_generation_and_observation_replay_token_identical':True,
    'all_gate_values_recomputed_from_raw_and_released_threshold':True,'all_contribution_norms_checked':True,
    'source_and_selected_weight_hashes_verified':True,'download_manifest_hashes_verified':True,
    'native_mlp_and_attention_unmodified':True,'gate_measurements':len(all_tokens),
    'standalone_CH_answer_cases':sum(bool(r['answer_CH_tokens']) for r in summary),
    'any_gate_active_on_standalone_CH_answer':any(any(v and v>0 for v in r['answer_CH_gate_max'].values()) for r in summary),
    'capped_outputs':[r['id'] for r in summary if r['stop_reason']!='eos']}
(ROOT / 'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
for row in summary:
    print(row['id'],repr(row['answer']),row['source_gate_max'],row['answer_CH_gate_max'])
