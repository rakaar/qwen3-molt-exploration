"""Readable three-part readout figure and a complete vocabulary-score CSV."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
tokenizer_path = Path(sys.argv[3]).resolve()
data = json.loads((root / 'artifacts/results.json').read_text())
audit = json.loads((out / 'audit.json').read_text())
assert data['status'] == 'complete' and audit['status'] == 'pass'
tokenizer = json.loads(tokenizer_path.read_text())
vocabulary = {int(i): piece for piece, i in tokenizer['model']['vocab'].items()}
for token in tokenizer['added_tokens']:
    vocabulary[token['id']] = token['content']
with np.load(root / 'artifacts/readout-arrays.npz', allow_pickle=False) as a:
    scores = a['scores'].copy()
    ranks = np.stack([np.searchsorted(np.sort(-row), -row, side='left') + 1 for row in scores])
with (out / 'all-vocabulary-scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['token_id', 'tokenizer_piece', *[name + '_score' for name in data['parts']],
                     *[name + '_rank' for name in data['parts']]])
    for token_id in range(data['vocabulary_size']):
        writer.writerow([token_id, vocabulary.get(token_id, f'<unmapped-id:{token_id}>'),
                         *[float(scores[i, token_id]) for i in range(3)],
                         *[int(ranks[i, token_id]) for i in range(3)]])
with (out / 'target-token-scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['token', *data['parts']])
    for row in data['target_rows']:
        writer.writerow([row['token'], *[row['scores'][part] for part in data['parts']]])

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK JP']
fig = plt.figure(figsize=(13.0, 8.7), facecolor='white')
grid = fig.add_gridspec(2, 3, height_ratios=[3.8, 2.1], hspace=.31, wspace=.18)
fig.suptitle('L29–T2375: the full output, the removed part, and the remainder',
             x=.045, y=.975, ha='left', fontsize=17, weight='bold')
fig.text(.045, .929, 'Prompt: the eleventh month of the year is   |   Final input token: ·is, position 7', fontsize=11)
fig.text(.045, .901, 'Same output matrix and frozen native final RMSNorm for all three columns. Scores are not probabilities.',
         fontsize=10, color='#475569')
titles = ['1  Full MLP output', '2  Removed MOLT contribution', '3  MLP after removal']
subtitles = ['M', 'c = gate × operator output', 'BF16(M − c)']
for index, part in enumerate(data['parts']):
    ax = fig.add_subplot(grid[0, index])
    ax.axis('off')
    ax.text(0, 1.07, titles[index], transform=ax.transAxes, fontsize=12, weight='bold')
    ax.text(0, 1.00, subtitles[index], transform=ax.transAxes, fontsize=10, color='#64748b')
    top = data['rankings'][part]['highest'][:10]
    rows = [[str(i + 1), row['token'].replace(' ', '·').replace('\n', '\\n'), f"{row['score']:+.3f}"]
            for i, row in enumerate(top)]
    table = ax.table(cellText=rows, colLabels=['Rank', 'Token', 'Score'], colWidths=[.14, .60, .26],
                     cellLoc='left', colLoc='left', bbox=[0, 0, 1, .93])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#e2e8f0')
        if i == 0:
            cell.set_facecolor('#f1f5f9')
            cell.set_text_props(weight='bold', color='#334155')
        else:
            token = top[i - 1]['token'].strip().lower()
            if token == 'november': cell.set_facecolor('#fff1df')
            elif token == 'december': cell.set_facecolor('#e5f1ff')
            else: cell.set_facecolor('white')
            if j == 2: cell.set_text_props(ha='right')

ax = fig.add_subplot(grid[1, :])
ax.axis('off')
ax.text(0, 1.06, 'November and December tokens — all three parts', transform=ax.transAxes,
        fontsize=12, weight='bold')
rows = [[row['token'].replace(' ', '·'), *[f"{row['scores'][part]:+.3f}" for part in data['parts']]]
        for row in data['target_rows']]
table = ax.table(cellText=rows, colLabels=['Token', 'Full MLP', 'Removed part', 'After removal'],
                 cellLoc='center', colLoc='center', bbox=[0, .06, 1, .86])
table.auto_set_font_size(False)
table.set_fontsize(11.5)
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('#e2e8f0')
    if i == 0:
        cell.set_facecolor('#f1f5f9')
        cell.set_text_props(weight='bold', color='#334155')
    else:
        cell.set_facecolor('#fff1df' if i <= 2 else '#e5f1ff')
fig.text(.045, .065, '· denotes a leading space. Top tables show the 10 highest-scoring tokens across the full vocabulary.',
         fontsize=9.8, color='#475569')
fig.text(.045, .039, 'The remainder includes the actual BF16 rounding. Incoming residual and later-layer computations are outside these readouts.',
         fontsize=9.8, color='#475569')
fig.subplots_adjust(left=.045, right=.97, top=.832, bottom=.118)
for suffix in ['png', 'svg']:
    fig.savefig(out / f'three-part-readout.{suffix}', dpi=180)

reference = json.loads((root / 'reference.json').read_text())
actual_margins = {}
for name in ['native', 'gate_off']:
    p = reference[name]['steps'][0]['probe_probabilities']
    actual_margins[name] = float(np.log(p['84223']) - np.log(p['89011']))
summary = {'prompt': data['prompt'], 'position': 7, 'normalization_scale': data['normalization_scale'],
           'target_rows': data['target_rows'], 'direct_lowercase_margin': audit['november_minus_december_margin'],
           'actual_model_lowercase_margin': actual_margins,
           'actual_model_margin_change': actual_margins['gate_off'] - actual_margins['native'],
           'direct_margin_change': audit['november_minus_december_margin']['after_removal'] - audit['november_minus_december_margin']['full_mlp'],
           'vocabulary_rows': data['vocabulary_size'], 'audit': 'pass'}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
lines = ['# Three-part L29 MLP vocabulary readout', '',
         'Exact raw prompt: `the eleventh month of the year is`; final input position 7, token ` is`.', '',
         'The three primary vectors are native L29 MLP output M, positive intended L29-T2375 contribution c, and the actual edited output BF16(M - c). The removed vector c is shown with its original positive sign; the intervention was -c.', '',
         'Every vector uses the same native final normalization: `W_out @ (gamma * vector / sqrt(mean(native_final_residual^2) + epsilon))`. The scalar denominator is derived from the whole native residual after layer 35, at the same token. The learned final RMSNorm gamma is included. No vector is independently normalized.', '',
         'These are direct vocabulary scores, not probabilities or complete-model answers. They omit the incoming residual contribution and the computations in later layers.', '',
         '![Three-part vocabulary readout](three-part-readout.png)', '',
         '| Token | Full MLP | Removed MOLT | After removal |', '|---|---:|---:|---:|']
for row in data['target_rows']:
    token = row['token'].replace(' ', '·')
    lines.append(f"| `{token}` | " + ' | '.join(f"{row['scores'][part]:+.6f}" for part in data['parts']) + ' |')
lines += ['', 'The capitalized November token ranks first in the full MLP readout; the capitalized December token ranks first after subtraction. The removed contribution favors November and mildly disfavors December in this context. Its highest-scoring tokens also include fragments such as ` nov`, ` ele` and ` novel`; this does not establish a general numerical operation.', '',
          f"For lowercase tokens, the direct MLP November-minus-December margin changes from {summary['direct_lowercase_margin']['full_mlp']:+.6f} to {summary['direct_lowercase_margin']['after_removal']:+.6f}. The actual complete-model margin changes from {actual_margins['native']:+.6f} to {actual_margins['gate_off']:+.6f}. The direct readout matches the direction of the effect, but is not a quantitative explanation of the whole forward-pass change. Downstream computation and normalization changes remain to be separated.", '',
          f"The ideal FP32 decomposition is additive to within {data['ideal_additivity_max_absolute_error']:.8f} over the full vocabulary. The actual BF16 remainder differs from the ideal remainder, with maximum vocabulary-score difference {data['actual_bf16_score_closure_max_absolute_error']:.8f}. The primary display uses the actual BF16 remainder.", '',
          f"The native MLP vector and native final RMSNorm output reproduced exactly. CPU auditing checked {audit['sampled_output_rows']} saved output-matrix rows, including the highest and lowest tokens for all parts and the four target tokens. All passed. The full {data['vocabulary_size']:,}-row readouts are retained in the verified array archive and exported CSV.", '',
          '[All vocabulary scores and ranks](all-vocabulary-scores.csv) · [Target-token scores](target-token-scores.csv) · [Audit](audit.json)', '',
          'The CSV tokenizer_piece field preserves raw tokenizer vocabulary strings; unmapped output rows are labeled explicitly. All ranks are competition ranks, with ties assigned the same rank.', '']
for part, title in zip(data['parts'], titles):
    lines += ['', '## ' + title, '', '| Rank | Token | Score |', '|---:|---|---:|']
    for index, row in enumerate(data['rankings'][part]['highest']):
        lines.append(f"| {index + 1} | `{row['token'].replace(' ', '·')}` | {row['score']:+.6f} |")
lines += ['', '## Lowest-scoring tokens for the removed contribution', '', '| Token | Score |', '|---|---:|']
for row in data['rankings']['removed_molt']['lowest']:
    lines.append(f"| `{row['token'].replace(' ', '·')}` | {row['score']:+.6f} |")
(out / 'README.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'figure': str(out / 'three-part-readout.png'), 'vocabulary_rows': data['vocabulary_size'],
                  'actual_model_margin_change': summary['actual_model_margin_change'],
                  'direct_margin_change': summary['direct_margin_change']}))
