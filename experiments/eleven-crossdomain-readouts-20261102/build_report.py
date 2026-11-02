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
case = data['case']
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
fig = plt.figure(figsize=(13.0, 9.0), facecolor='white')
grid = fig.add_gridspec(2, 3, height_ratios=[3.8, 2.1], hspace=.31, wspace=.18)
fig.suptitle(f"L29–T2375: {case['family']} — three-part readout",
             x=.045, y=.975, ha='left', fontsize=17, weight='bold')
fig.text(.045, .929, f"Prompt: {data['prompt']}   |   Final input token: ·is, position {data['position']}", fontsize=11)
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
    rows = [[str(int(ranks[index, row['token_id']])), row['token'].replace(' ', '·').replace('\n', '\\n'), f"{row['score']:+.3f}"]
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
            if token in {x.strip().lower() for x in case['target_tokens'][:2]}: cell.set_facecolor('#fff1df')
            elif token in {x.strip().lower() for x in case['target_tokens'][2:]}: cell.set_facecolor('#e5f1ff')
            else: cell.set_facecolor('white')
            if j == 2: cell.set_text_props(ha='right')

ax = fig.add_subplot(grid[1, :])
ax.axis('off')
ax.text(0, 1.06, case['target_title'], transform=ax.transAxes,
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
if case['id'] == 'roman':
    fig.text(.045, .013, 'Roman XII starts with ·x then ii here. ·x / ·X are prefixes, not complete XII tokens.', fontsize=9.5, color='#475569')
fig.subplots_adjust(left=.045, right=.97, top=.832, bottom=.118)
for suffix in ['png', 'svg']:
    fig.savefig(out / f'three-part-readout.{suffix}', dpi=180)

reference = json.loads((root / 'reference.json').read_text())
target_by_text = {row['token']: row for row in data['target_rows']}
pair = case['contrast_pair']
token_ids = [target_by_text[text]['token_id'] for text in pair]
actual_margins = {}
for name in ['native', 'gate_off']:
    p = reference[name]['steps'][0]['probe_probabilities']
    actual_margins[name] = float(np.log(p[str(token_ids[0])]) - np.log(p[str(token_ids[1])]))
summary = {'case': case, 'prompt': data['prompt'], 'position': data['position'],
           'normalization_scale': data['normalization_scale'], 'target_rows': data['target_rows'],
           'direct_contrast_margin': audit['contrast_margin'],
           'actual_model_first_token_margin': actual_margins,
           'actual_model_first_token_margin_change': actual_margins['gate_off'] - actual_margins['native'],
           'direct_margin_change': audit['contrast_margin']['after_removal'] - audit['contrast_margin']['full_mlp'],
           'vocabulary_rows': data['vocabulary_size'], 'audit': 'pass',
           'native_response': reference['native']['answer'], 'saved_gate_off_response': reference['gate_off']['answer']}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')

import html


def token_markup(text):
    return '<code>' + html.escape(text.replace(' ', '·').replace('\n', '\\n')).replace('|', '&#124;') + '</code>'


lines = ['# ' + case['family'] + ': three-part L29 MLP readout', '',
         f"Exact raw prompt: `{data['prompt']}`; final input position {data['position']}, token ` is`.", '',
         'The three primary vectors are native L29 MLP output M, positive intended L29-T2375 contribution c, and the actual edited output BF16(M - c). The contribution is displayed with its original positive sign; the gate-off intervention subtracts it.', '',
         'All three use `W_out @ (gamma * vector / sqrt(mean(native_final_residual^2) + epsilon))`. The denominator comes from the whole native final residual for this prompt, after layer 35 at the same position. Learned final RMSNorm gamma is included. The three vectors share that scale and are not independently normalized.', '',
         'These are direct vocabulary scores, not probabilities or complete-model answers. Incoming residual and later-layer computations are excluded from these three MLP-part projections.', '',
         case['token_note'], '',
         '![Three-part vocabulary readout](three-part-readout.png)', '',
         '| Token | Full MLP | Removed MOLT | After removal |', '|---|---:|---:|---:|']
for row in data['target_rows']:
    lines.append('| ' + token_markup(row['token']) + ' | ' + ' | '.join(f"{row['scores'][part]:+.6f}" for part in data['parts']) + ' |')
top_before = data['rankings']['full_mlp']['highest'][0]['token']
top_after = data['rankings']['after_removal']['highest'][0]['token']
lines += ['', 'The highest full-MLP readout token is ' + token_markup(top_before) + '; the highest remainder token is ' + token_markup(top_after) + '.', '',
          'For the displayed contrast ' + token_markup(pair[0]) + ' versus ' + token_markup(pair[1]) +
          f", the direct margin changes from {audit['contrast_margin']['full_mlp']:+.6f} to {audit['contrast_margin']['after_removal']:+.6f}. The saved complete-model first-token margin changes from {actual_margins['native']:+.6f} to {actual_margins['gate_off']:+.6f}. This first-token comparison does not give complete-answer probabilities.", '',
          f"The native MLP vector reproduced exactly on the new forward pass, and the native final RMSNorm output reconstructed exactly. CPU auditing passed over {audit['sampled_output_rows']} saved output-matrix rows, including the highest and lowest tokens for every part and all target tokens.", '',
          f"The ideal FP32 decomposition is additive to within {data['ideal_additivity_max_absolute_error']:.8f} over the full vocabulary. The primary display uses the actual BF16 remainder. Its maximum score difference from ideal M-minus-c is {data['actual_bf16_score_closure_max_absolute_error']:.8f}.", '',
          'The direct readout is a diagnostic of these saved vectors. It does not by itself explain the full generated-answer change or establish a general add-one mechanism.', '',
          '[All vocabulary scores and ranks](all-vocabulary-scores.csv) · [Target scores](target-token-scores.csv) · [CPU audit](audit.json)', '',
          'The CSV preserves raw tokenizer-piece strings and explicitly labels unmapped output rows. Ranks use competition ranking, so exact ties share a rank.', '',
          '## Recorded native continuation', '', '> ' + reference['native']['answer'].strip(), '',
          '## Recorded final-input-only gate-off continuation', '', '> ' + reference['gate_off']['answer'].strip(), '',
          'These historical continuations hit the 32-token cap and can drift. This readout analysis adds a fresh native prefill to reproduce the vectors and obtain the contextual normalization scale; it does not rerun generation or gate-off inference.', '']
for part, title in zip(data['parts'], titles):
    lines += ['', '## ' + title, '', '| Rank | Token | Score |', '|---:|---|---:|']
    for row in data['rankings'][part]['highest']:
        rank = int(ranks[data['parts'].index(part), row['token_id']])
        lines.append(f"| {rank} | " + token_markup(row['token']) + f" | {row['score']:+.6f} |")
lines += ['', '## Lowest-scoring tokens for the removed contribution', '', '| Token | Score |', '|---|---:|']
for row in data['rankings']['removed_molt']['lowest']:
    lines.append('| ' + token_markup(row['token']) + f" | {row['score']:+.6f} |")
(out / 'README.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'case': case['id'], 'figure': str(out / 'three-part-readout.png'),
                  'direct_contrast_margin': audit['contrast_margin'], 'actual_first_token_margin': actual_margins}))

