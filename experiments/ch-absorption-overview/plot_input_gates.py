"""Combine the original seven rows with frozen L28-T1123 input gates."""
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


FEATURES = ['L3-T2334', 'L3-T1372', 'L4-T153', 'L11-T2226', 'L28-T1123',
            'L29-T2334', 'L32-T1516', 'L34-T929']
LABELS = ['L3–T2334 · CH onset', 'L3–T1372 · Child', 'L4–T153 · ch / new',
          'L11–T2226 · Child / baby', 'L28–T1123 · CH prefix',
          'L29–T2334 · CH onset', 'L32–T1516 · CH culinary', 'L34–T929 · CH prefix']
COLUMNS = [('child', 'child'), ('chin', 'chin'), ('chair', 'chair'),
           ('chief', 'chief'), ('cheef', 'che'), ('cheef', 'ef')]


def build_input_gate_figure(root, out):
    original_path = root / 'results/2026-10-19-ch-crosslayer-activation/input-word-gates.csv'
    added_path = out / 'L28-T1123-input-gates.json'
    with original_path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    added = json.loads(added_path.read_text())
    fields = list(records[0])
    records.extend(added['records'])
    assert len(records) == 96
    lookup = {(r['task'], r['feature'], r['word'], r['token']): r for r in records}
    assert len(lookup) == len(records)
    matrices = {task: np.array([[float(lookup[(task, feature, word, token)]['gate'])
                                for word, token in COLUMNS] for feature in FEATURES])
                for task in ['uppercase', 'lowercase']}
    assert all(m.shape == (8, 6) for m in matrices.values())
    assert all(m[0, 0] == 0 and m[1, 0] > 0 and m[4, 0] > 0 for m in matrices.values())
    assets = out / 'assets'
    assets.mkdir(exist_ok=True)
    with (assets / 'input-word-gates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in matrices:
            for feature in FEATURES:
                for word, token in COLUMNS:
                    writer.writerow(lookup[(task, feature, word, token)])
    vmax = max(float(m.max()) for m in matrices.values())
    cmap = LinearSegmentedColormap.from_list('gates', ['#f1f3f5', '#c8e8ef', '#55afc1', '#06657e'])
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.45), sharey=True, layout='constrained')
    column_labels = [word if word == token else word + '\n[' + token + ']' for word, token in COLUMNS]
    for ax, (task, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        ax.set_title('Few-shot ' + task + ' task', fontsize=14, pad=12)
        ax.set_xticks(np.arange(len(COLUMNS)), column_labels, fontsize=11)
        ax.set_yticks(np.arange(len(FEATURES)), LABELS, fontsize=11)
        ax.set_xticks(np.arange(-.5, len(COLUMNS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(FEATURES), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.6)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.axvline(3.5, color='#6c7b83', lw=1.1)
        for i in range(len(FEATURES)):
            for j in range(len(COLUMNS)):
                gate = matrix[i, j]
                ax.text(j, i, '0' if gate == 0 else f'{gate:.2f}', ha='center', va='center',
                        fontsize=11, color='white' if gate > vmax * .63 else '#172f3b')
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(image, ax=axes, shrink=.9, pad=.025)
    bar.set_label('Natural gate value', fontsize=11)
    fig.suptitle('MOLT gates on the input CH-word tokens', fontsize=17)
    fig.supxlabel('Each cell is one input token.  cheef = che + ef; shown separately.  No intervention.', fontsize=11)
    # Stable vector output makes rebuilds comparable.
    with matplotlib.rc_context({'svg.hashsalt': 'ch-input-gates-eight-features'}):
        fig.savefig(assets / 'input-word-gates.png', dpi=180)
        fig.savefig(assets / 'input-word-gates.svg', metadata={'Date': None})
    plt.close(fig)
    original_stats = json.loads((original_path.parent / 'summary.json').read_text())['features']
    stats = dict(original_stats)
    stats['L28-T1123'] = {'label': added['label'], 'source_tokens_active': 12,
                          'source_tokens_total': 12,
                          'max_source_gate': max(float(r['gate']) for r in added['records']),
                          'child_gates': {task: float(m[4, 0]) for task, m in matrices.items()}}
    proof = {'selected_features': FEATURES, 'input_token_gate_values': len(records),
             'columns': [{'word': w, 'token': t} for w, t in COLUMNS],
             'matrices': {task: m.tolist() for task, m in matrices.items()},
             'original_seven_rows_unchanged': True,
             'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [original_path, added_path]},
             'L28_independent_cpu_gate_max_abs_error': added['provenance']['independent_cpu_gate_max_abs_error'],
             'no_new_inference': True}
    (out / 'input-word-gates-verification.json').write_text(json.dumps(proof, indent=2) + '\n')
    return stats
