"""Compare original native-space MOLT contributions; no model inference or final RMSNorm."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/2026-11-03-eleven-contribution-similarity'
OUT.mkdir(exist_ok=True)
case_roots = {
    'Month': ROOT / 'comparisons/2026-11-02-eleven-contribution-readout/remote-export/eleven-contribution-readout-20261102',
    'Alphabet': ROOT / 'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/alphabet',
    'Roman': ROOT / 'comparisons/2026-11-02-eleven-crossdomain-readouts/remote-export/eleven-crossdomain-readouts-20261102/roman',
}
vectors, records = [], []
for name, root in case_roots.items():
    result = json.loads((root / 'artifacts/results.json').read_text())
    array_path = root / 'artifacts/readout-arrays.npz'
    assert hashlib.sha256(array_path.read_bytes()).hexdigest() == result['arrays_sha256']
    with np.load(array_path, allow_pickle=False) as a:
        contribution = a['vectors'][1].copy()
    reference = json.loads((root / 'reference.json').read_text())
    reference_path = root / 'reference-vectors.npz'
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == reference['vectors_sha256']
    with np.load(reference_path, allow_pickle=False) as a:
        j = list(a['positions']).index(result['position'])
        gate = float(a['gate'][0, j])
        operator = a['operator'][0, j].copy()
        assert gate > 0
        assert np.array_equal(contribution, -a['intended_delta'][0, j])
        assert np.array_equal(contribution, np.float32(gate) * operator)
    c64, op64 = contribution.astype(np.float64), operator.astype(np.float64)
    c_norm, op_norm = float(np.linalg.norm(c64)), float(np.linalg.norm(op64))
    assert abs(c_norm - gate * op_norm) / c_norm < 1e-6
    vectors.append(c64)
    records.append({'name': name, 'prompt': result['prompt'], 'position': result['position'], 'token': ' is',
                    'gate': gate, 'operator_l2': op_norm, 'contribution_l2': c_norm,
                    'source_arrays': str(array_path.relative_to(ROOT)), 'source_arrays_sha256': result['arrays_sha256']})
vectors = np.stack(vectors)
unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
similarity = unit @ unit.T
assert np.allclose(similarity, similarity.T, atol=1e-12)
assert np.allclose(np.diag(similarity), 1, atol=1e-12)
np.fill_diagonal(similarity, 1.)
names = list(case_roots)
report = {'feature': 'L29-T2375', 'date_local': '2026-11-03', 'names': names,
          'vector_definition': 'Positive intended contribution c = gate(z) * ((z @ V @ U) * output_std), where z = (native_residual - input_mean) / input_std.',
          'space': 'Native MLP output/residual-stream units, before BF16 edit rounding and before final RMSNorm or vocabulary readout.',
          'extra_rmsnorm_applied': False, 'molt_input_output_standardizers_retained': True,
          'matrix_compares': 'Original full prompt-specific contribution vectors, rather than fitted shared-component projections.',
          'similarity': similarity.tolist(), 'magnitudes': records,
          'cosine_note': 'Cosine itself divides by vector L2 lengths. A positive scalar gate changes magnitude but not direction or cosine.',
          'limits': 'Three selected prompts with the same ordinal wording; alignment is not proof of a general ordinal mechanism or causal importance.'}
(OUT / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
np.savez_compressed(OUT / 'contribution-vectors.npz', names=np.array(names), contributions=vectors,
                    cosine_similarity=similarity)
with (OUT / 'cosine-similarity.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['prompt', *names])
    for name, row in zip(names, similarity): writer.writerow([name, *row])
with (OUT / 'magnitudes.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['name', 'gate', 'operator_l2', 'contribution_l2'])
    writer.writeheader()
    for row in records: writer.writerow({key: row[key] for key in writer.fieldnames})

fig, (ax, table_ax) = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={'width_ratios': [1, 1.55]})
fig.patch.set_facecolor('white')
fig.suptitle('L29–T2375: contribution similarity and magnitude', x=.04, y=.985, ha='left', fontsize=17, weight='bold')
fig.text(.04, .905, 'Full gate × operator output at the final input token ·is; native MLP-output space', fontsize=10.5, color='#475569')
heat = ax.imshow(similarity, vmin=0, vmax=1, cmap='Blues')
ax.set_xticks(range(3), names)
ax.set_yticks(range(3), names)
ax.set_title('Cosine similarity', fontsize=12, pad=12)
for i in range(3):
    for j in range(3): ax.text(j, i, f'{similarity[i, j]:.4f}', ha='center', va='center', color='white', fontsize=13, weight='bold')
fig.colorbar(heat, ax=ax, fraction=.045, pad=.04, ticks=[0, .5, 1])
table_ax.axis('off')
table_ax.text(0, .90, 'Magnitude at each prompt', fontsize=12, weight='bold', transform=table_ax.transAxes)
rows = [[r['name'], f"{r['gate']:.3f}", f"{r['operator_l2']:.3f}", f"{r['contribution_l2']:.3f}"] for r in records]
table = table_ax.table(cellText=rows, colLabels=['Prompt', 'Gate', 'Operator L2', 'Contribution L2'],
                      cellLoc='center', colLoc='center', colWidths=[.23, .15, .29, .33], bbox=[0, .26, 1, .56])
table.auto_set_font_size(False)
table.set_fontsize(10.5)
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('#cbd5e1')
    cell.set_facecolor('#eef4fb' if i == 0 else 'white')
    if i == 0: cell.set_text_props(weight='bold')
table_ax.text(0, .14, 'Contribution L2 ≈ gate × operator L2', fontsize=10, color='#475569', transform=table_ax.transAxes)
fig.text(.04, .078, 'MOLT input/output standardization is retained. No extra RMSNorm or output-matrix projection is applied here.', fontsize=9.6, color='#475569')
fig.text(.04, .035, 'Cosine compares directions; the table separately reports their Euclidean (L2) lengths. · denotes a space.', fontsize=9.6, color='#475569')
fig.subplots_adjust(left=.085, right=.98, top=.80, bottom=.19, wspace=.28)
for suffix in ['png', 'svg']: fig.savefig(OUT / f'contribution-similarity.{suffix}', dpi=180)
print(json.dumps({'matrix': similarity.tolist(), 'magnitudes': records, 'figure': str(OUT / 'contribution-similarity.png')}))
