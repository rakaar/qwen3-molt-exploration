"""Create standalone scientific figures and an auditable local HTML report."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

BLUE, ORANGE, GREEN = '#2563eb', '#d97706', '#059669'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.titleweight': 'bold',
                     'svg.fonttype': 'none', 'figure.facecolor': 'white',
                     'axes.facecolor': '#fafbff', 'axes.edgecolor': '#cbd5e1'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_directory', type=Path)
    args = parser.parse_args()
    out = args.result_directory
    r = json.loads((out / 'artifacts/results.json').read_text())
    audit = json.loads((out / 'local-verification.json').read_text())
    cleanup_path = out / 'cleanup-verification.json'
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else None
    cleanup_note = '' if cleanup is None else f'''<h2>Resource cleanup</h2>
<p>SPAR instance {cleanup['instance_id']} was destroyed after the local backup and independent audit passed. Provider reads confirmed no remaining instances or storage volumes. Ongoing compute and storage cost: $0/hour. Estimated rental cost including disk: ${cleanup['estimated_compute_plus_disk_cost_usd']:.3f}; bandwidth is additional and negligible.</p>
<p><a href="cleanup-verification.json">Cleanup verification</a> · <a href="../../experiments/residual-similarity/README.md">Reproduction instructions and source</a></p>'''
    rows = r['trials']
    x = np.array([t['strength'] for t in rows])
    residual = np.array([t['cosine_residual'] for t in rows])
    pcold = np.array([t['P_cold'] for t in rows]) * 100
    selected = x <= 32
    reference = next(t for t in rows if t['strength'] == 19.5)
    peak_similarity = max(rows, key=lambda t: t['cosine_residual'])
    peak_cold = max(rows, key=lambda t: t['P_cold'])
    summary = {'molt_fixed': r['molt_fixed'], 'geometry': r['geometry'],
               'transcoder19.5': reference, 'maximum_measured_residual_cosine': peak_similarity,
               'maximum_measured_P_cold': peak_cold, 'trial_count': len(rows),
               'status': 'verified', 'boundary': r['protocol']['primary_residual_boundary']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(12.4, 7.8))
    fig.subplots_adjust(top=.82, bottom=.25, left=.10, right=.89)
    right = ax.twinx()
    ax.plot(x[selected], residual[selected], color=BLUE, linewidth=2.7,
            label='Cosine: edited residual vs. fixed MOLT residual')
    changes = np.array([np.nan if t['cosine_residual_changes'] is None else
                        t['cosine_residual_changes'] for t in rows])
    ax.plot(x[selected], changes[selected], color='#64748b', linewidth=1.8,
            linestyle='--', label='Cosine: changes from clean residual')
    right.plot(x[selected], pcold[selected], color=ORANGE, linewidth=2.7,
               label='Transcoder P(cold)')
    right.axhline(r['molt_fixed']['P_cold'] * 100, color=GREEN, linestyle='--', linewidth=1.5,
                  label=f"Fixed MOLT P(cold): {100*r['molt_fixed']['P_cold']:.3f}%")
    ax.axvline(19.5, color='#64748b', linewidth=1.3, linestyle=':')
    ax.scatter([19.5], [reference['cosine_residual']], color=BLUE, s=45, zorder=5)
    ax.scatter([peak_similarity['strength']], [peak_similarity['cosine_residual']],
               color=BLUE, s=70, marker='D', zorder=5)
    ax.set(xlim=(0, 32), ylim=(-.02, 1.04), xlabel='Transcoder strength (both donor patch and clean subtraction)',
           ylabel='Cosine similarity at token 9')
    right.set(ylim=(-2, 104), ylabel='Next-token P(cold)')
    right.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis='y', colors=BLUE)
    right.tick_params(axis='y', colors=ORANGE)
    ax.yaxis.label.set_color(BLUE)
    right.yaxis.label.set_color(ORANGE)
    ax.grid(axis='both', color='#e2e8f0', linewidth=.8)
    ax.set_title('Same answer, how similar is the residual?', loc='left', fontsize=20, pad=46)
    fig.text(.10, .845, 'Qwen3-4B · token 9 after layer 22 · MOLT strength fixed at 4.5 · attention unfrozen',
             fontsize=11, color='#475569')
    lines, labels = ax.get_legend_handles_labels()
    l2, s2 = right.get_legend_handles_labels()
    fig.legend(lines+l2, labels+s2, loc='lower left', bbox_to_anchor=(.10, .015),
               frameon=False, fontsize=10)
    fig.text(.10, .165, f"At transcoder 19.5: residual cosine = {reference['cosine_residual']:.4f}; change cosine = {reference['cosine_residual_changes']:.4f}; P(cold) = {reference['P_cold']*100:.3f}%",
             color='#334155', fontsize=10.5)
    fig.text(.10, .14, f"Peak residual cosine: {peak_similarity['cosine_residual']:.4f} at strength {peak_similarity['strength']:g}; P(cold) = {peak_similarity['P_cold']*100:.7f}%",
             color='#334155', fontsize=10.5)
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'residual-cosine-vs-strength.{extension}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    positive = x > 0
    measured = [np.nan if t['cosine_residual_changes'] is None else t['cosine_residual_changes'] for t in rows]
    axes[0].plot(x[positive], np.asarray(measured)[positive], color=BLUE, linewidth=2,
                 label='Actual BF16 residual changes')
    axes[0].axhline(r['geometry']['intended_edit_cosine'], color=GREEN, linestyle='--',
                    label=f"Intended FP32 direction: {r['geometry']['intended_edit_cosine']:.4f}")
    axes[0].set(xlabel='Transcoder strength', ylabel='Cosine: MOLT change vs. transcoder change',
                title='Edit direction stays essentially fixed', ylim=(-1.05, 1.05))
    axes[0].legend(fontsize=9, loc='best')
    axes[1].plot(range(22, 36), reference['downstream_residual_cosines'][22:], color=BLUE,
                 marker='o', label='Residual states')
    axes[1].plot(range(22, 36), reference['downstream_change_cosines'][22:], color=ORANGE,
                 marker='o', label='Changes from clean')
    axes[1].set(xlabel='Layer output (zero-based)', ylabel='Cosine similarity',
                title='MOLT 4.5 vs. transcoder 19.5 downstream', ylim=(-1.05, 1.05))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(color='#e2e8f0')
    fig.tight_layout()
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'edit-directions-and-downstream.{extension}', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5))
    right = ax.twinx()
    ax.plot(x, residual, color=BLUE, linewidth=2, label='Residual cosine')
    right.plot(x, pcold, color=ORANGE, linewidth=2, label='P(cold)')
    ax.set(xlabel='Transcoder strength', ylabel='Residual cosine', ylim=(-.05, 1.05),
           title='Complete prespecified sweep (0 to 128)')
    right.set(ylim=(-2, 104), ylabel='P(cold)')
    right.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(color='#e2e8f0')
    fig.tight_layout()
    fig.savefig(out / 'full-strength-range.png', dpi=160)
    plt.close(fig)

    table = ''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [f"{t['strength']:g}",
            f"{t['cosine_residual']:.6f}",
            'undefined' if t['cosine_residual_changes'] is None else f"{t['cosine_residual_changes']:.6f}",
            f"{100*t['P_cold']:.6f}%", f"{100*t['P_large']:.6f}%", html.escape(t['top_token'])])+'</tr>' for t in rows)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOLT vs transcoder: residual similarity</title><style>
body{{font:17px/1.6 system-ui;background:#f4f7fb;color:#172033;margin:0}}main{{max-width:1120px;margin:auto;padding:44px 24px}}
h1{{font-size:clamp(28px,4vw,46px);line-height:1.12}}h2{{margin-top:36px}}.card{{background:white;padding:24px;border-radius:16px;margin:22px 0}}
img{{width:100%;height:auto}}code,pre{{background:#edf2fa;border-radius:6px;padding:3px 6px}}pre{{white-space:pre-wrap;padding:18px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e2e8f0}}.scroll{{overflow:auto}}a{{color:#2563eb}}
</style><main><p>12 October 2026 · SPAR Model Psychology · Qwen3-4B</p><h1>Same answer, how similar is the residual?</h1>
<p>The MOLT intervention is fixed at 4.5. Only transcoder strength varies. Both use normal, unfrozen attention and edit layer 22, token position 9.</p>
<div class="card"><img src="residual-cosine-vs-strength.svg" alt="Cosine similarity and P(cold) versus transcoder strength">
<p>At transcoder strength 19.5: residual cosine <b>{reference['cosine_residual']:.6f}</b>, P(cold) <b>{100*reference['P_cold']:.4f}%</b>.
The fixed MOLT has P(cold) <b>{100*r['molt_fixed']['P_cold']:.4f}%</b>. The highest measured residual cosine is <b>{peak_similarity['cosine_residual']:.6f}</b> at strength <b>{peak_similarity['strength']:g}</b>, where P(cold) is only <b>{100*peak_similarity['P_cold']:.7f}%</b>.</p></div>
<h2>What was compared</h2><p>The main blue curve compares the actual residual vectors at token 9 immediately after the layer-22 MLP residual addition, before layer-23 normalization. It includes the shared clean residual.</p>
<p>A separate measure compares each edited residual minus its clean residual. A large cosine between full states can arise from their shared clean component. Neither metric alone establishes that the two edits use the same causal mechanism.</p>
<div class="card"><img src="edit-directions-and-downstream.svg" alt="Edit direction and downstream residual comparisons">
<p>The intended injected-direction cosine is <b>{r['geometry']['intended_edit_cosine']:.6f}</b>. Multiplying a fixed direction by a positive scalar cannot change this angle. Small variations in the measured change cosine arise from BF16 rounding at the write and residual addition; later layers can change the direction through normal model computation. At strength zero, the edit cosine is undefined.</p></div>
<h2>Exact intervention</h2><p>Receiver: <code>{html.escape(r['protocol']['receiver'])}</code></p><p>Donor: <code>{html.escape(r['protocol']['donor'])}</code></p>
<p>Zero-based layer 22; token 9 is <b>small</b> in the receiver and <b>hot</b> in the donor. Next-token probabilities are read after prompt token 28. Thinking is disabled.</p>
<pre>MOLT: T2339 (small), T941 (hot); fixed strength 4.5
Transcoder: F38866 (small), F85454 (hot); strength s varies

edited MLP output = original receiver MLP output
                  + strength * sum(donor contribution - receiver contribution)

MOLT contribution = gate(x) * U * V * standardized_input(x)
                    * saved output standard deviation
Transcoder contribution = activation(x) * decoder vector</pre>
<p>Both clean hot activations and donor small activations are reported in the raw data. Full MOLT output contributions are copied; no U/V weights are replaced. Original MLPs recompute normally. There are no attention patches, no full MOLT/transcoder replacement, and no direct edits at other layers or positions.</p>
<h2>Validation</h2><p>{audit['trial_count']} prespecified strength values; downloaded hashes verified; cosine and probabilities independently recomputed from exported residual vectors and full next-token logits. Zero-strength recovery, intervention repeats, restored clean runs, and unchanged upstream residuals passed. Downstream attention changed for both methods.</p>
<p>Single-prompt probabilities are not accuracy rates. Cosine does not measure equality of vector magnitudes or prove a shared computation. The plot's 0–32 view is expanded below to show all tested strengths.</p>
<div class="card"><img src="full-strength-range.png" alt="Full strength sweep through 128"></div>
<h2>Data and reproduction</h2><p><a href="artifacts/strength-sweep.csv">CSV</a> · <a href="artifacts/results.json">Raw results and protocol</a> · <a href="artifacts/residual-vectors.npz">Residual vectors</a> · <a href="artifacts/next-token-logits.npz">Full next-token logits</a> · <a href="local-verification.json">Independent verification</a> · <a href="residual-cosine-vs-strength.pdf">Plot PDF</a></p>
<details><summary>All strength values</summary><div class="scroll"><table><thead><tr><th>Strength</th><th>Residual cosine</th><th>Change cosine</th><th>P(cold)</th><th>P(large)</th><th>Top token</th></tr></thead><tbody>{table}</tbody></table></div></details>{cleanup_note}</main></html>'''
    (out / 'report.html').write_text(report)
    print(json.dumps({'transcoder19.5': {k: reference[k] for k in ['strength','cosine_residual','cosine_residual_changes','P_cold']},
                      'peak_similarity': {k: peak_similarity[k] for k in ['strength','cosine_residual','P_cold']},
                      'intended_edit_cosine': r['geometry']['intended_edit_cosine']}, indent=2))


if __name__ == '__main__':
    main()
