"""Render the saved cross-prompt trial results without rerunning inference."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    out = parser.parse_args().result_directory
    r = json.loads((out / 'artifacts/results.json').read_text())
    c = json.loads((out / 'artifacts/random-control-results.json').read_text())
    audit = json.loads((out / 'local-verification.json').read_text())
    assert audit['status'] == 'passed'
    primary = r['protocol']['primary_pair']
    full = [t for t in r['trials'] if t['method'] == 'full_contribution']
    target = sorted([t for t in full if t['pair'] == primary], key=lambda t: t['strength'])
    at32 = next(t for t in target if t['strength'] == 32)
    gate32 = next(t for t in r['trials'] if t['pair'] == primary and t['method'] == 'gate_only' and t['strength'] == 32)
    best = max(target, key=lambda t: t['P6'])
    gaussians = [t for t in c['trials'] if t['kind'] == 'norm_matched_gaussian']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=.73, bottom=.24, left=.08, right=.98, wspace=.20)
    strengths = [t['strength'] for t in target]
    axes[0].plot(strengths, [100*t['P6'] for t in target], 'o-', color='#2563eb', label="P('6')", linewidth=2)
    axes[0].plot(strengths, [100*t['P8'] for t in target], 'o-', color='#d97706', label="P('8')", linewidth=2)
    axes[0].axvline(32, color='#94a3b8', linestyle=':', linewidth=1)
    axes[0].set_title('Explicit ant donor → implicit webs receiver')
    axes[0].set_ylabel('Next-token probability')
    references = [
        (primary, 'Explicit ant → implicit webs', '#2563eb', '-'),
        ('explicit_spider_ant', 'Explicit ant → explicit spider', '#059669', '-'),
        ('web_anthill', 'Implicit anthills → implicit webs', '#64748b', ':'),
        ('silk_spider_explicit_ant', 'Explicit ant → implicit silk webs', '#9333ea', '--'),
    ]
    for pair, label, color, style in references:
        rows = sorted([t for t in full if t['pair'] == pair], key=lambda t: t['strength'])
        axes[1].plot([t['strength'] for t in rows], [100*t['P6'] for t in rows],
                     marker='o', color=color, linestyle=style, linewidth=2, label=label)
    axes[1].set_title("P('6'): prompt-pair comparisons")
    for ax in axes:
        ax.set(xlabel='Full-contribution patch strength', xlim=(-1, 65), ylim=(-2, 103))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(color='#e2e8f0')
        ax.legend(fontsize=9, loc='best')
    fig.suptitle('L3–T2309: explicit ant donor → implicit spider question',
                 x=.08, ha='left', y=.96, fontsize=18, fontweight='bold')
    fig.text(.08, .865, 'Small Invertebrate and Insect Token Gate · rank 32 · Qwen3-4B · attention unfrozen', color='#475569')
    fig.text(.08, .81, 'Donor: " ant" (token 8) → receiver: " webs" (token 12), at original layer-3 MLP output.', color='#475569')
    fig.text(.08, .14, f"At 32×: P('6') = {100*at32['P6']:.6f}%, P('8') = {100*at32['P8']:.6f}%; generated answer: {c['strength32_generation']!r}.")
    fig.text(.08, .095, 'Silk-webs rephrasing uses only prespecified strengths; lines connect measured points, not additional runs.', fontsize=9, color='#475569')
    fig.text(.08, .055, 'One MOLT and one token edited per trial. These answer probabilities do not establish semantic concept replacement.', fontsize=9, color='#475569')
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'cross-prompt-strength-sweep.{ext}', dpi=190)
    plt.close(fig)
    esc = html.escape
    pair_sections = []
    for p in r['protocol']['pairs']:
        rows = sorted([t for t in full if t['pair'] == p['id']], key=lambda t: t['strength'])
        table = ''.join(f"<tr><td>{t['strength']}</td><td>{100*t['P6']:.7f}%</td><td>{100*t['P8']:.7f}%</td><td>{esc(t['top_token'])}</td><td>{t['intended_delta_norm']:.4f}</td></tr>" for t in rows)
        pair_sections.append(f"<h2>{esc(p['id'])}</h2><p><b>Receiver:</b> {esc(p['receiver'])}<br><b>Donor:</b> {esc(p['donor'])}</p><table><tr><th>Strength</th><th>P(6)</th><th>P(8)</th><th>Top token</th><th>Added-vector norm</th></tr>{table}</table>")
    controls = ''.join(f"<tr><td>{esc(t['kind'])}</td><td>{esc(str(t.get('donor_id', t.get('seed','—'))))}</td><td>{100*t['P6']:.7f}%</td><td>{100*t['P8']:.7f}%</td><td>{esc(t['top_token'])}</td></tr>" for t in c['trials'])
    gate_rows = [t for t in r['trials'] if t['pair'] == primary and t['method'] in ['gate_only','ablate_receiver']]
    gate_table = ''.join(f"<tr><td>{esc(t['method'])}</td><td>{t['strength']}</td><td>{100*t['P6']:.7f}%</td><td>{100*t['P8']:.7f}%</td></tr>" for t in gate_rows)
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>L3–T2309 cross-prompt experiment</title>
    <style>body{{max-width:1120px;margin:40px auto;padding:0 22px;font:16px/1.65 system-ui;color:#172033}}h1,h2{{line-height:1.3}}img{{width:100%}}table{{border-collapse:collapse;width:100%;margin-bottom:25px}}th,td{{padding:8px 12px;border-bottom:1px solid #dbe1eb;text-align:left}}pre{{background:#eef3fb;padding:18px;white-space:pre-wrap}}.result{{background:#edf6ff;padding:20px;border-radius:12px}}code{{background:#eef3fb}}</style>
    <h1>Can an explicit ant donor steer an implicit spider question?</h1><p>2026-10-13 · Qwen3-4B · layer 3 · MOLT 2309 · rank 32 · unfrozen eager attention · original MLP retained.</p>
    <div class="result"><b>Fixed 32× full-contribution result:</b> P(6) = {100*at32['P6']:.7f}%; P(8) = {100*at32['P8']:.7f}%. Actual generated answer: <b>{esc(c['strength32_generation'])}</b>.<br>Highest measured primary P(6): {100*best['P6']:.7f}% at strength {best['strength']}.<br><b>Gate-only control at 32×:</b> P(6) = {100*gate32['P6']:.7f}%.</div>
    <h2>Exact intervention</h2><p>Dashboard feature name: <b>Small Invertebrate and Insect Token Gate</b>. Donor token <code>" ant"</code>, zero-based position 8. Receiver token <code>" webs"</code>, position 12. Indices include the chat template.</p>
    <pre>original receiver MLP output[12] += strength × (donor contribution[8] − receiver contribution[12])
    contribution = scalar gate × input-dependent low-rank output
    receiver gate = {at32['receiver_gate']}; donor gate = {at32['donor_gate']:.9f}
    exact implementation: z = (pre-normalization residual − checkpoint input_mean) / input_std
    contribution = gate(z) × ((z @ V_2309) @ U_2309) × checkpoint output_std</pre>
    <p>The donor and receiver vectors are computed from clean runs. U/V weights are unchanged. The vector is added to the original raw layer-3 MLP output before residual addition. No MLP replacement and no attention freezing occur. The shared output-standardizer mean is not added as a per-feature contribution.</p>
    <img src="cross-prompt-strength-sweep.svg" alt="Cross-prompt strength sweep">
    {''.join(pair_sections)}
    <h2>Primary gate-only and ablation controls</h2><p>The gate-only patch adds <code>strength × (donor gate − receiver gate) × receiver low-rank output</code>. It uses the receiver's UVx rather than the donor's UVx. Its successful 32× next-token result shows that copying the full donor output vector is not required to achieve an answer flip on this prompt.</p><table><tr><th>Method</th><th>Strength</th><th>P(6)</th><th>P(8)</th></tr>{gate_table}</table>
    <h2>Fixed-strength donor and random controls</h2><p>At strength 32, ten Gaussian vectors are matched to the full intervention's norm. {sum(t['P6']>=.99 for t in gaussians)}/10 produced P(6) ≥ 99%; maximum random-control P(6) was {100*max(t['P6'] for t in gaussians):.7f}%. This is a small control sample.</p><table><tr><th>Kind</th><th>Donor / seed</th><th>P(6)</th><th>P(8)</th><th>Top token</th></tr>{controls}</table>
    <h2>Reproducibility and limits</h2><p>Base revision: <code>{r['protocol']['base_revision']}</code>; MOLT revision: <code>{r['protocol']['molt_revision']}</code>. Torch {esc(r['runtime']['torch'])}; Transformers {esc(r['runtime']['transformers'])}; BF16. Feature, prompt pair, strength grid, and fixed strength-32 controls were specified before observing this run. Silk-webs wording is evaluated at fixed strengths 1, 32, and the best primary strength.</p>
    <p>Independent local verification passed: {audit['trial_count']} sweep/control trials plus {audit['additional_control_count']} follow-up controls. Saved full-vocabulary logits were used to recompute every reported probability, and all exported hashes and intervention norms were checked. Answer steering alone does not establish that the model internally replaced a spider representation with an ant representation.</p>
    <p><a href="strength-sweep.csv">All sweep rows</a> · <a href="artifacts/results.json">Raw results</a> · <a href="artifacts/random-control-results.json">Follow-up controls</a> · <a href="local-verification.json">Verification</a> · <a href="experiment-source.tar.gz">Executed source</a></p></html>"""
    if (out / 'cleanup-verification.json').exists():
        cleanup = json.loads((out / 'cleanup-verification.json').read_text())
        assert cleanup['task_instance_absent'] and not cleanup['volumes']
        section = '<h2>GPU cleanup</h2><p>SPAR Vast instance 55604487 and its local disk were deleted after backup and independent verification. No team instances or storage volumes remained at the final inventory check. Ongoing task compute/storage cost: $0/hour. <a href="cleanup-verification.json">Provider verification</a>.</p>'
        report = report.replace('</html>', section + '</html>')
    (out / 'report.html').write_text(report)
    (out / 'summary.json').write_text(json.dumps({'at32':at32,'best_primary':best,
        'gate_only_at32':gate32,'strength32_generation':c['strength32_generation'], 'random_controls_above99':sum(t['P6']>=.99 for t in gaussians),
        'random_control_count':len(gaussians),'trial_count':audit['trial_count']+audit['additional_control_count']},indent=2)+'\n')
    print(out / 'report.html')


if __name__ == '__main__':
    main()
