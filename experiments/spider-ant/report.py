"""Render an auditable scientific report for the single-layer spider-ant pilot."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_directory', type=Path)
    out = parser.parse_args().result_directory
    r = json.loads((out / 'artifacts/results.json').read_text())
    audit = json.loads((out / 'local-verification.json').read_text())
    controls = json.loads((out / 'artifacts/random-control-results.json').read_text()) if (out / 'artifacts/random-control-results.json').exists() else None
    features, rows = r['protocol']['features'], r['trials']
    e = html.escape
    colors = ['#2563eb', '#059669', '#d97706', '#9333ea', '#dc2626', '#0891b2', '#64748b', '#94a3b8']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7), sharey=True)
    fig.subplots_adjust(top=.80, bottom=.25, left=.08, right=.97, wspace=.12)
    for ax, site, title in zip(axes, ['cue_end', 'readout'], ['Clue token: webs (position 12)', 'Final prompt token (position 29)']):
        for f, color in zip(features, colors):
            selected = sorted([x for x in rows if x['pair'] == 'web_anthill' and x['layer'] == f['layer'] and x['feature'] == f['id'] and x['site'] == site and x['method'] == 'full_contribution'], key=lambda x: x['strength'])
            ax.plot([x['strength'] for x in selected], [100*x['P6'] for x in selected], color=color, marker='o', markersize=3, linewidth=2, label=f"L{f['layer']}–T{f['id']}" + (' · lexical control' if f['role'].startswith('lexical') else ''))
        ax.axhline(r['baselines']['web_anthill_donor']['P6'] * 100, color='#475569', linestyle=':', linewidth=1.3)
        ax.set(xlabel='Contribution-swap strength', title=title, ylim=(-2, 103))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(color='#e2e8f0')
    axes[0].set_ylabel('Next-token P(6)')
    fig.suptitle('Can one MOLT redirect the web-weaving answer from 8 to 6?', fontsize=18, x=.08, ha='left', y=.96)
    fig.text(.08, .855, 'Qwen3-4B · one layer and one position per trial · full donor contribution swap · attention unfrozen', color='#475569')
    fig.text(.08, .16, 'Dotted line: ant-donor P(6). Other tested positions (have, ?) and all control trials are included in the report.', color='#475569')
    fig.text(.08, .135, 'Interpretation: the strongest edit fails on held-out wording; 1 of 20 equal-norm random directions also gives >99.99% P(6).', color='#9a3412', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.08, .025), ncol=4, frameon=False, fontsize=9)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'single-layer-steering.{ext}', dpi=180)
    plt.close(fig)

    def maps(name):
        return {(x['layer'], x['id']): x for x in r['activation_maps'][name]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    labels = [f"L{x['layer']}–T{x['id']}" for x in features]
    y = np.arange(len(features))
    for ax, a, b, title in zip(axes, ['explicit_spider_ant_receiver', 'web_anthill_receiver'], ['explicit_spider_ant_donor', 'web_anthill_donor'], ['Explicitly named spider vs ant', 'Inferred animal: webs vs anthills']):
        ma, mb = maps(a), maps(b)
        ax.barh(y-.18, [ma[(f['layer'],f['id'])]['site_gates']['cue_end'] for f in features], height=.34, color='#2563eb', label='Spider receiver')
        ax.barh(y+.18, [mb[(f['layer'],f['id'])]['site_gates']['cue_end'] for f in features], height=.34, color='#d97706', label='Ant donor')
        ax.set(yticks=y, yticklabels=labels, xlabel='MOLT gate activation at clue token', title=title)
        ax.grid(axis='x', color='#e2e8f0')
        ax.legend(fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / 'candidate-gates.png', dpi=180)
    fig.savefig(out / 'candidate-gates.svg')
    plt.close(fig)

    def candidates(pair, feature=None, site=None, method='full_contribution'):
        return [x for x in rows if x['pair']==pair and x['method']==method and
                (feature is None or (x['layer'],x['feature'])==(feature['layer'],feature['id'])) and (site is None or x['site']==site)]
    primary_semantic = [x for x in candidates('web_anthill') if not x['role'].startswith('lexical')]
    best = max(primary_semantic, key=lambda x:x['P6'])
    best_cue = max([x for x in primary_semantic if x['site']=='cue_end'], key=lambda x:x['P6'])
    explicit_best = max([x for x in candidates('explicit_spider_ant') if not x['role'].startswith('lexical')], key=lambda x:x['P6'])
    summary = {'best_primary_semantic':best, 'best_primary_semantic_at_clue':best_cue, 'trial_count':len(rows), 'baselines':r['baselines'], 'heldout_at_primary_selected_conditions':[]}
    feature_table = ''
    for f in features:
        key=(f['layer'],f['id'])
        explicit_rec, explicit_don=maps('explicit_spider_ant_receiver')[key], maps('explicit_spider_ant_donor')[key]
        inferred_rec, inferred_don=maps('web_anthill_receiver')[key], maps('web_anthill_donor')[key]
        href=f"https://amsterdam.taild80f17.ts.net/layers/layer-{f['layer']}/autointerp-layer/features/{f['id']}.html"
        values=[f'<a href="{href}">L{f["layer"]}–T{f["id"]}</a>',e(f['label']),e(f['role']),f"{explicit_rec['site_gates']['cue_end']:.3f} / {explicit_don['site_gates']['cue_end']:.3f}",f"{inferred_rec['site_gates']['cue_end']:.3f} / {inferred_don['site_gates']['cue_end']:.3f}"]
        feature_table += '<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
    primary_table = ''
    for f in features:
        for site in r['protocol']['sites']:
            winner=max(candidates('web_anthill',f,site), key=lambda x:x['P6'])
            gate=max(candidates('web_anthill',f,site,'gate_only'), key=lambda x:x['P6'])
            full=candidates('web_anthill',f,site,'full_mlp_positive_control')[0]
            values=[f"L{f['layer']}–T{f['id']}",e(site),str(winner['receiver_position']),e(winner['receiver_token']),f"{winner['strength']:g}",f"{100*winner['P6']:.6f}%",f"{100*winner['P8']:.6f}%",f"{100*gate['P6']:.6f}%",f"{100*full['P6']:.6f}%"]
            primary_table+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
        chosen=max(candidates('web_anthill',f),key=lambda x:x['P6'])
        held=[x for x in candidates('silk_colony',f,chosen['site']) if x['strength']==chosen['strength']]
        if held:
            summary['heldout_at_primary_selected_conditions'].append({'primary':chosen,'heldout':held[0]})
    held_table=''
    for item in summary['heldout_at_primary_selected_conditions']:
        a,b=item['primary'],item['heldout']
        values=[f"L{a['layer']}–T{a['feature']}",e(a['site']),f"{a['strength']:g}",f"{100*a['P6']:.6f}%",f"{100*b['P6']:.6f}%",f"{100*b['P8']:.6f}%"]
        held_table+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
    baseline_table=''
    for name,b in r['baselines'].items():
        if b['generation'] is not None:
            baseline_table+='<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in [name,b['text'],b['generation'],f"{100*b['P6']:.6f}%",f"{100*b['P8']:.6f}%"])+'</tr>'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    control_note=''
    if controls:
        control_table=''
        for pair in ['web_anthill','silk_colony']:
            selected=[x for x in controls['trials'] if x['pair']==pair]
            random=[x for x in selected if x['kind']=='norm_matched_gaussian']
            for kind in ['repeat_full_contribution','matched_strength_gate_only']:
                x=next(x for x in selected if x['kind']==kind)
                values=[e(pair),e(kind),f"{100*x['P6']:.6f}%",f"{100*x['P8']:.6f}%",e(x['top_token'])]
                control_table+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
            x=max(random,key=lambda x:x['P6'])
            values=[e(pair),f"Highest-P(6) of {len(random)} norm-matched random directions",f"{100*x['P6']:.6f}%",f"{100*x['P8']:.6f}%",e(x['top_token'])]
            control_table+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
        all_controls=''
        for x in controls['trials']:
            values=[e(x['pair']),e(x['kind']),e(str(x.get('donor_id',x.get('seed','')))),f"{100*x['P6']:.6f}%",f"{100*x['P8']:.6f}%",f"{100*x['P4']:.6f}%",e(x['top_token']),f"{x['intended_delta_norm']:.3f}"]
            all_controls+='<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
        control_note=f'''<h2>Matched-strength and donor-specificity controls</h2>
<p>These follow-up controls were selected after the initial sweep and are reported separately. All use L11–T39 at the clue token, strength 16. Twenty Gaussian directions are scaled to the full intervention's norm for each prompt. On the primary question, one of 20 random directions also produced P(6) &gt;99.99%. The explicitly named ant donor did not reproduce the anthill-donor effect. Thus this is a demonstrated numerical answer flip, not a validated selective spider-to-ant concept replacement.</p>
<div class="scroll"><table><tr><th>Prompt pair</th><th>Control</th><th>P(6)</th><th>P(8)</th><th>Top token</th></tr>{control_table}</table></div>
<p>The same full-contribution edit fails on the held-out wording. A gate-only edit at the same strength succeeds there but fails on the primary wording. No identical tested layer/site/method/strength produced &gt;99% P(6) on both implicit questions in the matched comparisons.</p>
<details><summary>All 60 follow-up control trials</summary><div class="scroll"><table><tr><th>Prompt</th><th>Control</th><th>Donor / seed</th><th>P(6)</th><th>P(8)</th><th>P(4)</th><th>Top token</th><th>Delta norm</th></tr>{all_controls}</table></div></details>
<p><a href="artifacts/random-control-results.json">Control results</a> · <a href="artifacts/control-logits.npz">Full control logits</a></p>'''
    cleanup_note=''
    if (out/'cleanup-verification.json').exists():
        c=json.loads((out/'cleanup-verification.json').read_text())
        cleanup_note=f"<h2>Resource cleanup</h2><p>SPAR instance {c['instance_id']} destroyed after verified local backup. Provider reads confirmed no remaining instances or volumes. Ongoing task GPU/storage cost: $0/hour. <a href='cleanup-verification.json'>Cleanup record</a>.</p>"
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Single-layer MOLT spider-ant pilot</title><style>
body{{font:17px/1.6 system-ui;color:#172033;background:#f4f7fb;margin:0}}main{{max-width:1180px;margin:auto;padding:36px 24px}}h1{{font-size:40px;line-height:1.15}}h2{{margin-top:38px}}img{{width:100%;height:auto}}.card{{background:white;padding:24px;border-radius:16px;margin:22px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dde4ef}}a{{color:#2563eb}}code,pre{{background:#eaf0f8;border-radius:5px;padding:3px 6px}}pre{{white-space:pre-wrap;padding:18px}}
</style><main><p>12 October 2026 · SPAR Model Psychology · Qwen3-4B</p><h1>Can a single MOLT replace an inferred spider with an ant?</h1>
<p>The target is to redirect the answer from <b>8</b> to <b>6</b>. Every trial edits one MOLT in one layer at one token position. Attention and the original MLPs recompute normally.</p>
<div class="card"><img src="single-layer-steering.svg" alt="P6 versus strength for each single-layer MOLT"><p>Best sampled semantic-candidate result on the primary question: <b>L{best['layer']}–T{best['feature']}</b>, <b>{e(best['site'])}</b>, position <b>{best['receiver_position']}</b>, strength <b>{best['strength']:g}</b>; P(6) <b>{100*best['P6']:.6f}%</b>, P(8) <b>{100*best['P8']:.6f}%</b>. At the clue token specifically, the best P(6) was <b>{100*best_cue['P6']:.6f}%</b> with L{best_cue['layer']}–T{best_cue['feature']} at strength {best_cue['strength']:g}.</p><p><b>Interpretation:</b> this flip is prompt-sensitive and is also achievable by one tested random direction of equal magnitude. It does not validate a selective spider-to-ant replacement.</p></div>
<h2>Baseline questions and answers</h2><p>Primary receiver: <code>{e(r['protocol']['pairs'][0]['receiver'])}</code><br>Ant donor: <code>{e(r['protocol']['pairs'][0]['donor'])}</code></p>
<p>The model generated the expected 8 and 6 before any manipulation. The explicit spider/ant pair is a diagnostic; the silk-web/ant-colony paraphrase is held out from strength selection. Numeric probabilities below are for the single next-token digits 6 and 8, not task accuracy.</p><div class="scroll"><table><tr><th>Condition</th><th>Prompt</th><th>Generated answer</th><th>P(6)</th><th>P(8)</th></tr>{baseline_table}</table></div>
<h2>Candidate representations</h2><p>Fresh dashboard coverage: 34 layers and 80,703 observed MOLTs. Georg's Hugging Face release was freshly checked at revision <code>{r['protocol']['molt_revision']}</code>: 74 files, weights/configs/README, no label bundle. Dashboard labels and activation examples were used. These candidates are broad animal or arthropod representations; they are not established dedicated spider or ant features. Lexical controls are explicitly marked. The winning L11–T39 is one rank-512 transform, within a 2,560-dimensional residual space.</p>
<div class="card"><img src="candidate-gates.svg" alt="Candidate gate activations for explicit and inferred animals"></div><div class="scroll"><table><tr><th>MOLT</th><th>Dashboard label</th><th>Role</th><th>Explicit spider / ant gate</th><th>Implicit webs / anthills gate</th></tr>{feature_table}</table></div>
<h2>Exact intervention</h2><pre>Full contribution swap:
edited MLP output = original receiver MLP output
                  + strength * (donor MOLT contribution - receiver MOLT contribution)

MOLT contribution = gate(x) * low-rank output(x)
low-rank output(x) = standardized input(x) * V * U * saved output standard deviation

Gate-only control:
delta = strength * (donor gate - receiver gate) * receiver low-rank output

Receiver ablation:
delta = -strength * receiver MOLT contribution</pre>
<p>The donor's entire input-dependent contribution is copied in the full swap. U/V weights and the model input are not replaced. The shared output-standardizer mean is not added twice. MOLT gates use the pre-normalization residual, matching the checkpoint's input location; deltas are added to the raw MLP output before residual addition. All layer and position indices are zero based.</p>
<p>Primary clue: token 12, <b>" webs"</b>, paired with donor token 12, <b>"ills"</b>, the last subtoken of <b>anthills</b> (tokens 11–12 are " anth" + "ills"). Also tested separately: <b>have</b> at 13, <b>?</b> at 14, and the final prompt token at 29. The last site is a later readout intervention and does not by itself establish a spider-to-ant intermediate reasoning swap. The donor contains no answer digit.</p>
<h2>All primary layer/site conditions</h2><p>Each full-swap row reports the highest P(6) among prespecified strengths 0, 0.5, 1, 2, 4.5, 8, 16, 32. Gate-only controls test 1 and 4.5. The full-MLP donor replacement at strength 1 is a positive control for whether that layer/site can carry the answer change; it is not a MOLT intervention. Inactive gates produce a zero edit and unchanged logits.</p><div class="scroll"><table><tr><th>MOLT</th><th>Site</th><th>Position</th><th>Token</th><th>Best strength</th><th>Full-swap P(6)</th><th>Full-swap P(8)</th><th>Best gate-only P(6)</th><th>Full-MLP control P(6)</th></tr>{primary_table}</table></div>
<h2>Explicit spider-to-ant diagnostic</h2><p>With the animals explicitly named, the best semantic-candidate edit is <b>L{explicit_best['layer']}–T{explicit_best['feature']}</b>, site <b>{e(explicit_best['site'])}</b>, receiver token <b>{explicit_best['receiver_position']}</b>, strength <b>{explicit_best['strength']:g}</b>: P(6) <b>{100*explicit_best['P6']:.6f}%</b>, P(8) <b>{100*explicit_best['P8']:.6f}%</b>. This diagnostic tests an explicit animal-name intervention rather than the inferred web-weaving animal.</p>
<h2>Held-out paraphrase at primary-selected settings</h2><p>For each MOLT, the layer/site/strength is selected using only the primary question, then replayed on the silk-web/ant-colony pair. Additional fixed strengths 1 and 4.5 are retained in the raw data but not used to select this validation table.</p><div class="scroll"><table><tr><th>MOLT</th><th>Site</th><th>Strength</th><th>Primary P(6)</th><th>Held-out P(6)</th><th>Held-out P(8)</th></tr>{held_table}</table></div>
{control_note}<h2>Verification and limits</h2><p>{audit['trial_count']} sweep trials and {audit.get('additional_control_count',0)} additional controls verified independently from exported full-vocabulary logits and contribution vectors. Hashes, zero/inactive edit recovery, unchanged upstream states and MLP inputs, one-layer/one-position editing, residual-addition boundary, attention methods, and restored clean output checks passed. Broad labels, activation and a successful answer edit do not independently prove a specific semantic representation or establish general task accuracy. The octopus diagnostic is retained as a model response rather than biological leg-count validation: octopuses have arms.</p>
<p><a href="strength-sweep.csv">All trials CSV</a> · <a href="artifacts/results.json">Raw results, tokens and protocol</a> · <a href="artifacts/next-token-logits.npz">Full logits</a> · <a href="artifacts/intervention-vectors.npz">Edit vectors</a> · <a href="artifacts/baseline-captures.npz">Baseline activations</a> · <a href="local-verification.json">Independent audit</a> · <a href="single-layer-steering.pdf">Figure PDF</a> · <a href="../../experiments/spider-ant/protocol.json">Preregistered protocol</a></p>{cleanup_note}</main></html>'''
    (out/'report.html').write_text(report)
    print(json.dumps({'best_primary':{k:best[k] for k in ['layer','feature','site','strength','P6','P8']},'best_clue':{k:best_cue[k] for k in ['layer','feature','site','strength','P6','P8']},'trials':len(rows)},indent=2))


if __name__=='__main__':
    main()
