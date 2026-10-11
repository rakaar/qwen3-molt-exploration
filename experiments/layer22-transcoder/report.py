"""Create a data-backed scientific plot and standalone follow-up report."""
import argparse
import base64
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def percent(x):
    if 0 < x < .000001:
        return f'{100 * x:.3e}%'
    return f'{100 * x:.5f}%'


def build(root):
    r = json.loads((root / 'artifacts/results.json').read_text())
    assert r['status'] == 'complete'
    groups = {g['name']: g for g in r['groups']}
    best, pair = r['best'], r['best_pair']
    chosen = r['smallest_successful_tested'] or best
    group = groups[chosen['group']]
    molt = next(t for t in r['trials'] if t['group'] == 'molt22-pair')
    cleanup = json.loads((root / 'cleanup.json').read_text())
    audit = json.loads((root / 'artifacts/verification.json').read_text())
    def primary_trial(name, strength):
        return next(t for t in r['trials'] if t['prompt_id'] == 'primary' and
                    t['group'] == name and t['mode'] == 'coupled' and t['strength'] == strength)
    comparison = [('Transcoder: 2 features', primary_trial(pair['group'], s))
                  for s in [1, 4.5, 13.5, 19.5]]
    comparison += [('Transcoder: 6 features', primary_trial(best['group'], s)) for s in [1, 4.5]]
    comparison += [('Transcoder: 6 features, highest measured', best), ('MOLT: 2 transforms', molt)]
    comparison_rows = ''.join(f'<tr><td>{e_name}</td><td>{t["strength"]:.6g}×</td>'
        f'<td>{percent(t["P_cold"])}</td><td>{percent(t["P_large"])}</td>'
        f'<td>{t["delta_norm"]:.4f}</td></tr>' for e_name, t in comparison)
    choices = [best]
    if pair and (pair['group'], pair['mode']) != (best['group'], best['mode']):
        choices.append(pair)
    if (chosen['group'], chosen['mode']) not in [(c['group'], c['mode']) for c in choices]:
        choices.append(chosen)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    palette = ['#6b3fc6', '#d46c26', '#147c79']
    for curve, color in zip(choices, palette):
        points = [t for t in r['trials'] if t['prompt_id'] == 'primary' and
                  t['group'] == curve['group'] and t['mode'] == curve['mode']]
        points.sort(key=lambda x: x['strength'])
        n = groups[curve['group']]['n_features']
        mode = 'coupled' if curve['mode'] == 'coupled' else 'small zeroed'
        label = f'Transcoder: {n} features, {mode}'
        for ax in axes:
            ax.plot([p['strength'] for p in points], [100 * p['P_cold'] for p in points],
                    '.-', markersize=3, linewidth=1.5, color=color, label=label)
    for ax in axes:
        ax.scatter([4.5], [100 * molt['P_cold']], marker='*', s=115,
                   color='#12624c', zorder=5, label='MOLT: 2 transforms, 4.5x (rerun)')
        ax.axhline(99, color='#626a75', linestyle='--', linewidth=1, label='99% threshold')
        ax.set_ylim(-2, 101.5)
        ax.set_xlabel('Patch strength s')
        ax.set_ylabel('P(exact next token cold), %')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_xlim(0, 20)
    axes[0].set_title('Strengths 0–20')
    axes[1].set_xscale('symlog', linthresh=1, base=2)
    axes[1].set_xlim(0, 512)
    axes[1].set_title('Full tested range (log scale after 1x)')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center' if
               matplotlib.__version__.startswith('3.8') else 'lower center',
               bbox_to_anchor=(.5, -.18), fontsize=8, frameon=False)
    fig.savefig(root / 'strength-curves.png', dpi=200, bbox_inches='tight')
    fig.savefig(root / 'strength-curves.svg', bbox_inches='tight')
    plt.close(fig)
    e = html.escape
    feature_rows = []
    for role in ['small', 'hot']:
        for feature in groups[best['group']][role]:
            c = next(x for x in r['candidates'] if x['feature'] == feature and x['role'] == role)
            feature_rows.append(f'<tr><td>{e(role)}</td><td>L22-F{feature}</td>'
                f'<td>{e(c["label"])}</td><td>{c["clean_activation"]:.6g}</td>'
                f'<td>{c["donor_activation"]:.6g}</td><td>'
                f'{"Pair + six-feature group" if feature in group[role] else "Six-feature group"}</td></tr>')
    heldout_rows = []
    for p in r['prompts'][1:]:
        t = next(x for x in r['trials'] if x['prompt_id'] == p['id'])
        gen = next(x for x in r['generation'] if x['prompt_id'] == p['id'])
        heldout_rows.append(f'<tr><td>{e(p["receiver"])}</td><td>{p["position"]}</td>'
            f'<td>{percent(t["P_cold"])}</td><td>{percent(t["P_cold"]+t["P_Cold"])}</td><td>{percent(t["P_large"])}</td>'
            f'<td>{e(gen["text"])}</td></tr>')
    control_rows = []
    for t in r['trials']:
        if t['mode'] == 'control':
            control_rows.append(f'<tr><td>{e(t["group"])}</td><td>{e(t["component"])}</td>'
                f'<td>{t["strength"]:.6g}</td><td>{percent(t["P_cold"])}</td>'
                f'<td>{percent(t["P_large"])}</td><td>{t["delta_norm"]:.6g}</td>'
                f'<td>{e(t["top_token"])}</td></tr>')
    check_rows = ''.join(f'<li>{e(k.replace("_", " "))}: <strong>{"PASS" if v else "FAIL"}</strong></li>'
                         for k, v in r['checks'].items())
    pattern_change = max(x['maximum_pattern_change'] for x in r['attention_probe'][23:])
    image = base64.b64encode((root / 'strength-curves.png').read_bytes()).decode()
    if best['P_cold'] >= .99:
        verdict = 'Yes: a single layer can reach at least 99% P(cold).'
    else:
        verdict = 'No 99% result was found in this finite single-layer search.'
    pair_summary = f'Best two-feature result: {percent(pair["P_cold"])} at {pair["strength"]:.6g}×.' if pair else 'No eligible two-feature group was available.'
    rule = ('delta = s × sum((donor activation − receiver activation) × decoder vector)'
            if chosen['mode'] == 'coupled' else
            'delta = −sum(clean small contributions) + s × sum(hot donor-minus-clean contributions)')
    baseline = r['prompts'][0]['baseline']
    data = json.dumps({'trials': r['trials'], 'groups': r['groups']}, separators=(',', ':')).replace('<', '\\u003c')
    markup = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Layer 22 transcoder: unfrozen attention strength search</title>
<style>
:root{{color-scheme:light;--ink:#14263d;--muted:#54647a;--line:#d8e1ec;--violet:#6b3fc6;--teal:#147c79}}
*{{box-sizing:border-box}}body{{margin:0;background:#f5f8fc;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:35px 24px 80px}}h1{{font-size:clamp(28px,4vw,45px);line-height:1.15}}h2{{margin-top:0}}
.eyebrow{{letter-spacing:.12em;font-size:12px;font-weight:700;color:var(--teal)}}.sub{{color:var(--muted)}}
section,.cards>div{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin:20px 0}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.cards>div{{margin:0}}.big{{font-size:31px;font-weight:750;color:var(--violet)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#edf3fa}}
.scroll{{overflow:auto}}code,pre{{font-family:ui-monospace,monospace}}pre{{white-space:pre-wrap;background:#eef3f9;padding:16px;border-radius:8px}}
img{{max-width:100%;height:auto}}a{{color:var(--violet)}}code,pre{{overflow-wrap:anywhere}}.cards>div{{min-width:0}}select,input{{font:inherit;padding:8px;border:1px solid var(--line);border-radius:6px;max-width:100%}}
.filters{{display:flex;gap:14px;flex-wrap:wrap;margin:15px 0}}.small{{font-size:13px}}.limit{{border-left:4px solid #d68831;padding-left:16px}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:18px}}}}
</style><main>
<a href="../">← Original matched experiment</a>
<p class="eyebrow">FOLLOW-UP · 11 OCTOBER 2026 · QWEN3-4B</p>
<h1>Layer 22 transcoder: attention unfrozen</h1>
<p class="sub">Can single-layer feature patching make the model answer <strong>cold</strong>, as the MOLT layer-22 edit did?</p>
<section><h2>{e(verdict)}</h2><p>{e(pair_summary)}</p>
<p class="limit">This is an exploratory search on one primary prompt. A 99% next-token probability is not 99% accuracy across prompts. The smallest successful tested group is not a proven global minimum.</p></section>
<div class="cards"><div>Highest tested transcoder P(cold)<div class="big">{percent(best['P_cold'])}</div>{groups[best['group']]['n_features']} features · {best['strength']:.6g}× · {e(best['mode'])}</div>
<div>Selected result<div class="big">{percent(chosen['P_cold'])}</div>{group['n_features']} features · {chosen['strength']:.6g}× · {e(chosen['mode'])}</div>
<div>MOLT layer-22 control, rerun<div class="big">{percent(molt['P_cold'])}</div>2 transforms · 4.5× · full contribution patch</div></div>
<section><h2>Goal and patch location</h2><p><strong>Receiver:</strong> {e(r['protocol']['receiver']).replace('small','<mark title="Edited token, position 9">small</mark>')}</p><p><strong>Donor:</strong> {e(r['protocol']['donor']).replace('hot','<mark title="Donor token, position 9">hot</mark>')}</p>
<p>The receiver’s <strong>small</strong> token and the donor’s <strong>hot</strong> token both occupy <strong>position 9</strong>. Only the receiver’s layer-22 MLP output at that position is edited. The primary next-token readout is position 28. All indices are zero-based and include the chat template.</p>
<p>Clean receiver: <strong>{e(baseline['top_token'])}</strong>, P(large) = {percent(baseline['P_large'])}; P(cold) = {100*baseline['P_cold']:.3e}%.</p>
<p>Attention, RMSNorm, and the native MLPs recompute normally. No attention forward method is replaced, no clean MLP output is pinned, and no model weights are changed. The transcoder only supplies fixed feature contribution vectors for additive edits.</p></section>
<section><h2>Direct comparison at layer 22</h2><p>Every row below uses the same model, receiver/donor prompts, token position, and freely recomputed attention. The transcoder changes scalar activation × decoder vector; the MOLT control changes the full gate × low-rank output vector. Their vectors differ, so equal numerical strength does not mean equal output perturbation.</p><div class="scroll"><table><thead><tr><th>Recipe</th><th>Strength s</th><th>P(cold)</th><th>P(large)</th><th>Delta L2 norm</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>
<p><strong>Two-feature recipe:</strong> small L22-F38866; hot L22-F85454. At 13.5×, P(cold) first crosses 99% in this pair’s sampled curve; its measured peak is 99.75234% at 19.5×. At 19.5×, the effective clean-reference small contribution is −18.5×, and the injected hot donor contribution is +19.5×.</p>
<p><strong>Six-feature recipe:</strong> small L22-F38866, L22-F42644, L22-F113890; hot L22-F85454, L22-F122802, L22-F154689. At 4.5× it reaches 99.80232%: effective small multiplier −3.5×, donor hot multiplier +4.5×. The highest sampled P(cold), 99.92866%, occurs at approximately 6.10417×.</p>
<p><strong>MOLT rerun:</strong> small L22-F2339; hot L22-F941; delta = −4.5 × clean small full contribution + 4.5 × donor hot full contribution. P(cold) = {percent(molt['P_cold'])}.</p>
</section>
<section><h2>Feature definitions and exact edit</h2><p>The smallest successful tested group was <strong>{e(chosen['group'])}</strong>, at strength {chosen['strength']:.6g}×; total delta norm {chosen['delta_norm']:.6g}. It is used for the controls and held-out wording below.</p><pre>{e(rule)}

For the two-feature pair specifically:
delta = −s × 5.59375 × decoder[L22-F38866]
        +s × 9.0 × decoder[L22-F85454]
receiver MLP output[position 9] += delta</pre>
<p>Each transcoder contribution is activation × fixed decoder vector. Small features with zero donor activation have an effective clean-reference multiplier of 1 − s in the coupled sweep. Hot features with zero receiver activation receive s × their natural donor activation. These are clean-reference output edits, not clamps on a freshly re-encoded live feature.</p>
<div class="scroll"><table><thead><tr><th>Role</th><th>Feature</th><th>Existing label</th><th>Receiver activation</th><th>Donor activation</th><th>Used in</th></tr></thead><tbody>{''.join(feature_rows)}</tbody></table></div>
<p>P(cold) = <strong>{percent(chosen['P_cold'])}</strong>; P(large) = {percent(chosen['P_large'])}. MOLT comparison delta norm = {molt['delta_norm']:.6g}; equal strength does not imply equal intervention magnitude.</p></section>
<section><h2>Strength curves</h2><img src="data:image/png;base64,{image}" alt="Transcoder next-token cold probability across tested strengths, with the rerun MOLT comparison point"><p>Both plots show the same measured curves. The right plot includes strengths up to 512×. Only measured points and local peak refinements are shown. The MOLT star is one rerun at 4.5×, not a newly measured MOLT curve.</p><a href="strength-curves.svg">Download vector plot</a> · <a href="strength-curves.png">Download PNG</a></section>
<section><h2>Search scope and every trial</h2><p>{len(r['groups'])} feature groups; {len(r['trials'])} total trials including controls, local refinements, and held-out prompts. Candidate labels were taken from the earlier complete available layer-22 explanation export. Eligibility required natural activation differences in the expected direction. Groups were saved before the first manipulated probability was measured.</p>
<p>Tests include the three previously selected layer-22 small features against active thermal donors, individual feature pairs, and activation-ranked groups. The coupled sweep varies small suppression and hot injection together; the second sweep zeroes the published small group once and varies hot injection independently. Three recipe peaks are locally refined.</p>
<div class="filters"><label>Group <select id="group"></select></label><label>Mode <select id="mode"><option value="">All modes</option><option>coupled</option><option>zero_small_vary_hot</option><option>control</option></select></label><label>Prompt <select id="prompt"><option value="primary">Primary</option><option value="">All prompts</option></select></label></div><p id="count" class="small"></p>
<div class="scroll"><table><thead><tr><th>Group</th><th>Mode</th><th>Strength</th><th>P(cold)</th><th>P(large)</th><th>Top token</th><th>Delta norm</th></tr></thead><tbody id="trials"></tbody></table></div></section>
<section><h2>Held-out wording: same selected features and strength</h2><p>Feature membership and strength were held fixed at the selected two-feature, 13.5× recipe. Donor activations were measured afresh for each matched hot prompt. These prompts were not used to tune the selected recipe. All four greedy answers are cold or Cold.; the exact lowercase-token probability does not stay above 99%. The combined column counts only the two next-token IDs cold and Cold, and is not a measure of full-answer accuracy.</p><div class="scroll"><table><thead><tr><th>Receiver</th><th>Edited token</th><th>P(cold)</th><th>P(cold) + P(Cold)</th><th>P(large)</th><th>Greedy answer</th></tr></thead><tbody>{''.join(heldout_rows)}</tbody></table></div></section>
<section><h2>Controls and execution checks</h2><div class="scroll"><table><thead><tr><th>Control/group</th><th>Component</th><th>Strength</th><th>P(cold)</th><th>P(large)</th><th>Delta norm</th><th>Top token</th></tr></thead><tbody>{''.join(control_rows)}</tbody></table></div><ul>{check_rows}</ul>
<p>Maximum measured downstream attention-pattern change after layer 22: {pattern_change:.6g}. The unchanged attention functions recomputed these patterns from the perturbed activations.</p>
<p>Transcoder reconstruction relative L2 error at the operand: receiver {r['reconstruction']['receiver']['relative_l2_error']:.5g}, donor {r['reconstruction']['donor']['relative_l2_error']:.5g}. These diagnostics do not turn the decoder contributions into an exact decomposition of the native MLP.</p></section>
<section><h2>Evidence and reproducibility</h2><p><a href="results.json">All results</a> · <a href="protocol.json">Prespecified protocol</a> · <a href="run.py">Experiment code</a> · <a href="prepare.py">Pinned download code</a> · <a href="download-manifest.json">Verified weight hashes</a> · <a href="verification.json">Independent tensor audit</a> · <a href="results-export.tar.gz">Full captured results and tensors (8.3 MB)</a> · <a href="cleanup.json">GPU cleanup record</a></p>
<p>Base revision: <code>{e(r['protocol']['base_revision'])}</code><br>Transcoder revision: <code>{e(r['protocol']['transcoder_revision'])}</code></p><pre>{e(json.dumps(r['runtime'],indent=2))}</pre>
<p>The native model and MOLT comparison were measured together on this GPU. Prior experiment runtime versions differed; use the rerun control for the within-run comparison. Feature labels are hypotheses, and the search does not establish an exhaustive impossibility result or multi-hop reasoning behavior.</p>
<p>Independent audit: {audit['audited_trace_count']} saved intervention traces passed. Local backup: all ten exported file hashes matched. The SPAR account funded this experiment; all three task rentals are destroyed, with no remaining instance or volume. Estimated total compute and storage: ${cleanup['estimated_all_rentals_compute_storage_usd']:.3f}; this is an elapsed-time estimate, not a final invoice. The first rental was replaced because of poor download throughput; a second empty rental was discarded after an instance-specific temporary credential appeared in diagnostic output. That credential was removed from saved rental records. No account credential was uploaded to the GPU.</p></section>
<script id="data" type="application/json">{data}</script><script>
const data=JSON.parse(document.getElementById('data').textContent), g=document.getElementById('group'), m=document.getElementById('mode'), p=document.getElementById('prompt');
const names=[...new Set(data.trials.map(x=>x.group))];g.add(new Option('All groups',''));for(const name of names)g.add(new Option(name,name));g.value={json.dumps(chosen['group'])};
for(const name of [...new Set(data.trials.map(x=>x.prompt_id))].filter(x=>x!=='primary'))p.add(new Option(name,name));
function draw(){{const rows=data.trials.filter(x=>(!g.value||x.group===g.value)&&(!m.value||x.mode===m.value)&&(!p.value||x.prompt_id===p.value));const body=document.getElementById('trials');body.replaceChildren();for(const x of rows){{const tr=document.createElement('tr');for(const value of [x.group,x.mode,x.strength,(100*x.P_cold).toFixed(5)+'%',(100*x.P_large).toExponential(4)+'%',x.top_token,x.delta_norm.toFixed(5)]){{const td=document.createElement('td');td.textContent=value;tr.append(td)}}body.append(tr)}}document.getElementById('count').textContent=rows.length+' matching trials'}}
for(const field of [g,m,p])field.addEventListener('change',draw);draw();
</script></main></html>'''
    (root / 'report.html').write_text(markup)
    summary = {'verdict': verdict, 'best': best, 'best_pair': pair,
               'selected': chosen, 'selected_group': group, 'molt_control': molt,
               'trial_count': len(r['trials']), 'group_count': len(r['groups'])}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
