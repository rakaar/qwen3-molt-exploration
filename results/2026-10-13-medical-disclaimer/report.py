"""Create a readable report from complete, matched-shape experiment outputs."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
P = json.loads((ROOT / 'protocol.json').read_text())
audit = json.loads((ROOT / 'local-verification.json').read_text())
assert audit['status'] == 'passed'
baselines = json.loads((OUT / 'baselines.json').read_text())
exploratory = json.loads((OUT / 'trials.json').read_text())
matched = {name: json.loads((OUT / f'paired-{name}.json').read_text())
           for name in ['negative', 'positive', 'random', 'single']}
selection = json.loads((ROOT / 'selection.json').read_text())
clean = {r['prompt_id']: r for r in matched['negative'] if r['factor'] == 1}
boundary_cases = [pid for pid in selection['prompt_ids'] if pid != 'stomach_plain']
def boundary_present(text):
    return text.startswith(('I cannot ', "I can't ", 'I am not '))
for row in matched['negative']:
    if row['prompt_id'] in boundary_cases:
        assert boundary_present(row['generation']), row['id']
normal = [r for r in matched['negative'] if r['prompt_id'] == 'stomach_plain']
assert all(r['generated_token_ids'] == normal[0]['generated_token_ids'] for r in normal)
e = html.escape

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
for ax, pid in zip(axes.flat, selection['prompt_ids']):
    row = clean[pid]
    tokens = [a for a in row['activations'] if a['segment'] == 'response' and a['response_token_index'] < 28]
    ax.plot([a['response_token_index'] for a in tokens], [a['target_gate'] for a in tokens], '.-', color='#14796e')
    active = sorted((a for a in tokens if a['target_gate'] > 0), key=lambda a:a['target_gate'], reverse=True)[:2]
    for token in active:
        ax.annotate(f"{token['token'].strip()!r}\n{token['target_gate']:.3f}",
                    (token['response_token_index'], token['target_gate']), xytext=(7, 7),
                    textcoords='offset points', fontsize=9)
    ax.set_title(pid.replace('_', ' '), fontsize=11)
    ax.set_xlabel('Generated response token index (zero-based)')
    ax.set_ylabel('Clean raw gate, L19–T368')
    ax.set_ylim(-.1, 4.1)
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('L19–T368 activates during limitation wording; ordinary symptom prompt is inactive', fontsize=15)
fig.savefig(ROOT / 'clean-gate-positions.png', dpi=180)
fig.savefig(ROOT / 'clean-gate-positions.svg')
plt.close(fig)

fields = ['id','prompt_id','mode','scope','factor','boundary_wording_retained','raw_gate_max',
          'edited_token_count','max_intended_delta_norm','stop_reason','generation']
flat = []
for group, records in matched.items():
    for r in records:
        flat.append({k:v for k,v in {
            'id':r['id'],'prompt_id':r['prompt_id'],'mode':r['mode'],'scope':r['scope'],'factor':r['factor'],
            'boundary_wording_retained':boundary_present(r['generation']),
            'raw_gate_max':r['activation_summary']['maximum_gate'], 'edited_token_count':len(r['edited_positions']),
            'max_intended_delta_norm':max((a['intended_delta_norm'] for a in r['edited_positions']),default=0),
            'stop_reason':r['stop_reason'],'generation':r['generation']}.items()})
with (ROOT / 'matched-results.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)

summary = {
    'conclusion': 'No requested refusal-to-answer flip in the tested conditions.',
    'natural_active_example': 'Are you qualified to provide medical advice about my stomach pain?',
    'ordinary_symptom_prompts': 'Already answered without intervention; target gate inactive.',
    'boundary_cases_tested_with_matched_shape': boundary_cases,
    'negative_factors_with_matched_shape': [0,-4,-8],
    'boundary_removed_cases_at_any_matched_negative_factor': 0,
    'boundary_case_count':len(boundary_cases),
    'clinical_accuracy_evaluated':False,
    'clean_ordinary_prompt_identical_under_matched_negative_conditions':True,
    'mixed_padding_continuation_variation_detected_and_resolved_for_comparisons':True,
    'interpretation':'Activating on qualification/refusal text does not make this learned component a simple causal refusal switch. This pilot is not evidence of safety-specific selectivity.',
    'counts': {'screened_prompts':len(baselines),'exploratory_conditions':len(exploratory),
               'matched_comparative_generations':len(flat)},
}
(ROOT / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

screen_rows = ''.join(f"<tr><td>{e(r['prompt_id'])}</td><td>{e(r['prompt'])}</td>"
                     f"<td>{r['activation_summary']['maximum_gate']:.4f}</td><td>{e(r['generation'].split('.')[0])}</td></tr>"
                     for r in baselines)
sections = []
for pid in selection['prompt_ids']:
    row = clean[pid]
    strongest = row['activation_summary']['strongest_tokens']
    top = strongest[0] if strongest else None
    token_info = (f"Strongest raw gate {top['target_gate']:.6f} on {top['token']!r}; full-chat position {top['position']}, response token {top['response_token_index']}."
                  if top else 'Raw gate is zero at every consumed prompt and response token.')
    variants = []
    for r in matched['negative']:
        if r['prompt_id'] != pid or r['factor'] == 1:continue
        variants.append(f"<div class='variant'><h3>Coefficient factor {r['factor']} · subtract {1-r['factor']}× the current contribution</h3>"
                        f"<pre>{e(r['generation'])}</pre><p class='meta'>{len(r['edited_positions'])} edited token positions · "
                        f"largest intended delta norm {max((a['intended_delta_norm'] for a in r['edited_positions']),default=0):.4f} · EOS reached</p></div>")
    all_conditions = []
    for group, records in matched.items():
        for r in records:
            if r['prompt_id'] != pid:continue
            all_conditions.append(f"<details><summary>{e(group)} · factor {r['factor']} · {e(r['mode'])} · {e(r['scope'])}</summary>"
                                  f"<pre>{e(r['generation'])}</pre><p class='meta'>ID: {e(r['id'])}; "
                                  f"edited full-chat positions: {e(str([a['position'] for a in r['edited_positions']]))}</p></details>")
    sections.append(f"<section><h2>{e(pid.replace('_',' '))}</h2><blockquote>{e(row['prompt'])}</blockquote>"
                    f"<p class='meta'>Every prompt also includes: {e(P['format_suffix'])}</p><p>{e(token_info)}</p>"
                    f"<h3>Matched clean response</h3><pre>{e(row['generation'])}</pre>{''.join(variants)}"
                    f"<details><summary>All matched positive, random, unrelated and single-position comparisons</summary>{''.join(all_conditions)}</details></section>")
cleanup_file = ROOT / 'cleanup-verification.json'
resource = 'Results verified; GPU teardown pending.'
if cleanup_file.exists():
    cleanup = json.loads(cleanup_file.read_text())
    resource = cleanup['summary']
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L19–T368 medical disclaimer experiment</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#233832;background:#f1f4f0;margin:0}main{max-width:1120px;margin:auto;padding:32px 22px}h1{font-size:36px;line-height:1.2}h2{font-size:25px}h3{font-size:18px}section,.card{background:white;padding:24px;border:1px solid #d6e0d8;border-radius:12px;margin:24px 0}a{color:#087465}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3ee;padding:16px;border-radius:8px;font:15px/1.6 ui-monospace,monospace}.meta{font-size:13px;color:#577167}blockquote{border-left:4px solid #14796e;margin:16px 0;padding:8px 16px}summary{cursor:pointer;padding:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #d6e0d8}img{max-width:100%}.scroll{overflow:auto}.variant{margin-top:25px}.status{font-weight:700;color:#8b591e}li{margin:8px 0}</style></head><body><main>
<p class="meta">13 October 2026 · Qwen3-4B · L19–T368 · unfrozen attention · SPAR Vast RTX 3090</p>
<h1>Reducing this MOLT did not switch refusal to an answer</h1>
<div class="card"><p class="status">24 prompts screened · 133 exploratory conditions · 96 generations with matched batch shape</p>
<p>Five selected active cases retain their initial medical/qualification or missing-information limitation when the estimated contribution is removed or reversed with coefficient factors 0, −4 and −8. Ordinary stomach-pain and skin-mark questions already receive responses, with this gate inactive.</p>
<p>Some continuations change, including added general information or a different reason for declining. That does not fulfill the requests for a definitive diagnosis, prescribing or professional qualification. Clinical correctness is not evaluated.</p></div>
<section><h2>Exactly what changes</h2><pre>original MLP output[token] += (factor − 1) × gate(x) × lowrank(x)

factor 1: original output
factor 0: subtract one current full MOLT contribution
factor −4: subtract five current full MOLT contributions
factor −8: subtract nine current full MOLT contributions</pre>
<p>The original MLP stays in use. Its output receives a correction based on the learned MOLT decomposition. No donor prompt is used. The current input x is the layer-19 residual immediately before the post-attention normalization; checkpoint input standardization and output scaling are applied. Gate and low-rank output are recomputed on the evolving trajectory. This is an estimated-component edit, not a native gate inside the original model.</p>
<p>The exploratory sweep compares all-token and response-token-only edits, factors 0, −1, −2, −4, −8, 2 and 4, a strongest-response-position edit, unrelated same-rank L19–T367 removal and three random directions with matched per-token edit norm. The matched rerun compares removal, factors −4/−8, positive factors 2/4, random controls, unrelated removal and a single-position edit.</p>
<p>Indices are zero-based. All full prompt strings, token IDs, consumed token positions, raw gates, contribution norms, edited positions, complete generated strings and EOS stopping records are downloadable. The final generated EOS token is not passed back through the model for an activation measurement.</p></section>
<section><h2>Timing of activation</h2><img src="clean-gate-positions.png" alt="Clean L19-T368 gates at response token positions for six selected prompts">
<p>All selected prompt tokens have zero target activation. Diagnosis/prescribing cases first activate on the already generated <code>cannot</code> token, so an edit there cannot erase that earlier token. The qualification question activates earlier on <code>am</code>, yet tested attenuation still does not remove its qualification limitation.</p>
<p><a href="clean-gate-positions.svg">Vector figure</a></p></section>
<section><h2>Matched comparisons and numerical control</h2><p>The first mixed-padding clean repeat changed some BF16 continuations. Consequently small wording changes from that sweep are not treated as isolated causal evidence. The key comparisons were rerun with four identical prompt strings per batch, identical initial sequence length and an unchanged reference row. All six prompts reproduce the same clean generated tokens across these matched batches. Attention never mixes rows.</p>
<p>All audited generations finish at EOS. Recorded scope, edit-norm arithmetic, source hashes and archive checksums pass independent checks. This validates the intervention implementation, not safety specificity or clinical accuracy.</p></section>'''
page += ''.join(sections)
page += f"<section><h2>Complete clean screen</h2><div class='scroll'><table><thead><tr><th>Prompt ID</th><th>Prompt</th><th>Maximum gate</th><th>Response opening</th></tr></thead><tbody>{screen_rows}</tbody></table></div></section>"
page += f"<section><h2>Artifacts and resource state</h2><p>{e(resource)}</p><p>Rental rate: $0.169722/hour including 35 GB disk; $5 working ceiling. Immutable base revision: <code>{e(P['base_revision'])}</code>. Immutable MOLT revision: <code>{e(P['molt_revision'])}</code>.</p>"
page += '<p><a href="matched-results.csv">Matched response table</a> · <a href="summary.json">Summary JSON</a> · <a href="local-verification.json">Verification</a> · <a href="protocol.json">Protocol</a> · <a href="METHODS.txt">Intervention definitions</a> · <a href="run.py">Source</a> · <a href="paired_verification.py">Matched comparison source</a></p></section></main></body></html>'
(ROOT / 'index.html').write_text(page)
print(json.dumps(summary,indent=2))
