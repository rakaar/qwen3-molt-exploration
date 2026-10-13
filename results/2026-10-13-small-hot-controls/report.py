"""Report the requested same-layer random controls and exact MLP replacement."""
import csv
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
R = json.loads((ROOT / 'artifacts/results.json').read_text())
V = json.loads((ROOT / 'verification.json').read_text())
assert V['status'] == 'passed'
rows = R['rows']; by_id = {r['id']: r for r in rows}
labels = {f['id']: f for f in json.loads((ROOT / 'labels.json').read_text())['features']}
selected = by_id['selected_pair']; whole = by_id['whole_mlp_donor_replace']
e = html.escape


def group(name):
    return [r for r in rows if r['group'] == name]


def percent(value):
    return f'{100 * value:.6g}%'


def formula(row):
    if row['id'] == 'whole_mlp_donor_replace':
        return 'MLP_receiver[22,9] = MLP_donor[22,9]'
    if row['mode'] == 'replace':
        return 'MLP_receiver[22,9] = clean MLP_receiver[22,9]'
    return 'MLP_receiver[22,9] += ' + ' + '.join(f'{coefficient:+g} × C_{feature}({source})' for source, feature, coefficient in row['terms']) if row['terms'] else 'No edit'


positive, negative, purple = '#d27832', '#427cba', '#773f8d'
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
for ax in axes.flat:
    ax.set_ylim(-3, 103); ax.set_ylabel('P(cold), % of full vocabulary'); ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
ax = axes[0, 0]
for name, color, offset, marker, label in [('uniform_single_positive', positive, -.12, 'o', '+4.5 × donor contribution'), ('uniform_single_negative', negative, .12, 'v', '−4.5 × receiver contribution')]:
    rr = group(name); ax.scatter([r['sample_index'] + 1 + offset for r in rr], [100 * r['P_cold'] for r in rr], color=color, marker=marker, s=20, alpha=.75, label=label)
ax.scatter([103], [100 * by_id['selected_hot_only']['P_cold']], c=positive, marker='*', s=160, label='Selected hot-only L22–T941')
ax.scatter([105], [100 * by_id['selected_small_only']['P_cold']], c=negative, marker='*', s=160, label='Selected small-only L22–T2339')
ax.set_title('100 uniform random MOLTs, individual edits'); ax.set_xlabel('Random sample index; selected singles at right'); ax.legend(fontsize=8, loc='upper left')
ax.text(.03, .40, 'All 100 sampled MOLTs have zero gates\nat both source tokens. Random edits are zero.', transform=ax.transAxes, fontsize=10, color='#53645d')
ax = axes[0, 1]
for name, color, marker, label in [('uniform_pair', positive, 'o', '−4.5 receiver A +4.5 donor B'), ('uniform_pair_reverse', negative, 'v', '+4.5 receiver A −4.5 donor B')]:
    rr = group(name); ax.scatter([r['sample_index'] + 1 for r in rr], [100 * r['P_cold'] for r in rr], color=color, marker=marker, s=20, alpha=.75, label=label)
ax.scatter([103], [100 * selected['P_cold']], color=purple, marker='*', s=200, label=f"Selected pair: {percent(selected['P_cold'])}")
ax.scatter([105], [100 * by_id['selected_pair_reverse']['P_cold']], color=purple, marker='v', s=80, label='Selected pair, reversed sign')
ax.set_title('100 uniform random pairs, same two-term recipe'); ax.set_xlabel('Random pair index; selected pair at right'); ax.legend(fontsize=8, loc='upper left')
ax.text(.03, .40, 'All random pair edits are zero.\nSee active alternatives below.', transform=ax.transAxes, fontsize=10, color='#53645d')
ax = axes[1, 0]
active = group('active_pair')
ax.scatter(range(1, len(active) + 1), [100 * r['P_cold'] for r in active], color=positive, s=65, label='Gate-active random pairs')
ax.scatter([len(active) + 2], [100 * selected['P_cold']], color=purple, marker='*', s=200, label='Selected pair')
ax.set_xticks(list(range(1, len(active) + 1)) + [len(active) + 2], [f"T{r['terms'][0][1]} / T{r['terms'][1][1]}" for r in active] + ['Selected'])
ax.tick_params(axis='x', labelrotation=25, labelsize=9); ax.set_title(f'{len(active)} distinct pairs with both source gates active'); ax.set_xlabel('Subtracted receiver MOLT / added donor MOLT'); ax.legend(fontsize=8)
ax = axes[1, 1]
refs = [by_id['clean_receiver'], whole, selected]
x = np.arange(len(refs)); width = .34
ax.bar(x - width / 2, [100 * r['P_cold'] for r in refs], width, color=purple, label='cold')
ax.bar(x + width / 2, [100 * r['P_large'] for r in refs], width, color='#4f8962', label='large')
ax.set_xticks(x, ['Clean receiver', 'Exact donor MLP replacement', 'Selected MOLT pair']); ax.tick_params(axis='x', labelsize=9); ax.set_title('Whole native MLP output versus selected contributions'); ax.legend()
fig.suptitle('Qwen3-4B · layer 22, token 9 · strength 4.5 · attention unfrozen\nSelected pair: −4.5 receiver L22–T2339 +4.5 donor L22–T941', fontsize=14)
fig.savefig(ROOT / 'controls-scatter.png', dpi=170); fig.savefig(ROOT / 'controls-scatter.svg'); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for name, color, label in [('uniform_pair', '#839da8', 'Uniform pairs'), ('active_pair', positive, 'Gate-active pairs')]:
    rr = group(name); ax.scatter([r['actual_delta_over_original_mlp_norm'] for r in rr], [100 * r['P_cold'] for r in rr], color=color, alpha=.7, label=label)
ax.scatter([selected['actual_delta_over_original_mlp_norm']], [100 * selected['P_cold']], color=purple, s=170, marker='*', label='Selected pair')
ax.scatter([whole['actual_delta_over_original_mlp_norm']], [100 * whole['P_cold']], c='#4f8962', s=70, marker='D', label='Whole MLP replacement')
ax.set_xlabel('Actual correction norm / clean MLP-output norm'); ax.set_ylabel('P(cold), %'); ax.set_ylim(-3, 103); ax.grid(alpha=.2); ax.legend(); ax.set_title('Equal strength does not imply equal perturbation size')
fig.savefig(ROOT / 'norm-control.png', dpi=170); fig.savefig(ROOT / 'norm-control.svg'); plt.close(fig)

fields = ['id', 'group', 'sample_index', 'edit', 'P_cold', 'P_large', 'top_token', 'nonzero_edit', 'intended_delta_norm', 'actual_delta_norm', 'actual_delta_over_original_mlp_norm', 'terms', 'source_gates', 'logit_row', 'vector_row']
with (ROOT / 'conditions.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for r in rows:
        row = {key: r.get(key) for key in fields}; row['edit'] = formula(r)
        row['terms'] = json.dumps(r['terms']); row['source_gates'] = json.dumps(r.get('source_gates', [])); writer.writerow(row)

summary_groups = ['uniform_single_positive', 'uniform_single_negative', 'uniform_pair', 'uniform_pair_reverse', 'active_single_positive', 'active_single_negative', 'active_pair']
table = ''.join(f"<tr><td>{e(name.replace('_', ' '))}</td><td>{V['groups'][name]['count']}</td><td>{V['groups'][name]['nonzero_edits']}</td><td>{percent(V['groups'][name]['maximum_P_cold'])}</td><td>{V['groups'][name]['at_least_99_percent']}</td><td>{V['groups'][name]['at_least_selected_P_cold']}</td></tr>" for name in summary_groups)
reference_names = ['clean_receiver', 'clean_donor', 'selected_pair', 'selected_hot_only', 'selected_small_only', 'selected_pair_reverse', 'whole_mlp_donor_replace']
references = ''.join(f"<tr><td>{e(name.replace('_', ' '))}</td><td>{percent(by_id[name]['P_cold'])}</td><td>{percent(by_id[name]['P_large'])}</td><td>{e(by_id[name]['top_token'])}</td><td>{by_id[name]['actual_delta_norm']:.4f}</td></tr>" for name in reference_names)
active_ids = sorted(set(R['selection']['receiver_active_pool'] + R['selection']['donor_active_pool']))
gates = R['selection']; feature_table = ''
for f in [2339, 941] + active_ids:
    info = labels[f]
    feature_table += f"<tr><td>L22–T{f}</td><td>{e(info.get('label') or 'Unlabelled')}</td><td>{info.get('rank')}</td><td>{gates['all_receiver_gates'][f]:.6f}</td><td>{gates['all_donor_gates'][f]:.6f}</td></tr>"
active_results = ''.join(f"<tr><td>{e(formula(r))}</td><td>{percent(r['P_cold'])}</td><td>{percent(r['P_large'])}</td><td>{r['actual_delta_norm']:.4f}</td></tr>" for r in rows if r['group'].startswith('active_'))
successes = V['groups']['active_pair']['at_least_99_percent']
finding = (f"{successes} of {len(active)} gate-active alternative pairs also reach at least 99% P(cold). The selected pair is not unique at this fixed strength." if successes else f"None of the {len(active)} gate-active alternative pairs reaches 99% P(cold) at this fixed strength. This small set does not establish general semantic selectivity.")
css = "body{margin:0;background:#f3f5f1;color:#223b35;font:16px/1.6 system-ui}main{max-width:1160px;margin:auto;padding:28px 22px}h1{line-height:1.2}section{background:white;border:1px solid #dce4dc;border-radius:12px;padding:24px;margin:24px 0}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3ef;padding:18px;border-radius:8px}a{color:#08796b}.meta{font-size:13px;color:#58756d}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 8px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4dc}.scroll{overflow-x:auto}blockquote{border-left:4px solid #23816c;margin:12px 0;padding:12px}input,select{font:inherit;padding:8px;border:1px solid #a6b9ae;border-radius:6px;margin:5px}button{font:inherit;padding:8px 14px;margin:5px}"
page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Layer22 small-to-hot: random MOLT and whole-MLP controls</title><style>{css}</style></head><body><main><p class='meta'>13 October 2026 · Qwen3-4B · SPAR RTX 3090 · unfrozen attention</p><h1>Does the selected MOLT pair outperform random MOLTs?</h1><section><h2>Results</h2><p>The selected L22–T2339/L22–T941 pair reaches <strong>{percent(selected['P_cold'])} P(cold)</strong>. Exact replacement with the donor’s complete MLP output gives <strong>{percent(whole['P_cold'])} P(cold)</strong> and {percent(whole['P_large'])} P(large).</p><p>{e(finding)}</p><p>{len(rows)} measured conditions, {len(R['batch_clean_checks'])} matched batches with clean identity rows, and independently recomputed full-vocabulary probabilities. All uniform samples, including zero edits, are retained.</p><img src='controls-scatter.png' alt='Random individual and paired MOLT controls, active controls, and whole MLP replacement'></section>"
page += "<section><h2>What the controls show</h2><p><strong>All 100 uniformly sampled MOLTs were inactive at both source tokens.</strong> Their individual and paired edits therefore leave the model unchanged. This is a weak null control by itself.</p><p>We also tested all other available gate-active individuals and all three distinct active pairs. None makes cold the top token. The pair −4.5 receiver T8 +4.5 donor T71 has correction norm 63.0534, versus 65.3121 for the selected pair, but still gives effectively zero P(cold). This provides a more informative comparison at a similar, though not exactly matched, magnitude.</p><p>Neither selected hot-only addition nor selected small-only subtraction reaches 99% P(cold). The successful condition uses both contributions.</p></section>"
page += f"<section><h2>Prompts and intervention position</h2><p><strong>Receiver:</strong></p><blockquote>{e(R['protocol']['receiver'])}</blockquote><p><strong>Donor:</strong></p><blockquote>{e(R['protocol']['donor'])}</blockquote><p>Only zero-based full-chat <strong>position 9</strong> is directly edited: <code>small</code> in the receiver and <code>hot</code> in the donor. Both chat sequences have 29 tokens; next-token probabilities are read at position 28. All other direct MLP-output positions remain bitwise unchanged.</p></section>"
page += "<section><h2>Exact formulas</h2><pre>C_f(x) = gate_f(standardized current residual) × lowrank_f(standardized current residual) × output_std\n\nSelected pair:\nMLP_receiver[22,9] += −4.5 × C_2339(receiver) +4.5 × C_941(donor)\n\nRandom individual positive:\nMLP_receiver[22,9] += +4.5 × C_f(donor)\nRandom individual negative:\nMLP_receiver[22,9] += −4.5 × C_f(receiver)\n\nRandom pair:\nMLP_receiver[22,9] += −4.5 × C_A(receiver) +4.5 × C_B(donor)\nReversed pair:\nMLP_receiver[22,9] += +4.5 × C_A(receiver) −4.5 × C_B(donor)\n\nWhole MLP control (replacement at strength 1):\nMLP_receiver[22,9,:] = native MLP_donor[22,9,:]</pre><p>The original receiver MLP remains in the MOLT conditions. Full gate × low-rank output contributions are transferred, using clean donor or receiver inputs as specified; U/V weights are not replaced. The MOLT input is the residual before MLP input normalization, after attention. Writes occur at the raw MLP output before its residual addition. The whole-MLP control copies all 2,560 output values exactly; it retains the receiver residual and is not a whole-residual replacement.</p></section>"
page += f"<section><h2>Random selection and gate sparsity</h2><p>Seed {R['protocol']['random_seed']}. The 100 uniform IDs are drawn without replacement from all 2,478 layer22 MOLTs other than T2339 and T941. The same IDs are tested at both individual signs. A shuffled derangement supplies 100 distinct two-term pairs, each tested in both sign directions. No intervention results enter selection.</p><p>Only {len(R['selection']['receiver_active_pool'])} non-target gates are active on the clean receiver and {len(R['selection']['donor_active_pool'])} on the clean donor. Supplementary controls test every available active individual and all {len(active)} distinct active pairs. When both relevant gates are zero, the edit is zero; such a trial measures inactivity rather than perturbation robustness.</p><div class='scroll'><table><thead><tr><th>Control group</th><th>N</th><th>Nonzero writes</th><th>Maximum P(cold)</th><th>≥99%</th><th>≥ selected</th></tr></thead><tbody>{table}</tbody></table></div></section>"
page += f"<section><h2>Reference conditions</h2><div class='scroll'><table><thead><tr><th>Condition</th><th>P(cold)</th><th>P(large)</th><th>Top token</th><th>Actual correction norm</th></tr></thead><tbody>{references}</tbody></table></div><p>Prior selected-pair P(cold): {percent(R['prior_selected_comparison']['prior_P_cold'])}; current P(cold): {percent(selected['P_cold'])}. {e(R['prior_selected_comparison']['note'])}</p></section>"
page += f"<section><h2>Selected and other active features</h2><p>Labels below are recorded dashboard descriptions from 12 October; they were not used to select the random IDs. A label describes gate activation and does not establish what its low-rank contribution writes.</p><div class='scroll'><table><thead><tr><th>MOLT</th><th>Label</th><th>Rank</th><th>Receiver gate</th><th>Donor gate</th></tr></thead><tbody>{feature_table}</tbody></table></div><h3>All active alternative controls</h3><div class='scroll'><table><thead><tr><th>Edit at L22 token9</th><th>P(cold)</th><th>P(large)</th><th>Actual correction norm</th></tr></thead><tbody>{active_results}</tbody></table></div></section>"
page += "<section><h2>Magnitude and interpretation</h2><img src='norm-control.png' alt='P(cold) against intervention size'><p>Equal coefficients do not imply equal correction norms. Ranks, gate magnitudes, and low-rank outputs differ. These random-MOLT controls are not norm-matched random-direction controls. Uniform inactive cases cannot establish semantic selectivity; the small active set and this single prompt pair limit the conclusion. Whole-MLP replacement at strength 1 also differs in magnitude and direction from the extrapolated selected MOLT edit at strength 4.5.</p><p>P(cold) is the full-vocabulary next-token probability for lowercase token 87072; P(large) uses token 16767. Uppercase variants are separate tokens. The probabilities are not renormalized over two answers.</p></section>"
page += "<section><h2>All conditions</h2><p>Search by ID or choose a group. Exact formulas, source gates, norms, and probability values are also available in the CSV and raw results.</p><input id='search' placeholder='Search condition'><select id='group'><option value=''>All groups</option>" + ''.join(f"<option value='{e(g)}'>{e(g.replace('_',' '))}</option>" for g in sorted({r['group'] for r in rows})) + "</select><p id='count'></p><div class='scroll'><table><thead><tr><th>Condition</th><th>Edit</th><th>P(cold), %</th><th>P(large), %</th><th>Norm</th></tr></thead><tbody id='conditions'></tbody></table></div><button id='prev'>Previous</button><button id='next'>Next</button></section>"
payload = [{'id': r['id'], 'group': r['group'], 'edit': formula(r), 'cold': r['P_cold'], 'large': r['P_large'], 'norm': r['actual_delta_norm']} for r in rows]
page += "<script id='data' type='application/json'>" + json.dumps(payload).replace('</', '<\\/') + "</script><script>const rows=JSON.parse(document.getElementById('data').textContent);let page=0;function render(){let rr=rows.filter(r=>(!document.getElementById('group').value||r.group===document.getElementById('group').value)&&r.id.includes(document.getElementById('search').value));page=Math.min(page,Math.max(0,Math.ceil(rr.length/25)-1));const body=document.getElementById('conditions');body.replaceChildren();for(const r of rr.slice(page*25,(page+1)*25)){let tr=document.createElement('tr');for(const value of [r.id,r.edit,(100*r.cold).toPrecision(6),(100*r.large).toPrecision(6),r.norm.toFixed(4)]){let td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}document.getElementById('count').textContent=`${rr.length} conditions · page ${page+1}`;}for(const id of ['search','group'])document.getElementById(id).addEventListener('input',()=>{page=0;render()});document.getElementById('prev').onclick=()=>{page=Math.max(0,page-1);render()};document.getElementById('next').onclick=()=>{page++;render()};render();</script>"
page += f"<section><h2>Verification and backup</h2><p>Independent CPU verification recomputes probabilities from all {V['vocabulary_size']:,} logits, reconstructs every intended contribution edit, reproduces BF16 write rounding, and checks exact full-MLP replacement. Runtime assertions verify unchanged upstream states, only token9 directly edited, normal downstream attention changes, inactive recovery, self/zero controls, selected repeat, and restoration.</p><p><a href='conditions.csv'>All conditions CSV</a> · <a href='artifacts/results.json'>Raw results</a> · <a href='artifacts/selection.json'>Prespecified random IDs</a> · <a href='verification.json'>Independent audit</a> · <a href='artifacts/next-token-logits.npz'>Lossless full-vocabulary logits</a> · <a href='artifacts/direct-write.npz'>Direct-write vectors</a> · <a href='artifacts/contributions.npz'>MOLT contributions</a> · <a href='protocol.json'>Protocol</a> · <a href='run.py'>Executed source</a> · <a href='README.md'>Reproduction details</a></p><p>SPAR instance 55734307 and its storage are retained at approximately $0.189/hour including disk. No lifecycle action was performed.</p></section></main></body></html>"
(ROOT / 'index.html').write_text(page)
(ROOT / 'summary.json').write_text(json.dumps({'selected_P_cold': selected['P_cold'], 'whole_mlp_P_cold': whole['P_cold'], 'whole_mlp_P_large': whole['P_large'], 'active_pair_99_percent_successes': successes, 'active_pair_count': len(active), 'finding': finding}, indent=2) + '\n')
print('Report and figures saved:', ROOT)
