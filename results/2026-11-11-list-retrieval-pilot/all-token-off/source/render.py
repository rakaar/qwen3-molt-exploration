"""Render the seven-case ablation table without rounding tiny probabilities to zero."""
from pathlib import Path
from html import escape
import json
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/2026-11-11-list-retrieval-pilot/all-token-off'
s = json.loads((OUT / 'analysis.json').read_text())
r = json.loads((OUT / 'results.json').read_text())
assert s['cases'] == 7 and s['native_baselines_and_zero_controls_exact']

def pct(p):
    value = p * 100
    return f'{value:.4f}' if value >= .001 else f'{value:.2e}'

def pair(row, target):
    return pct(row[f'native_p_{target}_word']) + ' → ' + pct(row[f'off_p_{target}_word'])

rows = ''
for c in s['rows']:
    label = f"List {c['list_id'].split('-')[-1]} · {c['format']}"
    if not c['native_twelfth_control_correct']:
        label += ' †'
    rows += f"<tr><td>{escape(label)}</td><td>{escape(c['eleventh_word'])} / {escape(c['twelfth_word'])}</td><td><code>{escape(c['native_response'])}</code></td><td><code>{escape(c['off_response'])}</code></td><td class='ablation-probs'>{pair(c,'eleventh')}</td><td class='ablation-probs'>{pair(c,'twelfth')}</td></tr>"

edits = ''
for c in r['cases']:
    tokens = [t for t in c['off']['trace'] if t['context'] == 'generation' and t['phase'] == 'input' and t['removed']]
    text = '; '.join(f"{t['position']}: {t['token'].replace(chr(10),'↵')} ({t['region']}, gate {t['natural_gate']:.6f} → 0)" for t in tokens)
    edits += f"<tr><td>{escape(c['id'])}</td><td>{escape(text)}</td></tr>"

cal = s['calibration']
html = f'''<section id="all-token-off">
<p class="eyebrow">Causal follow-up · correctly answered eleventh-word requests</p>
<h2>Turn T2375 off at every active token</h2>
<div class="findings"><p><strong>{s['switched_to_twelfth']}/7 switched to the twelfth word.</strong> All {s['retained_eleventh']}/7 retained the correct eleventh word; {s['responses_changed']}/7 response texts changed.</p></div>
<p>These are all seven revised eleventh-word cases that the native model answered correctly: five numbered lists and two bullet lists. Selection was frozen before seeing the edits. The intervention subtracts only T2375’s estimated contribution from the native layer-29 MLP output at <strong>every token where its natural gate is positive</strong>. Native attention and all MLPs otherwise run normally.</p>
<p>The policy covers the <strong>entire input and generation</strong>, including numbered-list reading and the chat answer prefix. It removed {s['edited_input_tokens']} active input contributions. During the actual greedy responses, {s['edited_generated_tokens']} generated-word/token contributions and {s['edited_terminal_tokens']} terminal contributions were active to remove. Thus the generation policy stayed enabled, while actual removals occurred only on input tokens.</p>
<p class="muted">All probabilities below are <strong>percentages, native → gate off</strong>. P11/P12 multiply the probabilities of <em>all tokens</em> in the exact lowercase 11th/12th word. They exclude following EOS, punctuation, extra whitespace and formatting variants. Scientific notation preserves tiny values: 1e-7% means 0.0000001%.</p>
<div class="scroll"><table id="ablation-results"><thead><tr><th>Case</th><th>11th / 12th target</th><th>Native response</th><th>Gate-off response</th><th>P11 native → off (%)</th><th>P12 native → off (%)</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="muted">† The saved matching twelfth-word question failed for both bullet cases: list 1 answered pencil instead of drum; list 5 answered hammer instead of ticket. All five numbered twelfth-word controls were correct. Both bullet cases are retained under the requested selection criterion: native eleventh-word correctness.</p>
<details class="ablation-details"><summary>Exact intervention positions, checks and month calibration</summary>
<p>Layer and token IDs are zero-based. For standardized pre-RMS input z, the subtracted vector is gate(z) × ((z @ V @ U) × output_std). The shared output mean is not removed. No MLP is replaced with MOLTs. Each active contribution is subtracted in FP32, followed by one BF16 output cast; inactive positions are unchanged.</p>
<div class="scroll"><table id="ablation-positions"><thead><tr><th>Case</th><th>Removed input contributions</th></tr></thead><tbody>{edits}</tbody></table></div>
<p>For every case, unhooked native and zero-strength runs exactly match the frozen baseline outputs, selected probabilities, every decoding-logit hash and the terminal-unused-logit hash. The zero-strength candidate-word probabilities also match exactly. Independent CPU reconstruction verifies the gates and operators, and every audited edited MLP output matches the exact expected BF16 rounding. The largest relative L2 difference between intended and rounded actual deltas was {100*s['audit']['maximum_actual_minus_intended_delta_relative_l2']:.3f}%.</p>
<p><strong>Separate implementation calibration:</strong> for the raw prefix “the eleventh month of the year is”, final-input-only removal changes a November-leading completion to a December-leading completion. P(“ november”) changes {pct(cal['scores']['native']['eleventh'])}% → {pct(cal['scores']['off']['eleventh'])}%; P(“ december”) changes {pct(cal['scores']['native']['twelfth'])}% → {pct(cal['scores']['off']['twelfth'])}%. These are literal leading-space lowercase month continuations, excluded from the seven list cases.</p>
<p>Captured native calibration response (32-token limit):</p><pre>{escape(cal['native_response'])}</pre><p>Captured edited calibration response (32-token limit):</p><pre>{escape(cal['off_response'])}</pre>
</details>
<p class="muted"><a href="all-token-off/analysis.json">Verified summary</a> · <a href="all-token-off/probabilities.csv">All probabilities, including word-plus-EOS (CSV)</a> · <a href="all-token-off/results.json">Full responses, token traces and scores</a> · <a href="all-token-off/protocol.json">Frozen selection and intervention</a> · <a href="all-token-off/source/run.py">Inference source</a> · <a href="all-token-off/source/analyze.py">Independent verification source</a></p>
</section>'''
(OUT / 'section.html').write_text(html + '\n')
(OUT / 'source').mkdir(exist_ok=True)
for name in ['run.py', 'analyze.py', 'render.py']:
    shutil.copy2(HERE / name, OUT / 'source' / name)
print('Rendered all seven exact responses and both full-word probability comparisons.')
