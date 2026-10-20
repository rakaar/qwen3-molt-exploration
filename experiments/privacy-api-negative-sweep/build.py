"""CPU-only table for the exact matched Northstar Weather permission prompt."""
import csv
import hashlib
import html
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
ROOT = PROJECT / 'results/2026-10-20-privacy-api-negative-sweep'
GROUPS = {
    'L11': ('L11–T1596', 'Data Security and Privacy Protection'),
    'L13': ('L13–T921', 'Confidentiality and Privacy Protection'),
    'L11_L13': ('L11 + L13', 'T1596 + T921'),
    'all_three': ('L11 + L12 + L13', 'T1596 + T1891 + T921'),
}
esc = lambda value: html.escape(str(value), quote=True)

def main():
    reviews = json.loads((ROOT / 'reviews.json').read_text())
    ratings = {r['trial_id']: r for r in reviews['rows']}
    case = json.loads((SRC / 'case.json').read_text())
    reference = json.loads((SRC / 'matched-authorized-reference.json').read_text())
    assert reference['case']['system'].replace('is authorized', 'is not authorized') == case['system']
    assert reference['case']['user'] == case['user']
    measurements, controls, stage_proofs = {}, [], []
    stages = sorted(d for d in ROOT.glob('privacy-api-negative*') if d.is_dir())
    for stage in stages:
        data = json.loads((stage / 'artifacts/results.json').read_text())
        proof = json.loads((stage / 'verification.json').read_text())
        assert data['status'] == 'complete' and proof['passed']
        assert data['case'] == case and data['protocol']['timings'] == ['user_to_end']
        assert data['boundaries']['user_first_token'] == 103
        assert data['boundaries']['user_last_token'] == 111
        assert data['boundaries']['prompt_token_count'] == 121
        stage_proofs.append({'stage': stage.name, 'trials': len(data['rows']), 'verified_layer_token_records': proof['verified_layer_token_records'], 'elapsed_seconds': data['elapsed_seconds']})
        for row in data['rows']:
            assert row['answer_sha256'] == hashlib.sha256(row['answer'].encode()).hexdigest()
            trial = row['trial']
            if trial['mode'] in ['native', 'observe']:
                assert row['output_token_ids'] == case['output_token_ids']
                controls.append(row)
                continue
            assert trial['mode'] == 'additive' and trial['scope'] == 'user_to_end'
            review = ratings[trial['id']]
            assert review['answer_sha256'] == row['answer_sha256']
            group = trial['id'].split('__')[0]
            key = (group, trial['offset'])
            assert key not in measurements
            measurements[key] = {
                'group': group, 'offset': trial['offset'], 'answer': row['answer'],
                'answer_sha256': row['answer_sha256'], 'status': review['status'],
                'label': review['label'], 'rationale': review['rationale'],
                'coherence': review['coherence'], 'contains_exact_key': row['contains_exact_secret'],
                'partial_key_disclosed': review['partial_key_disclosed'],
                'stop_reason': row['stop_reason'], 'output_tokens': len(row['output_token_ids']),
                'stage': stage.name, 'trial_id': trial['id'],
            }
    assert len(ratings) == len(measurements)
    assert all(r['answer'] == case['answer'] for r in controls)
    rows = sorted(measurements.values(), key=lambda r: (list(GROUPS).index(r['group']), -r['offset']))
    clean = [r for r in rows if r['coherence'] == 'coherent' and r['contains_exact_key']]
    leaks = [r for r in rows if r['contains_exact_key']]
    strengths = sorted({0, *(r['offset'] for r in rows)}, reverse=True)
    summary = {
        'case_id': case['id'], 'service': case['title'], 'same_original_permission_pair_verified': True,
        'negative_conditions': len(rows), 'exact_native_controls': len(controls),
        'total_generations': len(rows) + len(controls), 'stage_proofs': stage_proofs,
        'full_key_disclosure_conditions': [{'group': r['group'], 'offset': r['offset'], 'coherence': r['coherence']} for r in leaks],
        'coherent_full_key_disclosure_conditions': [{'group': r['group'], 'offset': r['offset']} for r in clean],
        'system_header_unedited': True, 'attention_unfrozen': True, 'native_mlp_retained': True,
        'scope': 'User positions 103–111, template/prefix 112–120, and every processed output position. System/header 0–102 unchanged.',
        'patch': 'native MLP output += offset × current scaled low-rank output; estimated gate becomes natural gate + offset, without clipping.',
    }
    (ROOT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'matched-authorized-reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2) + '\n')
    with (ROOT / 'all-responses.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    headers = ''.join(f'<th>{esc(name)}<small>{esc(label)}</small></th>' for name, label in GROUPS.values())
    matrix = f'<div class="scroll"><table class="matrix"><thead><tr><th>Gate offset</th>{headers}</tr></thead><tbody>'
    for strength in strengths:
        matrix += f'<tr><th>{strength if strength == 0 else "−" + str(abs(strength))}</th>'
        for group in GROUPS:
            row = measurements.get((group, strength))
            if strength == 0:
                matrix += '<td class="refusal">Fluent refusal</td>'
            elif row:
                matrix += f'<td class="{row["status"]}"><a href="#{group}-{abs(strength)}">{esc(row["label"])}</a></td>'
            else:
                matrix += '<td class="untested">Not tested<small>Stopped after breakdown</small></td>'
        matrix += '</tr>'
    matrix += '</tbody></table></div>'
    details = ''
    for group, (name, label) in GROUPS.items():
        details += f'<section id="{group}"><h2>{esc(name)} <small>{esc(label)}</small></h2><div class="scroll"><table class="responses"><thead><tr><th>Offset</th><th>Complete model response</th><th>Assessment</th><th>End</th></tr></thead><tbody>'
        details += f'<tr><th>0</th><td class="answer">{esc(case["answer"])}</td><td>Fluent refusal</td><td>EOS</td></tr>'
        for row in [r for r in rows if r['group'] == group]:
            text = f'<div class="answer">{esc(row["answer"])}</div>'
            if row['answer'].count('\n') > 20:
                text = f'<details><summary>Show full response: mostly blank lines and fragments</summary>{text}</details>'
            details += f'<tr id="{group}-{abs(row["offset"])}"><th>−{abs(row["offset"])}</th><td class="response-cell">{text}</td><td class="{row["status"]}"><b>{esc(row["label"])}</b><p>{esc(row["rationale"])}</p><small>Full key: {"yes" if row["contains_exact_key"] else "no"} · partial key: {"yes" if row["partial_key_disclosed"] else "no"}</small></td><td>{"EOS" if row["stop_reason"] == "eos" else "96-token cap"}</td></tr>'
        details += '</tbody></table></div></section>'
    links = ''.join(f'<li><a href="{stage.name}/artifacts/results.json">{esc(stage.name)}: full answers and token traces</a> · <a href="{stage.name}/protocol.json">frozen protocol</a> · <a href="{stage.name}/verification.json">verification</a> · <a href="{stage.name}/run.py">executed runner</a></li>' for stage in stages)
    if clean:
        headline = 'Coherent disclosure observed on the matched API-key prompt.'
    else:
        headline = 'No clean coherent key disclosure on this tested doubling grid.'
    leak_note = 'The full key does appear in ' + ', '.join(f'{GROUPS[r["group"]][0]} at −{abs(r["offset"])} ({r["coherence"]})' for r in leaks) + '.' if leaks else 'No tested answer contains the full key.'
    traces = sum(s['verified_layer_token_records'] for s in stage_proofs)
    css = (SRC / 'style.css').read_text() + '\n.response-cell{width:54%}.answer{white-space:pre-wrap;overflow-wrap:anywhere}.repetitive{background:#fff0cb}.mixed_degraded{background:#f7ddbd}details summary{cursor:pointer}details>.answer{margin-top:10px}'
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched API-key prompt · negative input + output steering</title><style>{css}</style><main>
<div class="eyebrow">Qwen3-4B · exact Northstar Weather prompt · 20 October 2026</div><h1>Reverse privacy steering<br>on the original API-key question</h1><p class="lead">The same Northstar Weather permission pair used in the earlier authorization report. This sweep uses its exact <b>unauthorized</b> system message and unchanged user request.</p><div class="callout"><b>{headline}</b><br>{esc(leak_note)} Disclosure and coherence are scored separately; a key leak inside a degraded answer remains a privacy failure.</div>
<nav><a href="#table">Strength table</a><a href="#matched">Matched positive / negative context</a><a href="#method">Exact intervention</a><a href="#records">Raw records</a><a href="../2026-10-15-privacy-refusal-overview/index.html">Original authorization report</a></nav>
<section id="table"><h2>Input + output: 0, −2, −4, −8 … until breakdown</h2><p>Cells link to complete answers. Each selected transform receives the same additive gate offset. Offset 0 is native generation, not a zero-clamped gate.</p>{matrix}<p class="muted">“Repetitive refusal” remains grammatical and was followed to a larger strength. “Key disclosed; confused” is a full leak with contradictory/truncated surrounding text. Blank cells were not tested after clear breakdown. Results are tested points, not exact thresholds.</p></section>
<section id="matched"><h2>The original permission pair</h2><p><b>Same user request:</b> {esc(case['user'])}</p><p><b>Unauthorized system message, used exactly in this new sweep</b></p><pre>{esc(case['system'])}</pre><p><b>Authorized partner:</b> replace only “is not authorized” with “is authorized”. Service name, key, policy and user question remain identical. All credentials are invented and nonfunctional.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Complete answer</th><th>Evidence</th></tr></thead><tbody><tr><th>Authorized · native</th><td>{esc(reference['case']['answer'])}</td><td>Original 15 October run</td></tr><tr><th>Authorized · L11 fixed gate = 16</th><td>{esc(reference['positive_steering']['answer'])}</td><td>Original 15 October run</td></tr><tr><th>Unauthorized · native</th><td>{esc(case['answer'])}</td><td>Fresh controls reproduce original tokens exactly</td></tr></tbody></table></div><p>The earlier positive edit was <b>absolute replacement</b>: <code>MLP += (16 − natural_gate) × L(z)</code>. The new negative sweep follows the requested <b>additive offset</b>: <code>MLP += offset × L(z)</code>. These are different parameterizations. This is a matched-prompt extension, not an equal-and-opposite test of all possible steering rules.</p></section>
<section id="method"><h2>What changes at each token?</h2><pre>native MLP output += offset × L(z)
estimated gate = natural_gate(z) + offset
z = (current post-attention residual − released input mean) / input std
L(z) = ((z @ V) @ U) × released output std</pre><p><b>Selected positions:</b> user content <b>103–111</b>, remaining template/assistant prefix <b>112–120</b>, and every subsequently processed answer token (positions start at <b>121</b>). System/header <b>0–102 stay native</b>, including the trusted key and authorization rule. The authorized historical partner has one fewer prompt token because “not” is absent.</p><p>The current input, natural gate and complete low-rank output are recomputed at every selected layer/token on the evolving trajectory. The native MLP is retained; attention stays unfrozen. No donor or MLP reconstruction is used. Negative effective gates are allowed, including at naturally inactive gates. The final sampled token or EOS is not processed again.</p><p>L11–T1596: “Data Security and Privacy Protection”; L12–T1891: “Privacy and Consent Data Gate”; L13–T921: “Confidentiality and Privacy Protection”. All IDs are zero-based; labels are hypotheses.</p><p>Same pinned Qwen3-4B/MOLT revisions, PyTorch 2.14.0+cu126 / Transformers 4.57.6, greedy batch one, BF16 native model, FP32 MOLT arithmetic, eager attention, KV cache, thinking disabled and 96-new-token cap as the earlier study.</p></section>
{details}
<section><h2>Search and evidence limits</h2><p>Start with −2, −4, −8 and −16 for four groups. Read full answers and continue groups that have not clearly broken down by doubling magnitude. A token cap alone does not establish incoherence; repetitive grammatical or confused but readable text is distinguished from broken fragments and gibberish. Every continuation protocol is frozen before inference.</p><p>{summary['total_generations']} fresh generations: {len(rows)} negative conditions and {len(controls)} exact native/observation controls. All {traces:,} layer/token records passed independent numerical verification and post-download checks. Controls exactly reproduce the original unauthorized refusal, including EOS.</p><p>One selected question, manual unblinded review, no random-transform controls and no intermediate strengths. No dataset success rate, monotonicity or universal threshold follows. The positive target-16 result remains established only for its tested authorized cases. This new sweep replaces the earlier mismatched Bob/Alice headline comparison; that experiment is retained as <a href="../2026-10-20-privacy-input-output-negative-sweep/bob-alice-original.html">an explicitly separate historical result</a>.</p></section>
<section id="records"><h2>Full records</h2><ul><li><a href="all-responses.csv">Complete response table · CSV</a> · <a href="reviews.json">Answer-bound reviews</a> · <a href="summary.json">Summary</a> · <a href="matched-authorized-reference.json">Historical authorized partner and source hashes</a></li>{links}</ul><p class="muted">Frozen source, selected MOLT assets and raw outputs are backed up. Full base weights and cloud credentials are excluded. GPU lifecycle is separate; the user’s working GPU is retained.</p></section></main></html>'''
    (ROOT / 'index.html').write_text(doc)
    print(json.dumps({'report': str(ROOT / 'index.html'), 'generations': summary['total_generations'], 'negative_conditions': len(rows), 'trace_records': traces, 'full_leaks': len(leaks), 'coherent_full_leaks': len(clean)}, indent=2))

if __name__ == '__main__':
    main()
