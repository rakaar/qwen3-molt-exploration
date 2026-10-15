"""Build the four-question comparison from saved outputs and hash-guarded reviews."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    root = args.result_dir
    reference = json.loads((root / 'comparison-reference.json').read_text())
    data_path = root / 'artifacts/all-results.json'
    if not data_path.exists():
        data_path = root / 'artifacts/added-gate-results.json'
    result = json.loads(data_path.read_text())
    review = json.loads((root / 'assessment.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    grades = {r['trial_id']: r for r in review['rows']}
    for row in result['rows']:
        assert grades[row['trial_id']]['answer_sha256'] == hashlib.sha256(row['answer'].encode()).hexdigest()
    esc = lambda value: html.escape(str(value))
    cases = []
    for case in reference['cases']:
        rows = [r for r in result['rows'] if r['case_id'] == case['case_id']]
        successful = [r for r in rows if r['added_gate'] is not None and r['added_gate'] > 0 and grades[r['trial_id']]['clean_coherent_abstention']]
        selected = min(successful, key=lambda r: r['added_gate']) if successful else None
        cases.append({**case, 'selected': selected, 'successful_strengths': [r['added_gate'] for r in successful], 'positive_trials': sum(r['added_gate'] is not None and r['added_gate'] > 0 for r in rows)})
    wins = sum(c['selected'] is not None for c in cases)
    distinct_vampire = sorted({r['added_gate'] for r in result['rows'] if r['case_id']=='P2-U5' and r['added_gate'] is not None and r['added_gate']>0})
    summary = {'converted_original_made_up_cases': wins, 'original_made_up_cases': 4, 'baseline_already_withheld': 6, 'combined_withholding_with_question_specific_settings': 6 + wins, 'not_a_single_shared_strength': True, 'fully_grounded_baseline_withholdings': 5, 'total_trials': len(result['rows']), 'vampire_distinct_positive_strengths': distinct_vampire, 'case_results': [{'case_id': c['case_id'], 'selected_tested_strength': c['selected']['added_gate'] if c['selected'] else None, 'successful_strengths': c['successful_strengths'], 'positive_trials': c['positive_trials']} for c in cases]}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    csv_rows = []
    for row in result['rows']:
        g = grades[row['trial_id']]
        csv_rows.append({'case_id': row['case_id'], 'question': row['question'], 'strength': row['added_gate'], 'answer': row['answer'], 'stop_reason': row['stop_reason'], **g})
    with (root / 'responses.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    cards = []
    for case in cases:
        chosen = case['selected']
        if chosen:
            strength = f"+{chosen['added_gate']:g}"
            answer = chosen['answer']
            badge = 'Model said “I don’t know”'
            state = 'success'
            note = 'A successful tested setting, not a claimed minimum. This is a fixed replay of the earlier result.' if case['case_id'] != 'P2-U5' else 'A successful setting selected from this search; not a proven continuous minimum.'
        else:
            example = next(r for r in result['rows'] if r['case_id'] == case['case_id'] and r['added_gate'] == 24)
            strength = '+24 · failed example'
            answer = example['answer']
            badge = 'Still unresolved'
            state = 'failure'
            note = 'No successful tested strength. This readable example still falsely denies rabies transmission and misdescribes the passage. A negative grid search does not rule out a narrow interval between tested values.'
        cards.append(f'''<article class="case" id="{esc(case['case_id'])}"><div class="case-head"><span class="case-id">{esc(case['case_id'])}</span><span class="badge {state}">{badge}</span></div><h2>{esc(case['question'])}</h2><div class="comparison"><section><h3>Original · no steering</h3><div class="answer original">{esc(case['original_answer'])}</div><p class="small">Model made up an answer: the passage does not establish this answer.</p></section><section><h3>Steered · <strong>{esc(strength)}</strong></h3><div class="answer steered {state}">{esc(answer)}</div><p class="small">{note}</p></section></div><details><summary>Exact receiver prompt and chat template</summary><pre>{esc(case['prompt'])}</pre><details><summary>Complete serialized chat</summary><pre>{esc(case['chat'])}</pre></details></details></article>''')
    vampire = [r for r in result['rows'] if r['case_id'] == 'P2-U5']
    vampire.sort(key=lambda r: -1 if r['added_gate'] is None else r['added_gate'])
    attempts = []
    tiles = []
    for row in vampire:
        g = grades[row['trial_id']]
        label = 'Native' if row['added_gate'] is None else f"+{row['added_gate']:g}"
        if row['trial_id'].startswith('fractional_'):
            label += ' ↻' if row['added_gate'] in [None, 0, 8, 16, 24, 32] else ''
        state = 'success' if g['clean_coherent_abstention'] else 'failure' if g['coherent'] else 'incoherent'
        verdict = 'Coherent, grounded I don’t know' if state == 'success' else 'Coherent but still answers / makes a false claim' if state == 'failure' else 'Incoherent / repetitive / unfinished'
        if state == 'failure' and not g['made_up_answer'] and not g['false_or_unsupported_explanation']:
            verdict = 'Question echo / no clean abstention'
        tiles.append(f'<a href="#trial-{esc(row["trial_id"])}" class="tile {state}" title="{esc(verdict)}">{esc(label)}</a>')
        attempts.append(f'''<details class="attempt {state}" id="trial-{esc(row['trial_id'])}"><summary><strong>{esc(label)}</strong><span>{esc(verdict)}</span></summary><div class="answer">{esc(row['answer'])}</div><p>{esc(g['reason'])}</p><p class="small">Stop: {esc(row['stop_reason'])}; {len(row['output_token_ids'])} generated tokens including any EOS. Trial {esc(row['trial_id'])}.</p></details>''')
    passages = ''.join(f'<details><summary>{esc(p["article_title"])} passage</summary><p>{esc(p["text"])}</p></details>' for p in reference['baseline_cases']['passages'])
    rental = json.loads((root / 'rental-record.json').read_text())
    equation = 'MLP_output += a × ((z × V_368) × U_368) × output_std_19\nEffective estimated gate = natural_gate(z) + a'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four questions: from made-up answers to I don’t know · MOLT</title><style>
    :root{{--ink:#183138;--muted:#546a6e;--line:#d9e4e0;--paper:#f6f8f4;--green:#0c6b58;--red:#9d382e;--amber:#946010}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:48px 24px 80px}}h1{{font-size:clamp(30px,5vw,52px);line-height:1.12;letter-spacing:-.035em;margin:12px 0 20px;max-width:900px}}h2{{font-size:24px;line-height:1.35;margin:14px 0 24px}}h3{{font-size:15px;margin:0 0 10px}}.eyebrow,.case-id{{font:700 12px/1.5 ui-monospace,monospace;text-transform:uppercase;letter-spacing:.1em}}.intro{{max-width:850px;color:var(--muted);font-size:18px}}.stats{{display:flex;flex-wrap:wrap;gap:14px;margin:28px 0}}.stat{{background:white;border:1px solid var(--line);padding:16px 22px;border-radius:12px;flex:1;min-width:170px}}.stat strong{{font-size:30px;display:block;line-height:1.2}}.small{{font-size:13px;color:var(--muted)}}.case,.panel{{border:1px solid var(--line);background:white;border-radius:16px;padding:26px;margin:24px 0}}.case-head{{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}}.badge{{font-size:12px;font-weight:700;padding:5px 11px;border-radius:30px}}.success{{background:#e7f5ed;color:var(--green)}}.failure{{background:#fff0ec;color:var(--red)}}.incoherent{{background:#fff3d6;color:var(--amber)}}.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}.answer{{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;border-radius:10px;border:1px solid var(--line);font-size:16px}}.original{{background:#fff7f4}}.steered.success{{background:#eef8f1}}details{{border-top:1px solid var(--line);padding:14px 0 3px;margin-top:12px}}summary{{cursor:pointer;color:var(--green);font-weight:600;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere;font-size:13px}}pre{{background:#f3f6f3;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;border-radius:8px}}a{{color:var(--green)}}.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(60px,1fr));gap:7px;margin:20px 0}}.tile{{display:block;padding:7px 4px;text-align:center;border-radius:6px;text-decoration:none;font:600 13px/1.5 ui-monospace,monospace;border:1px solid currentColor}}.attempt{{padding:14px;border-radius:8px;margin:8px 0;border:1px solid var(--line)}}.attempt summary{{display:flex;gap:20px;color:inherit;font-size:14px}}.attempt .answer{{background:white;color:var(--ink);margin:15px 0}}.attempt p{{color:var(--ink)}}footer{{color:var(--muted);font-size:13px;margin-top:35px}}@media(max-width:650px){{main{{padding:28px 14px 50px}}.case,.panel{{padding:20px 16px}}.comparison{{grid-template-columns:1fr}}.attempt summary{{flex-direction:column;gap:4px}}.stats{{gap:8px}}.stat{{padding:13px;min-width:130px}}}}
    </style></head><body><main><header><div class="eyebrow">Qwen3-4B · L19–T368 · 15 October 2026</div><h1>From made-up answers<br>to “I don’t know.”</h1><p class="intro">Three of four original made-up answers became coherent “I don’t know” responses at replayed question-specific strengths. The vampire-bat question did not convert at any of the 128 tested strengths (0.5–64 in steps of 0.5).</p><div class="stats"><div class="stat"><strong>{wins}/4</strong>original failures converted</div><div class="stat"><strong>{6 + wins}/10</strong>combined with six saved baseline withholdings</div><div class="stat"><strong>{len(distinct_vampire)}</strong>distinct vampire-bat strengths tested</div></div><p class="small">The six baseline “I don’t know” cases come from the previous run and are not rerun here. One has an inaccurate explanation: withholding counts and fully grounded explanations are different. This is a selected pilot, not a held-out test or one shared steering strength.</p></header>{''.join(cards)}<section class="panel"><div class="eyebrow">The remaining case</div><h2>Every vampire-bat answer, by strength</h2><p>Green: coherent, grounded “I don’t know.” Red: readable but fails clean abstention through an answer, false claim, or question echo. Amber: incoherent, repetitive, or unfinished. Click a strength to read the complete output and review. ↻ marks a repeated control or integer anchor in the fractional follow-up.</p><div class="tiles">{''.join(tiles)}</div>{''.join(attempts)}</section><section class="panel"><h2>Exactly what changed inside the model</h2><p><strong>L19–T368 — Onset of Refusal or Ignorance Statements</strong>, rank 128 (zero-based IDs). Same prompt and native model; attention remains unfrozen.</p><pre>{esc(equation)}</pre><p><strong>a</strong> is the displayed strength. <strong>z</strong> is the current residual after layer-19 attention and before its post-attention normalization, standardized using the released MOLT input statistics. The low-rank output is recomputed from the current trajectory at each processed output token. The native MLP is retained. We do not subtract the MOLT estimate of its natural contribution. There is no donor prompt or absolute gate clamp.</p><p>Prompt positions are untouched. The first output token is selected normally; the edit begins while processing that emitted token and affects later predictions. Every processed output-token position receives the edit. EOS and the final token of a capped output are not processed again.</p><p>Greedy decoding, thinking disabled, KV cache, batch one, 96-token cap; BF16 native model and FP32 MOLT math. Pinned PyTorch {esc(protocol['runtime']['torch'])}, Transformers {esc(protocol['runtime']['transformers'])}. Base checkpoint <code>{esc(protocol['base_revision'])}</code>; MOLT checkpoint <code>{esc(protocol['molt_revision'])}</code>.</p><details><summary>Passages</summary>{passages}</details></section><section class="panel"><h2>Controls, grading and limits</h2><p>Native and zero-addition controls and fixed prior positive anchors must reproduce saved token IDs exactly before results are accepted. Complete answers are reviewed against the passages. Uncertainty language alongside a fabricated answer or contradiction does not count as success. Severely broken outputs have null content-correctness judgments; they are not counted as zero hallucination or successful abstention. A grid search does not establish a continuous minimum, and a failed grid does not rule out every fractional setting.</p><p>The four cases use the original strict passage-only criterion. The crystallization answer can be a plausible domain inference, but is not established by this passage. This experiment measures whether a setting exists for these known failures; it does not provide a strength-selection rule for unseen questions or establish that this MOLT selectively controls hallucination.</p><p><a href="responses.csv">Every answer and judgment</a> · <a href="assessment.json">Hash-guarded reviews</a> · <a href="artifacts/all-results.json">All raw tokens and gate traces</a> · <a href="summary.json">Counts</a> · <a href="search-protocol.json">Both search protocols</a> · <a href="run.py">Runner</a> · <a href="report.py">Report source</a> · <a href="verification.json">Verification</a></p></section><footer>SPAR instance {rental['instance_id']} is retained at the user’s request, ${rental['hourly_with_disk_usd']:.3f}/hour including disk; transfers billed separately. Earlier failed rental was removed with explicit approval. Selected SQuAD2.0 text: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0. Complete-response review by Codex with condition labels visible; no independent human reviewer.</footer></main></body></html>'''
    (root / 'index.html').write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
