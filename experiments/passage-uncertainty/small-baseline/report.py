"""Build the complete baseline answer table from saved generations and manual judgments."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir',type=Path)
    args = parser.parse_args()
    root=args.result_dir
    data=json.loads((root/'cases.json').read_text())
    run=json.loads((root/'artifacts/generations.json').read_text())
    assessment=json.loads((root/'assessment.json').read_text())
    assert run['status']=='complete' and len(run['rows'])==20
    rows={r['id']:r for r in run['rows']}
    grades={r['id']:r for r in assessment['rows']}
    assert set(rows)==set(grades)=={r['id'] for r in data['cases']}
    assert all(r['stop_reason']=='eos' for r in rows.values())
    assert all(not g['incoherent'] for g in grades.values())
    joined=[]
    for case in data['cases']:
        row=rows[case['id']]
        grade=grades[case['id']]
        assert hashlib.sha256(row['answer'].encode()).hexdigest()==grade['answer_sha256']
        assert grade['wrong_answer'] is None if not case['is_answerable'] else isinstance(grade['wrong_answer'],bool)
        joined.append({**row,**grade})
    summaries=[]
    for pid in [p['id'] for p in data['passages']]+['All']:
        group=[r for r in joined if pid=='All' or r['passage_id']==pid]
        known=[r for r in group if r['is_answerable']]
        unknown=[r for r in group if not r['is_answerable']]
        wrong=sum(r['wrong_answer'] for r in known)
        hallucinations=sum(r['hallucination'] for r in unknown)
        summaries.append({'passage':pid,'answerable':len(known),'correct':len(known)-wrong,
             'wrong_answers':wrong,'wrong_answer_rate':wrong/len(known),
             'false_abstentions':sum(r['abstained'] for r in known),
             'unanswerable':len(unknown),'hallucinated':hallucinations,
             'hallucination_rate':hallucinations/len(unknown),
             'appropriate_abstention':sum(r['abstained'] and not r['hallucination'] for r in unknown),
             'supported_false_premise_corrections':sum(r['supported_false_premise_correction'] and not r['hallucination'] for r in unknown),
             'incoherent':sum(r['incoherent'] for r in group),
             'truncated':sum(r['stop_reason']=='token_limit' for r in group)})
    (root/'summary.json').write_text(json.dumps({'groups':summaries,'per_question_is_binary':True,
        'no_per_question_probability_estimate':True},indent=2)+'\n')
    fields=['id','passage_id','question','is_answerable','reference_answer','answer','verdict',
            'wrong_answer','hallucination','abstained','supported_false_premise_correction',
            'incoherent','stop_reason','grading_reason']
    with (root/'answers.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(joined)
    esc=html.escape
    overall=summaries[-1]
    cards=''.join(f'<div class="card"><strong>{value}</strong><span>{label}</span></div>' for value,label in [
       (f"{overall['correct']}/{overall['answerable']}",'Correct supported answers'),
       (f"{overall['wrong_answer_rate']:.0%}",f"Wrong answer rate · {overall['wrong_answers']}/{overall['answerable']} answerable"),
       (f"{overall['hallucination_rate']:.0%}",f"Hallucination rate · {overall['hallucinated']}/{overall['unanswerable']} unanswerable")])
    summaryrows=''.join('<tr>'+''.join(f'<td>{x}</td>' for x in [
       s['passage'],f"{s['correct']}/{s['answerable']}",f"{s['wrong_answers']}/{s['answerable']} ({s['wrong_answer_rate']:.0%})",
       f"{s['hallucinated']}/{s['unanswerable']} ({s['hallucination_rate']:.0%})",
       s['appropriate_abstention'],s['supported_false_premise_corrections']])+'</tr>' for s in summaries)
    sections=[]
    for passage in data['passages']:
        body=[]
        for r in joined:
            if r['passage_id']!=passage['id']:continue
            known=r['is_answerable']
            status='bad' if r['hallucination'] or r['wrong_answer'] else 'good' if known else 'abstain'
            wrong='Yes' if r['wrong_answer'] else 'No' if known else 'N/A'
            hall='Yes' if r['hallucination'] else 'No'
            body.append(f'<tr class="{status}"><td><b>{r["id"]}</b><small>{"Answer present" if known else "Answer absent"}</small></td>'
                f'<td>{esc(r["question"])}</td><td>{esc(r["reference_answer"])}</td>'
                f'<td class="answer">{esc(r["answer"])}</td><td>{wrong}</td><td>{hall}</td>'
                f'<td><b>{esc(r["verdict"])}</b><small>{esc(r["grading_reason"])}</small></td></tr>')
        sections.append(f'<section><h2>{passage["id"]} · {esc(passage["article_title"].replace("_"," "))}</h2>'
            f'<details open><summary>Read the passage used for every question below</summary><p class="passage">{esc(passage["text"])}</p></details>'
            '<div class="scroll"><table class="answers"><thead><tr><th>ID / ground truth</th><th>Question</th><th>Reference answer</th>'
            '<th>Complete model answer</th><th>Wrong answer?</th><th>Hallucination?</th><th>Manual judgment and evidence</th>'
            '</tr></thead><tbody>'+''.join(body)+'</tbody></table></div></section>')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Qwen3-4B · Two-passage grounding baseline</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#17283b;font:16px/1.6 system-ui,sans-serif}
    main{max-width:1500px;margin:auto;padding:36px 28px}h1{font-size:34px;line-height:1.2;margin:8px 0 16px}h2{margin-top:0;font-size:25px}
    a{color:#1359a6}.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:13px;color:#426481}.lead{max-width:1000px}
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:26px 0}.card{background:white;border:1px solid #d3dfea;border-radius:12px;padding:22px}
    .card strong{display:block;font-size:36px;line-height:1.2}.card span{display:block;margin-top:8px;color:#526273;font-size:14px}
    section{background:white;border:1px solid #d3dfea;border-radius:12px;padding:24px;margin:24px 0}
    .scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;padding:13px 12px;border-bottom:1px solid #dce4ed}
    th{background:#edf3fa;line-height:1.4}.answers{min-width:1150px;table-layout:fixed}.answers th:nth-child(1){width:8%}.answers th:nth-child(2){width:15%}
    .answers th:nth-child(3){width:12%}.answers th:nth-child(4){width:30%}.answers th:nth-child(5),.answers th:nth-child(6){width:7%}.answers th:nth-child(7){width:21%}
    td small{display:block;margin-top:7px;font-size:12px;color:#546373}.answer{white-space:pre-wrap}tr.bad td{background:#fff2f0}tr.good td:first-child{border-left:4px solid #338569}
    tr.abstain td:first-child{border-left:4px solid #367cb1}tr.bad td:first-child{border-left:4px solid #b95141}
    details{margin:18px 0 24px}summary{cursor:pointer;color:#185886}.passage{background:#f5f8fc;border-left:3px solid #6090bc;padding:18px;max-width:1100px}
    code{font-size:13px;overflow-wrap:anywhere}.fine{font-size:13px;color:#566879}.legend{font-size:14px;color:#526273}
    @media(max-width:700px){main{padding:22px 14px}h1{font-size:28px}.cards{grid-template-columns:1fr}.card{padding:16px}section{padding:16px}}
    </style></head><body><main><div class="eyebrow">14 October 2026 · Clean baseline · Native Qwen3-4B</div>
    <h1>Does the answer come from the passage?</h1><p class="lead">Two passages, five answerable and five unanswerable questions per passage: 20 independent conversations.
    Greedy decoding, thinking disabled, BF16, and unfrozen eager attention. No MOLT manipulation, hooks, or reconstruction was used.</p><div class="cards">'''+cards+'''</div>
    <section><h2>What the two rates mean</h2><p><b>Wrong answer rate:</b> answerable questions not correctly answered / answerable questions.
    False abstentions count as failures and are separately recorded. <b>Hallucination rate:</b> unanswerable questions receiving a substantive answer unsupported by the passage / unanswerable questions.
    An unsupported guess still counts if hedged. A supported correction of a false premise does not count as a hallucination.</p>
    <p>Each question has one deterministic response, so its table entries are yes/no judgments, not estimated probabilities. The percentages aggregate across questions.
    Unsupported by a passage does not necessarily mean false in the outside world. Internal confidence or intent is not inferred.</p>
    <div class="scroll"><table><thead><tr><th>Passage</th><th>Correct / answerable</th><th>Wrong answer rate</th><th>Hallucination rate</th><th>Appropriate abstentions</th><th>Supported premise corrections</th></tr></thead><tbody>'''+summaryrows+'''</tbody></table></div></section>
    <p class="legend">Green: correct answer. Blue: abstention or supported premise correction. Red: unsupported answer or incorrect answer. Scroll tables horizontally on a narrow screen.</p>
    '''+''.join(sections)+f'''
    <section><h2>Scope and reproducibility</h2><p>This is a small, manually selected pilot, not a representative benchmark score.
    Passages were chosen and references audited before inference. One original geology reference lists “igneous” for the immediate result of remelting; the passage explicitly says “new magma,” so that audited answer is used.
    All 20 responses reached EOS; no output was truncated or judged incoherent. The first question's repeat produced identical token IDs.</p>
    <p>Model revision: <code>{esc(run['protocol']['base_revision'])}</code>. Runtime: PyTorch {esc(run['runtime']['torch'])}, Transformers {esc(run['runtime']['transformers'])}.
    Cached model files were hash-verified against the prior pinned download manifest.</p>
    <p><a href="answers.csv">Full answer table (CSV)</a> · <a href="assessment.json">Manual judgments</a> · <a href="artifacts/generations.json">Raw generations and tokens</a> ·
    <a href="cases.json">Passages and original references</a> · <a href="protocol.json">Exact protocol</a> · <a href="run.py">Runner</a> · <a href="summary.json">Counts and rates</a></p>
    <p class="fine">SQuAD2.0 development text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0.
    <a href="https://rajpurkar.github.io/SQuAD-explorer/">Original dataset</a> · <a href="https://aclanthology.org/P18-2124/">Benchmark paper</a>.
    This study's manual grounding judgments are distinct from official span-based SQuAD exact-match scoring.</p></section></main></body></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
