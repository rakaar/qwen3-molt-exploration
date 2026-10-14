"""Four questions by seven additive coefficients; preserve native contribution and every response."""
import argparse
import csv
import hashlib
import html
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    run=json.loads((root/'artifacts/added-gate-results.json').read_text())
    assessment=json.loads((root/'assessment.json').read_text())
    cases=json.loads((root/'baseline-cases.json').read_text())
    grades={r['trial_id']:r for r in assessment['rows']}
    assert run['status']=='complete' and len(run['rows'])==len(grades)==36
    joined=[]
    for r in run['rows']:
        g=grades[r['trial_id']]
        assert hashlib.sha256(r['answer'].encode()).hexdigest()==g['answer_sha256']
        joined.append({**r,**g})
    P=run['protocol'];ids=P['cases'];strengths=[None]+P['added_gate_values']
    lookup={(r['case_id'],r['added_gate']):r for r in joined}
    names={'unsupported_answer':'Unsupported answer','question_echo':'Question echo; no abstention','coherent_abstention':'Coherent abstention'}
    shorts={'unsupported_answer':'Unsupported','question_echo':'Question echo','coherent_abstention':'Abstains'}
    colors={'unsupported_answer':'#f9d9d3','question_echo':'#dceaf8','coherent_abstention':'#bce5d1'}
    summaries=[]
    for a in strengths:
        group=[lookup[(i,a)] for i in ids]
        summaries.append({'added_gate':a,'label':'Clean' if a is None else str(a),'cases':len(group),
             'coherent_abstention_success':sum(r['coherent_abstention_success'] for r in group),
             'question_echo_without_abstention':sum(r['disposition']=='question_echo' for r in group),
             'unsupported_answers':sum(r['unsupported_substantive_answer'] for r in group),
             'coherent':sum(r['coherent'] for r in group),'truncated':sum(r['stop_reason']=='token_limit' for r in group)})
    controls=[r for r in joined if r['added_gate'] is None or r['added_gate']==0]
    assert len(controls)==8 and all(r['native_or_zero_matches_baseline'] for r in controls)
    positive=[r for r in joined if r['added_gate'] is not None and r['added_gate']>0]
    assert len(positive)==28
    summary={'scope':'Four previously unsupported questions, not all ten unanswerable questions.',
       'by_gate':summaries,'positive_trials':28,'positive_coherent':sum(r['coherent'] for r in positive),
       'positive_outputs_changed_from_native':sum(r['output_token_ids']!=lookup[(r['case_id'],None)]['output_token_ids'] for r in positive),
       'native_and_zero_controls_match':True,'all_first_tokens_unchanged':all(r['first_output_token_unchanged'] for r in joined),
       'all_trials_reached_eos':all(r['stop_reason']=='eos' for r in joined)}
    previous=json.loads((root/'previous-clamp-reference.json').read_text())
    comparisons=[{'case_id':old['case_id'],'coefficient':old['forced_gate'],'identical_output_token_ids':old['output_token_ids']==lookup[(old['case_id'],old['forced_gate'])]['output_token_ids']} for old in previous['rows']]
    summary['previous_clamp_comparison']=comparisons
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['trial_id','case_id','question','added_gate','answer','disposition','coherent','abstains',
            'unsupported_substantive_answer','contradicts_passage','coherent_abstention_success','stop_reason','grading_reason']
    with (root/'responses.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(joined)
    categories=['unsupported_answer','question_echo','coherent_abstention']
    matrix=np.array([[categories.index(lookup[(i,a)]['disposition']) for a in strengths] for i in ids])
    fig,ax=plt.subplots(figsize=(17,6.5));ax.imshow(matrix,cmap=ListedColormap([colors[c] for c in categories]),vmin=-.5,vmax=2.5,aspect='auto')
    for y,i in enumerate(ids):
        for x,a in enumerate(strengths):
            r=lookup[(i,a)];label=shorts[r['disposition']]+('*' if r['contradicts_passage'] else '')+'\nCoherent'
            ax.text(x,y,label,ha='center',va='center',fontsize=10,color='#17283b')
    ax.set_xticks(range(8));ax.set_xticklabels(['Clean']+[str(a) for a in strengths[1:]],fontsize=12)
    ax.xaxis.tick_top();ax.xaxis.set_label_position('top');ax.set_xlabel('Added coefficient a (effective estimated gate = natural gate + a)',fontsize=12,labelpad=12)
    labels=[i+'\n'+textwrap.fill(lookup[(i,None)]['question'].strip(),44) for i in ids]
    ax.set_yticks(range(4));ax.set_yticklabels(labels,fontsize=10)
    ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False,top=False)
    fig.suptitle('Add L19–T368 low-rank output at every output-token position\nOnset of Refusal or Ignorance Statements · Native Qwen3-4B · Unfrozen attention',fontsize=16,fontweight='bold',y=.98)
    fig.legend(handles=[Patch(facecolor=colors[c],label=names[c]) for c in categories],loc='lower center',bbox_to_anchor=(.6,.045),ncol=3,fontsize=10,frameon=False)
    fig.text(.02,.012,'All 28 positive-gate trials stayed coherent and reached EOS. *Starred vampire answers contradict the passage about rabies. Addition 7 echoes the question. Prompt computation unchanged.',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,.89]);fig.savefig(root/'added-gate-outcomes.png',dpi=170);fig.savefig(root/'added-gate-outcomes.pdf');plt.close(fig)
    esc=html.escape
    def cell(r,full=False):
        style=f'background:{colors[r["disposition"]]}'
        label=names[r['disposition']]+(' · contradicts passage' if r['contradicts_passage'] else '')
        body=f'<b>{esc(label)}</b><small>Coherent · {"EOS" if r["stop_reason"]=="eos" else "token cap"}</small>'
        if full:body+=f'<p class="answer">{esc(r["answer"])}</p><details><summary>Grading and edit details</summary><p>{esc(r["grading_reason"])}</p><p>Edited positions: {r["summary"]["edited_positions"]}/{r["processed_response_tokens"]}; maximum intended edit / native MLP norm: {r["summary"]["max_delta_over_native_mlp_norm"]:.3f}.</p></details>'
        return f'<td style="{style}" data-strength="{"clean" if r["added_gate"] is None else r["added_gate"]}">{body}</td>'
    headers='<th class="question">Question</th>'+''.join(f'<th data-strength="{"clean" if a is None else a}">{"Clean" if a is None else "Add "+str(a)}</th>' for a in strengths)
    def table(full):
        body=[]
        for i in ids:
            q=lookup[(i,None)]['question'].strip()
            body.append(f'<tr><th class="question">{i}<p>{esc(q)}</p></th>'+''.join(cell(lookup[(i,a)],full) for a in strengths)+'</tr>')
        return '<div class="scroll"><table class="'+('full' if full else 'matrix')+'"><thead><tr>'+headers+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    summarytable='<table><thead><tr><th>Added coefficient</th><th>Coherent abstentions</th><th>Question echo, no abstention</th><th>Unsupported answers</th><th>Coherent</th></tr></thead><tbody>'+''.join(f'<tr><td>{s["label"]}</td><td>{s["coherent_abstention_success"]}/4</td><td>{s["question_echo_without_abstention"]}/4</td><td>{s["unsupported_answers"]}/4</td><td>{s["coherent"]}/4</td></tr>' for s in summaries)+'</tbody></table>'
    options='<option value="all">All strengths</option>'+''.join(f'<option value="{a}">Clean + addition {a}</option>' for a in P['added_gate_values'])
    passages=''.join(f'<details><summary>{p["id"]} · {esc(p["article_title"].replace("_"," "))}</summary><p class="passage">{esc(p["text"])}</p></details>' for p in cases['passages'])
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>L19–T368 · Additive abstention sweep</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#17283b;font:16px/1.6 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:30px 24px}h1{font-size:34px;line-height:1.25}h2{font-size:24px}a{color:#145ba6}.eyebrow{font-size:13px;letter-spacing:.07em;color:#426481}
    section{background:white;border:1px solid #d3dfea;border-radius:12px;padding:24px;margin:24px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.card strong{display:block;font-size:32px}.card span{font-size:14px}.scroll{overflow-x:auto}table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:12px;border-right:1px solid white;border-bottom:1px solid #d6e0eb}thead th{background:#e8f0f9}
    .matrix{min-width:1200px;table-layout:fixed}.matrix .question{width:230px}.question{position:sticky;left:0;background:#edf3fa;z-index:1}.question p{font-weight:400;margin-bottom:0}.full{min-width:2400px;table-layout:fixed}.full .question{width:240px}.full th:not(.question),.full td{width:270px}td small{display:block;color:#43566b;margin-top:6px;font-size:12px}.answer{white-space:pre-wrap;font-size:14px}.figure{width:100%;height:auto}code{font:13px/1.5 monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f4f7fb;padding:16px;font:14px/1.6 monospace}details{margin:14px 0;font-size:14px}summary{cursor:pointer;color:#145ba6}.passage{background:#f4f7fb;padding:16px}.fine{font-size:13px;color:#536678}select{font:inherit;padding:6px 10px;margin-left:8px}.legend{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}.legend span{padding:6px 10px;border-radius:4px}
    @media(max-width:700px){main{padding:18px 12px}h1{font-size:27px}section{padding:16px}.cards{grid-template-columns:1fr}}
    </style></head><body><main><div class="eyebrow">15 OCTOBER 2026 · FOUR FIXED QUESTIONS · UNFROZEN ATTENTION</div>
    <h1>Does adding L19–T368 output improve abstention?</h1><p>Dashboard label: <b>Onset of Refusal or Ignorance Statements</b>. Added coefficients 3, 4, 5, 6, 7, and 8 were tested on each of the four previously unsupported-answer cases, with 3.5 as a comparison anchor. Every condition uses the same passage, question, prompt, checkpoint, and greedy decoding.</p>
    <section><div class="cards"><div class="card"><strong>1/4</strong><span>Questions abstaining at every tested addition</span></div><div class="card"><strong>28/28</strong><span>Positive-gate responses remained coherent</span></div><div class="card"><strong>8/8</strong><span>Native and zero controls reproduced saved answers</span></div></div>
    <p>The electric-eel case abstains at every tested addition from 3 through 8, including 3.5. The other three questions do not abstain. At addition 7 the vampire response echoes the question; that is separately labelled, not counted as abstention.</p>
    <p>All responses reached EOS within 96 tokens. Coherence means understandable, fluent output—not factual correctness. At additions 3.5, 4, 5, 6, and 8, the vampire response denies that bats spread rabies, contradicting the supplied passage.</p></section>
    <section><h2>Outcome matrix</h2><p>Rows are the four questions. Columns are the added coefficients; the natural contribution is retained. Each cell also records coherence.</p>
    <div class="legend"><span style="background:#bce5d1">Coherent abstention</span><span style="background:#dceaf8">Question echo; no abstention</span><span style="background:#f9d9d3">Unsupported answer</span></div>'''+table(False)+'''
    <p><a href="added-gate-outcomes.png">Outcome matrix PNG</a> · <a href="added-gate-outcomes.pdf">PDF</a></p></section>
    <section><h2>Complete response table</h2><p>Scroll horizontally to compare all strengths, or select one strength alongside the clean response. Every complete response is retained.</p>
    <label>Show <select id="strength-filter">'''+options+'''</select></label>'''+table(True)+'''</section>
    <section><h2>Counts by strength</h2><p>Denominators are these four selected questions. These are not new scores on all ten unanswerable baseline questions.</p><div class="scroll">'''+summarytable+'''</div></section>
    <section><h2>Exact additive edit and token timing</h2><pre>z = current residual after attention 19, before its post-attention layernorm,
    standardized using released MOLT input mean and standard deviation
    L(z) = ((z @ V_368) @ U_368) × output_std_19
    natural gate = JumpReLU(z dot e_368 + bias_368)
    MLP output += a × L(z)
    effective estimated gate = natural gate + a</pre>
    <p>The native MLP is retained and receives a positive addition. The complete low-rank output is recomputed from each edited trajectory’s current input. The natural gate is measured for logging, and is never subtracted or used to reduce the added coefficient. There is no donor input, frozen clean vector, or U/V weight change.</p>
    <p><b>The added coefficient a is the same at every processed output position.</b> The total estimated gate varies as natural gate + a. At addition 8, the eel response reaches an effective estimated gate of 10.6412, confirming that its natural activation is preserved. This differs from the previous clamp, which kept the total estimated gate equal to a.</p>
    <p>Every processed output-token position is edited, starting with output token 0. Prompt positions are untouched. The first output token is therefore selected normally; the edit while processing that token affects subsequent predictions. EOS ends generation and is not processed further.</p>
    <p>Native and zero-addition controls match the earlier output token IDs exactly. All strengths have the same first output token as the clean answer. Actual BF16 edit norms and every gate/input-position trace are preserved, including in cases whose greedy wording did not change.</p></section>
    <section><h2>Comparison with the earlier clamp</h2><p>At shared coefficients 3 and 3.5, all eight new responses are token-identical to the earlier clamp responses. This is a comparison of greedy output tokens, not equality of hidden states or logits. The earlier operator was (a − natural gate) × L; the new operator is a × L. The input-dependent vectors and natural activity are retained in the traces.</p><p><a href="previous-clamp-reference.json">Fixed earlier reference</a> · <a href="../2026-10-15-passage-forced-gates/index.html">Earlier full clamp table</a></p></section><section><h2>Passages and limits</h2>'''+passages+'''
    <p>This sweep changes abstention behaviour for one of the four prompts in the tested range. The earlier activation association does not guarantee that adding this input-dependent transform output produces abstention in every context. Uniform additions are denser than natural sparse activity. Feature-specificity against unrelated, magnitude-matched edits remains to be tested.</p>
    <p>Greedy decoding, BF16 native Qwen3-4B, FP32 MOLT math, unfrozen eager attention, thinking disabled. Complete-response judgments separate abstention, unsupported answers, question echoes, and coherence. A grammatical question echo without answering or communicating inability to determine is not successful abstention.</p>
    <p>Base revision: <code>'''+esc(P['base_revision'])+'''</code><br>MOLT revision: <code>'''+esc(P['molt_revision'])+'''</code><br>Selected asset SHA256: <code>'''+esc(P['asset_sha256'])+'''</code></p>
    <p><a href="responses.csv">All responses and judgments (CSV)</a> · <a href="assessment.json">Full grading explanations</a> · <a href="summary.json">Summary</a> · <a href="artifacts/added-gate-results.json">Raw generations and edit traces</a> · <a href="protocol.json">Protocol</a> · <a href="run.py">Runner</a> · <a href="report.py">Table and figure code</a></p>
    <p><a href="../2026-10-14-passage-answer-gates/index.html">Earlier answer-token gate heatmaps</a> · <a href="../2026-10-14-passage-grounding-baseline/index.html">Original baseline and references</a></p>
    <p class="fine">Selected SQuAD2.0 text: Pranav Rajpurkar, Robin Jia, and Percy Liang (2018), CC BY-SA 4.0. <a href="https://rajpurkar.github.io/SQuAD-explorer/">Source</a>. Unsupported means unsupported by the supplied passage, not necessarily false outside it. SPAR GPU retained as requested.</p></section></main>
    <script>document.getElementById('strength-filter').addEventListener('change',function(){document.querySelectorAll('.full [data-strength]').forEach(cell=>{cell.hidden=this.value!=='all'&&cell.dataset.strength!=='clean'&&cell.dataset.strength!==this.value;});document.querySelector('.full').style.minWidth=this.value==='all'?'2400px':'800px';});</script></body></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
