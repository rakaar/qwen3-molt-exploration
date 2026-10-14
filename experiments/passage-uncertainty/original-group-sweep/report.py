"""Local report: three scatter panels, fixed baseline cohorts, and complete responses."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    root=parser.parse_args().result_dir
    run=json.loads((root/'artifacts/added-gate-results.json').read_text())
    A=json.loads((root/'assessment.json').read_text())
    G=json.loads((root/'baseline-groups.json').read_text())
    D=json.loads((root/'cases.json').read_text())
    grades={r['trial_id']:r for r in A['rows']};joined=[]
    for r in run['rows']:
        g=grades[r['trial_id']];assert hashlib.sha256(r['answer'].encode()).hexdigest()==g['answer_sha256']
        joined.append({**r,**g})
    assert run['status']=='complete' and len(joined)==70
    P=run['protocol'];strengths=[None]+P['added_gate_values'];ids=P['cases']
    lookup={(r['case_id'],r['added_gate']):r for r in joined}
    group_names={'clean_unsupported':'Clean model gave an unsupported answer','clean_withheld':'Clean model already withheld the answer'}
    def summarize(group,a):
        rows=[lookup[(i,a)] for i in ids if group=='all' or lookup[(i,None)]['baseline_group']==group]
        result={'group':group,'added_coefficient':a,'label':'Clean' if a is None else str(a),'n':len(rows),
            **{k:sum(r[k] is True for r in rows) for k in ['withholding_decision','coherent_withholding','fully_grounded_withholding','unsupported_substantive_answer','mixed_answer_abstention','coherent','explanation_error']},
            'answer_assessable':sum(r['answer_assessable'] for r in rows),
            'truncated':sum(r['stop_reason']=='token_limit' for r in rows)}
        if result['answer_assessable']<result['n']:
            for key in ['unsupported_substantive_answer','mixed_answer_abstention','explanation_error']:result[key]=None
        return result
    summaries=[summarize(g,a) for g in ['all']+list(group_names) for a in strengths]
    controls=[r for r in joined if r['added_gate'] is None or r['added_gate']==0]
    assert len(controls)==20 and all(r['native_or_zero_matches_baseline'] for r in controls)
    assert run['native_repeat']['identical_output_token_ids']
    summary={'scope':'Ten reused original pilot questions. Baseline groups held fixed at four unsupported and six withheld. Previously tuned examples; exploratory, not held-out generalization.',
        'baseline_groups':G['counts'],'by_group_and_strength':summaries,'positive_trials':50,
        'native_zero_controls_match':20,'native_first_case_repeat_matches':True,
        'coherence_denominator':'All questions in each fixed cohort at each strength; incoherent and truncated outputs are retained.',
        'mixed_definition':'Unsupported substantive answer plus abstention language; also counted as unsupported.',
        'withholding_definition':A['withholding_definition']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['trial_id','case_id','baseline_group','question','added_gate','answer','answer_assessable','withholding_decision','coherent_withholding','fully_grounded_withholding','unsupported_substantive_answer','mixed_answer_abstention','coherent','explanation_error','stop_reason','grading_reason']
    with (root/'responses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(joined)
    panels=[('coherent','Coherence'),('unsupported_substantive_answer','Hallucination: unsupported target answer'),('withholding_decision','Abstention / “I don’t know” decision')]
    palette={'clean_unsupported':'#2874a6','clean_withheld':'#c77922'}
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,(key,title) in zip(axes,panels):
        for j,(group,name) in enumerate(group_names.items()):
            rows=[summarize(group,a) for a in strengths];n=rows[0]['n'];shift=-.16 if j==0 else .16
            points=[]
            for x,r in enumerate(rows):
                if key=='unsupported_substantive_answer' and r['answer_assessable']<n:continue
                count=r[key];y=100*count/n;points.append((x+shift,y))
                ax.annotate(f'{count}/{n}',(x+shift,y),xytext=(0,-18 if y>=90 else 10),textcoords='offset points',ha='center',fontsize=9,color=palette[group])
            ax.scatter([p[0] for p in points],[p[1] for p in points],s=n*28,color=palette[group],marker='o' if j==0 else 's',alpha=.9,label=f'{name} (n={n})')
        ax.set_xticks(range(6));ax.set_xticklabels(['Clean']+[str(a) for a in P['added_gate_values']])
        ax.set_xlim(-.5,5.5);ax.set_ylim(-8,110);ax.set_yticks([0,25,50,75,100]);ax.set_title(title,fontsize=12)
        ax.set_xlabel('Added coefficient a')
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Percent of questions in each fixed baseline group')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=10,bbox_to_anchor=(.5,.07))
    fig.suptitle('L19–T368 additive sweep · Original Geology / Amazon passages\nNative Qwen3-4B · Every processed output token · Unfrozen attention',fontsize=14,fontweight='bold',y=.99)
    fig.text(.015,.015,'No connecting lines. Point area reflects group size (4 or 6). Labels show counts. Incoherent answer correctness is unassessable; those answer-rate points are omitted.',fontsize=9)
    fig.tight_layout(rect=[0,.17,1,.86]);fig.savefig(root/'three-scatter-panels.png',dpi=170);fig.savefig(root/'three-scatter-panels.pdf');plt.close(fig)
    names={'coherent_abstention':'Abstains','supported_premise_correction':'Valid premise correction','abstention_explanation_error':'Abstains; explanation error','unsupported_answer':'Unsupported answer','mixed_answer_abstention':'False answer + abstention','repetitive_abstention':'Abstains; substantial repetition','incoherent':'Incoherent'}
    colors={'coherent_abstention':'#bce5d1','supported_premise_correction':'#bce5d1','abstention_explanation_error':'#d9e8bf','unsupported_answer':'#f9d9d3','mixed_answer_abstention':'#ffe4b0','repetitive_abstention':'#ddd4ee','incoherent':'#ddd4ee'}
    esc=html.escape
    def response_table(full):
        heads='<th class="question">Question / baseline group</th>'+''.join(f'<th data-strength="{"clean" if a is None else a}">{"Clean" if a is None else "Add "+str(a)}</th>' for a in strengths)
        body=[]
        for i in ids:
            clean=lookup[(i,None)];group=clean['baseline_group']
            row=f'<tr data-group="{group}"><th class="question">{i}<p>{esc(clean["question"])}</p><small>{esc(group_names[group])}</small></th>'
            for a in strengths:
                r=lookup[(i,a)]
                content=f'<b>{esc(names[r["disposition"]])}</b><small>{"Coherent" if r["coherent"] else "Not coherent"} · {esc(r["stop_reason"])}</small>'
                if full:content+=f'<p class="answer">{esc(r["answer"])}</p><details><summary>Judgment and edit details</summary><p>{esc(r["grading_reason"])}</p><p>Nonzero BF16 edits: {r["summary"]["edited_positions"]}/{r["processed_response_tokens"]}. Peak edit / native MLP norm: {r["summary"]["max_delta_over_native_mlp_norm"]:.3f}.</p></details>'
                row+=f'<td data-strength="{"clean" if a is None else a}" style="background:{colors[r["disposition"]]}">{content}</td>'
            body.append(row+'</tr>')
        return '<div class="scroll"><table class="'+('full' if full else 'matrix')+'"><thead><tr>'+heads+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    def counts_table(group):
        rows=[summarize(group,a) for a in strengths]
        return '<div class="scroll"><table class="counts"><thead><tr><th>Addition</th><th>Abstention decision</th><th>Coherent withholding</th><th>Grounded withholding</th><th>Unsupported answers</th><th>Mixed responses</th><th>Coherent</th><th>Token cap</th></tr></thead><tbody>'+''.join('<tr><td>'+r['label']+'</td>'+''.join('<td>N/A · incoherent</td>' if k in ['unsupported_substantive_answer','mixed_answer_abstention'] and r['answer_assessable']<r['n'] else f'<td>{r[k]}/{r["n"]}</td>' for k in ['withholding_decision','coherent_withholding','fully_grounded_withholding','unsupported_substantive_answer','mixed_answer_abstention','coherent','truncated'])+'</tr>' for r in rows)+'</tbody></table></div>'
    tables=''.join('<section><h2>'+esc(name)+' · n = '+str(G['counts'][g])+'</h2>'+counts_table(g)+'</section>' for g,name in group_names.items())
    passages=''.join(f'<details><summary>{p["id"]} · {esc(p["article_title"].replace("_"," "))}</summary><p class="passage">{esc(p["text"])}</p></details>' for p in D['passages'])
    audit=''.join(f'<details><summary>{c["id"]} · {esc(c["question"])}</summary><p>{esc(c["reference_audit_note"])}</p></details>' for c in D['cases'])
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Original questions · Three MOLT scatter panels</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#17283b;font:16px/1.6 system-ui,sans-serif}main{max-width:1550px;margin:auto;padding:26px 22px}h1{font-size:32px;line-height:1.25}h2{font-size:23px}section{background:white;border:1px solid #d3dfea;border-radius:12px;padding:22px;margin:22px 0}a{color:#145ba6}.eyebrow,small{font-size:13px;color:#426481}.scroll{overflow-x:auto}table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}th,td{padding:12px;vertical-align:top;text-align:left;border-right:1px solid white;border-bottom:1px solid #d6e0eb}thead th{background:#e8f0f9}.question{position:sticky;left:0;background:#edf3fa;z-index:1}.question p{font-weight:400}.matrix{min-width:1200px;table-layout:fixed}.matrix .question{width:270px}.full{min-width:2150px;table-layout:fixed}.full .question{width:280px}.full td{width:340px}.counts{min-width:700px}.answer{white-space:pre-wrap}small{display:block;margin-top:6px}details{margin:12px 0}summary{cursor:pointer;color:#145ba6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;padding:16px;font:14px/1.6 monospace}.passage{background:#f4f7fb;padding:16px}code{overflow-wrap:anywhere}select{font:inherit;padding:6px 10px;margin:4px 8px 4px 0}.figure{width:100%;height:auto}@media(max-width:700px){main{padding:16px 12px}section{padding:16px}h1{font-size:26px}}</style></head><body><main>
    <div class="eyebrow">15 OCTOBER 2026 · LOCAL REPORT · ORIGINAL TWO-PASSAGE PILOT</div><h1>Coherence, unsupported answers, and abstention as MOLT strength increases</h1>
    <p>Ten unanswerable questions from the original Geology and Amazon passages. All ten receive clean and added coefficients 4, 8, 16, 32, and 64, using L19–T368: Onset of Refusal or Ignorance Statements.</p>
    <section><p><b>Fixed clean-response groups:</b> four unsupported answers and six clean withholdings. The six include one baseline response with an inaccurate explanation, tracked separately. These groups are preserved at every strength.</p><p>This reuses previously tuned pilot examples at the user's request. It provides more conversion cases than the fresh Harvard/pharmacy screen, which had only one unsupported clean answer; it is not held-out generalization.</p><p>Incoherent outputs remain in the coherence and successful-abstention denominators. Their substantive answer correctness is unassessable. Mixed false answers with uncertainty count as unsupported, not clean abstention. Explanation errors are scored separately.</p></section>
    <section><h2>Three scatter plots</h2><img class="figure" src="three-scatter-panels.png" alt="Three scatter-only panels for coherence, unsupported answers, and abstention across fixed baseline groups"><p>Blue circles: four questions with unsupported clean answers. Orange squares: six questions with clean withholdings. Point area is proportional to the group size; counts are printed beside each point. Groups are offset within each strength for visibility. There are no connecting lines. The abstention panel measures withholding the requested answer without an unsupported target claim; coherence is assessed independently. Repetitive but assessable abstentions can remain on the right while failing the left panel.</p><p><a href="three-scatter-panels.png">PNG</a> · <a href="three-scatter-panels.pdf">PDF</a></p></section>'''+tables+'''
    <section><h2>All-ten summary</h2>'''+counts_table('all')+'''<p>Withholding includes explicit abstention and supported false-premise correction. Fully grounded withholding additionally requires that its explanation contains no unsupported or contradicted passage claim. Coherence measures intelligibility and substantial repetition/breakdown; three consecutive identical complete sentences count as repetitive degradation. Factual correctness is graded separately. The hallucination panel counts unsupported target answers; errors in an abstention explanation are additionally flagged here.</p></section>
    <section><h2>Outcome matrix</h2>'''+response_table(False)+'''</section>
    <section><h2>Every complete answer</h2><label>Group <select id="group-filter"><option value="all">Both groups</option><option value="clean_unsupported">Clean unsupported answer</option><option value="clean_withheld">Clean already withheld</option></select></label><label>Strength <select id="strength-filter"><option value="all">All strengths</option>'''+''.join(f'<option value="{a}">Clean + addition {a}</option>' for a in strengths[1:])+'''</select></label>'''+response_table(True)+'''</section>
    <section><h2>Exact additive edit</h2><pre>z = current residual after attention 19, before post-attention layernorm,
    standardized with released MOLT input mean and standard deviation
L(z) = ((z @ V_368) @ U_368) × output_std_19
MLP output += a × L(z)
Effective estimated gate = natural gate + a</pre><p>Native MLP retained; uniform added coefficient, with complete low-rank output recomputed from the current input at each position. The natural gate is measured and preserved. No donor, subtraction, clamp, frozen clean vector, or U/V change. Attention remains unfrozen.</p><p>All prompt tokens are native. The first output token is selected normally; edits begin while processing output token 0 and influence later predictions. Every output position processed to predict a next token receives the edit. On EOS runs, the last visible token is processed to predict EOS. On capped runs, the final generated token is not processed further. EOS itself is not processed again.</p><p>BF16 native Qwen3-4B, FP32 MOLT math, thinking disabled, greedy batch-one decoding, KV cache, unchanged 96-token cap. Ten native and ten zero-addition responses reproduce the original clean token IDs exactly; one extra native repeat also matches. Every natural gate, position, low-rank norm, intended edit and actual BF16 edit is preserved.</p><p>Base revision: <code>'''+P['base_revision']+'''</code><br>MOLT revision: <code>'''+P['molt_revision']+'''</code><br>Asset SHA256: <code>'''+P['asset_sha256']+'''</code></p></section>
    <section><h2>Passages and question audit</h2>'''+passages+audit+'''<p>Original benchmark questions and the earlier passage-only judgments are retained. The crystallization concept answer is a plausible domain inference, but is not established by this passage under the strict benchmark criterion. No question is dropped because of an intervention result.</p></section>
    <section><h2>Scientific record and limits</h2><p>This is an exploratory repeat on ten original pilot questions, including four cases already used to tune strength. It does not estimate held-out performance. Answerable-question over-refusal checks and unrelated magnitude-matched interventions remain future work.</p><p><a href="responses.csv">All 70 responses and judgments</a> · <a href="assessment.json">Manual judgments</a> · <a href="baseline-groups.json">Fixed baseline cohorts</a> · <a href="summary.json">Counts</a> · <a href="artifacts/added-gate-results.json">Raw generations and traces</a> · <a href="cases.json">Audited questions</a> · <a href="protocol.json">Protocol</a> · <a href="run.py">Runner</a> · <a href="report.py">Figure/report code</a></p><p class="eyebrow">Published report. SPAR GPU and attached disk removed on 15 October 2026; ongoing compute/storage billing verified at $0/hour. <a href="../2026-10-15-vast-teardown/teardown.json">Teardown verification</a>. Selected SQuAD2.0 text: Rajpurkar, Jia, and Liang (2018), CC BY-SA 4.0.</p></section></main>
    <script>function update(){let strength=document.getElementById('strength-filter').value,group=document.getElementById('group-filter').value;document.querySelectorAll('.full [data-strength]').forEach(c=>{c.hidden=strength!=='all'&&c.dataset.strength!=='clean'&&c.dataset.strength!==strength;});document.querySelectorAll('.full tbody tr').forEach(r=>{r.hidden=group!=='all'&&r.dataset.group!==group;});document.querySelector('.full').style.minWidth=strength==='all'?'2150px':'800px';}document.getElementById('strength-filter').addEventListener('change',update);document.getElementById('group-filter').addEventListener('change',update);</script></body></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps({'baseline_groups':G['counts'],'all_ten_by_strength':[r for r in summaries if r['group']=='all']},indent=2))

if __name__=='__main__':main()
