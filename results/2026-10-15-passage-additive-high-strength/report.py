"""Build the local HTML response table and static scientific outcome matrix."""
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
    P=run['protocol'];ids=P['cases'];strengths=[None]+P['added_gate_values']
    grades={r['trial_id']:r for r in assessment['rows']};joined=[]
    assert run['status']=='complete' and len(run['rows'])==len(grades)==18
    for r in run['rows']:
        g=grades[r['trial_id']]
        assert hashlib.sha256(r['answer'].encode()).hexdigest()==g['answer_sha256']
        joined.append({**r,**g})
    lookup={(r['case_id'],r['added_gate']):r for r in joined}
    names={'unsupported_answer':'Unsupported answer','coherent_abstention':'Coherent abstention','mixed_answer_abstention':'False answer + abstention language'}
    colors={'unsupported_answer':'#f9d9d3','coherent_abstention':'#bce5d1','mixed_answer_abstention':'#ffe4b0'}
    summaries=[]
    for a in strengths:
        group=[lookup[(i,a)] for i in ids]
        summaries.append({'added_coefficient':a,'label':'Clean' if a is None else str(a),'cases':3,
         'clean_abstentions':sum(r['coherent_abstention_success'] for r in group),
         'mixed_answer_abstention':sum(r['mixed_answer_and_abstention'] for r in group),
         'unsupported_answers':sum(r['unsupported_substantive_answer'] for r in group),
         'fluent':sum(r['fluent_and_understandable'] for r in group),'truncated':sum(r['stop_reason']=='token_limit' for r in group)})
    controls=[r for r in joined if r['added_gate'] is None or r['added_gate']==0]
    positive=[r for r in joined if r['added_gate'] is not None and r['added_gate']>0]
    anchors=[r for r in joined if r['added_gate']==8]
    assert len(controls)==6 and all(r['native_or_zero_matches_baseline'] for r in controls)
    assert len(anchors)==3 and all(r['matches_previous_eight'] for r in anchors)
    summary={'scope':'Three previously non-abstaining questions only. Eel question not retested.','by_strength':summaries,
      'positive_trials':12,'positive_fluent':sum(r['fluent_and_understandable'] for r in positive),
      'all_trials_reached_eos':all(r['stop_reason']=='eos' for r in joined),
      'native_zero_controls_match':True,'previous_eight_anchors_match':True,
      'all_first_output_tokens_unchanged':all(r['first_output_token_unchanged'] for r in joined)}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['trial_id','case_id','question','added_gate','answer','disposition','fluent_and_understandable','abstains','unsupported_substantive_answer','unsupported_explanation','contradicts_passage','coherent_abstention_success','stop_reason','grading_reason']
    with (root/'responses.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(joined)
    cats=list(names);matrix=np.array([[cats.index(lookup[(i,a)]['disposition']) for a in strengths] for i in ids])
    fig,ax=plt.subplots(figsize=(13,5.5));ax.imshow(matrix,cmap=ListedColormap([colors[c] for c in cats]),vmin=-.5,vmax=2.5,aspect='auto')
    short={'unsupported_answer':'Unsupported\nFluent','coherent_abstention':'Abstains\nFluent','mixed_answer_abstention':'Mixed answer\n+ uncertainty'}
    for y,i in enumerate(ids):
        for x,a in enumerate(strengths):
            r=lookup[(i,a)];label=short[r['disposition']]+('*' if r['unsupported_explanation'] and r['coherent_abstention_success'] else '')
            ax.text(x,y,label,ha='center',va='center',fontsize=11,color='#17283b')
    ax.set_xticks(range(len(strengths)));ax.set_xticklabels(['Clean']+[str(a) for a in strengths[1:]],fontsize=12)
    ax.xaxis.tick_top();ax.xaxis.set_label_position('top');ax.set_xlabel('Added coefficient a; natural contribution retained',labelpad=12,fontsize=12)
    ax.set_yticks(range(len(ids)));ax.set_yticklabels([i+'\n'+textwrap.fill(lookup[(i,None)]['question'].strip(),40) for i in ids],fontsize=10)
    ax.set_xticks(np.arange(-.5,len(strengths),1),minor=True);ax.set_yticks(np.arange(-.5,len(ids),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False,top=False)
    fig.suptitle('Higher additive strengths: L19–T368 · Onset of Refusal or Ignorance Statements\nNative Qwen3-4B · Every processed output token · Unfrozen attention',fontsize=14,fontweight='bold',y=.98)
    fig.legend(handles=[Patch(facecolor=colors[c],label=names[c]) for c in cats],loc='lower center',bbox_to_anchor=(.56,.065),ncol=3,fontsize=9,frameon=False)
    fig.text(.015,.015,'All 12 steered responses are fluent and reach EOS. Mixed responses are not clean abstentions. *Abstains, but explanation overclaims passage scope.',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.85]);fig.savefig(root/'high-strength-outcomes.png',dpi=170);plt.close(fig)
    esc=html.escape
    def table(full):
        headers='<th class="question">Question</th>'+''.join(f'<th data-strength="{"clean" if a is None else a}">{"Clean" if a is None else "Add "+str(a)}</th>' for a in strengths)
        body=[]
        for i in ids:
            row=f'<tr><th class="question">{i}<p>{esc(lookup[(i,None)]["question"].strip())}</p></th>'
            for a in strengths:
                r=lookup[(i,a)];label=names[r['disposition']]
                if r['unsupported_explanation'] and r['coherent_abstention_success']:label+=' · explanation overclaims'
                content=f'<b>{esc(label)}</b><small>Fluent / intelligible · {esc(r["stop_reason"])}</small>'
                if full:content+=f'<p class="answer">{esc(r["answer"])}</p><details><summary>Judgment and edit details</summary><p>{esc(r["grading_reason"])}</p><p>Nonzero edited positions: {r["summary"]["edited_positions"]}/{r["processed_response_tokens"]}. Peak intended edit / native MLP norm: {r["summary"]["max_delta_over_native_mlp_norm"]:.3f}.</p></details>'
                row+=f'<td data-strength="{"clean" if a is None else a}" style="background:{colors[r["disposition"]]}">{content}</td>'
            body.append(row+'</tr>')
        return '<div class="scroll"><table class="'+('full' if full else 'matrix')+'"><thead><tr>'+headers+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    counts='<table><thead><tr><th>Added coefficient</th><th>Clean abstentions</th><th>Mixed answer + abstention</th><th>Fluent</th></tr></thead><tbody>'+''.join(f'<tr><td>{r["label"]}</td><td>{r["clean_abstentions"]}/3</td><td>{r["mixed_answer_abstention"]}/3</td><td>{r["fluent"]}/3</td></tr>' for r in summaries)+'</tbody></table>'
    cases=json.loads((root/'baseline-cases.json').read_text())
    passages=''.join(f'<details><summary>{p["id"]} · {esc(p["article_title"].replace("_"," "))}</summary><p class="passage">{esc(p["text"])}</p></details>' for p in cases['passages'])
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>L19–T368 · Additive strengths 8–32</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#17283b;font:16px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:26px 22px}h1{font-size:32px;line-height:1.25}h2{font-size:23px}section{background:white;border:1px solid #d3dfea;border-radius:12px;padding:22px;margin:22px 0}a{color:#145ba6}.eyebrow,small{font-size:13px;color:#426481}.scroll{overflow-x:auto}table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px}th,td{padding:12px;vertical-align:top;text-align:left;border-right:1px solid white;border-bottom:1px solid #d6e0eb}thead th{background:#e8f0f9}.question{position:sticky;left:0;background:#edf3fa;z-index:1}.question p{font-weight:400}.matrix{min-width:1000px;table-layout:fixed}.matrix .question{width:250px}.full{min-width:1800px;table-layout:fixed}.full .question{width:260px}.full td{width:340px}.answer{white-space:pre-wrap}small{display:block;margin-top:6px}details{margin:12px 0}summary{cursor:pointer;color:#145ba6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fb;padding:16px;font:14px/1.6 monospace}.passage{background:#f4f7fb;padding:16px}.legend{display:flex;gap:12px;flex-wrap:wrap;margin:15px 0}.legend span{padding:6px 10px;border-radius:4px}select{font:inherit;padding:6px 10px}.figure{width:100%;height:auto}@media(max-width:700px){main{padding:16px 12px}section{padding:16px}h1{font-size:26px}}</style></head><body><main>
    <div class="eyebrow">15 OCTOBER 2026 · LOCAL REPORT · THREE FIXED QUESTIONS</div><h1>Higher strengths convert two more questions to abstention</h1>
    <p>Qwen3-4B · L19–T368: <b>Onset of Refusal or Ignorance Statements</b>. Additions 8, 16, 24, and 32 on the three cases that did not abstain in the previous sweep. The electric-eel case is not retested here.</p>
    <section><p><b>Minerals:</b> coherent abstention at 16, 24, and 32. <b>Crystallization:</b> coherent abstention at 24 and 32. <b>Vampire bats:</b> at 24 and 32 it mixes a false rabies answer with uncertainty/abstention language; that is not counted as a clean abstention.</p>
    <p>All 12 steered responses are fluent and understandable and reach EOS within the unchanged 96-token cap. Fluency does not imply factual consistency. The minerals response at 32 abstains but incorrectly narrows the passage to re-deposition; the vampire response at 32 has an unclear parenthetical self-correction.</p>
    <p>Native/zero controls reproduce the earlier saved token IDs (6/6). Repeated addition-8 outputs reproduce the previous sweep (3/3). Higher strengths genuinely make nonzero edits at every processed output position.</p></section>
    <section><h2>Outcome matrix</h2><div class="legend"><span style="background:#bce5d1">Coherent abstention</span><span style="background:#f9d9d3">Unsupported answer</span><span style="background:#ffe4b0">False answer + abstention language</span></div>'''+table(False)+'''<p><a href="high-strength-outcomes.png">Matrix PNG</a></p></section>
    <section><h2>Complete answers</h2><p>Scroll horizontally, or select one addition alongside the clean answer.</p><label>Show <select id="strength-filter"><option value="all">All strengths</option>'''+''.join(f'<option value="{a}">Clean + addition {a}</option>' for a in strengths[1:])+'''</select></label>'''+table(True)+'''</section>
    <section><h2>Counts for these three questions</h2><div class="scroll">'''+counts+'''</div><p>Sixteen and 24 are the lowest successful tested additions for minerals and crystallization respectively, not proven thresholds. We did not test the eel question at these higher strengths or assess all ten original unanswerable questions.</p></section>
    <section><h2>Exact edit and token timing</h2><pre>z = current residual after attention 19, before post-attention layernorm,
    standardized with released MOLT input mean and standard deviation
L(z) = ((z @ V_368) @ U_368) × output_std_19
MLP output += a × L(z)
Effective estimated gate = natural gate + a</pre>
    <p>The native MLP and natural contribution remain in place. The added scalar is uniform across processed output positions; the complete low-rank output is recomputed from each trajectory's current input. It is added even when the natural gate is zero. There is no donor, subtraction, clamp, fixed clean vector, or U/V weight change.</p>
    <p>Attention remains unfrozen. No prompt token is edited. Output token 0 is selected normally; editing while processing token j influences later predictions. Every visible output token is processed with the edit. EOS terminates generation and is not processed again.</p>
    <p>BF16 native model, FP32 MOLT math, greedy decoding, thinking disabled, KV cache, batch one. Intended and actual BF16 edit norms are retained separately. Large edits do not establish that this transform is a selective or universal abstention mechanism; unrelated magnitude-matched controls and answerable-question checks remain future work.</p>
    <p style="overflow-wrap:anywhere">Base revision: <code>'''+P['base_revision']+'''</code><br>MOLT revision: <code>'''+P['molt_revision']+'''</code><br>Selected asset SHA256: <code>'''+P['asset_sha256']+'''</code></p></section>
    <section><h2>Passages and scientific record</h2>'''+passages+'''<p><a href="responses.csv">All 18 answers and judgments</a> · <a href="assessment.json">Grading explanations</a> · <a href="summary.json">Summary</a> · <a href="artifacts/added-gate-results.json">Raw output tokens and edit traces</a> · <a href="protocol.json">Protocol</a> · <a href="run.py">Runner</a> · <a href="report.py">Report code</a> · <a href="previous-eight-reference.json">Earlier addition-8 anchor</a> · <a href="../2026-10-15-passage-additive-gates/index.html">Previous 3–8 sweep</a></p>
    <p class="eyebrow">Local report only; GitHub publication deferred. SPAR GPU retained. Selected SQuAD2.0 text: Rajpurkar, Jia, and Liang (2018), CC BY-SA 4.0.</p></section></main>
    <script>document.getElementById('strength-filter').addEventListener('change',function(){document.querySelectorAll('.full [data-strength]').forEach(c=>{c.hidden=this.value!=='all'&&c.dataset.strength!=='clean'&&c.dataset.strength!==this.value;});document.querySelector('.full').style.minWidth=this.value==='all'?'1800px':'800px';});</script></body></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
