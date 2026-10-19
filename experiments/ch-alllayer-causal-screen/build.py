"""Independent edit verification and a compact scientific result report."""
import csv,hashlib,html,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/'results/2026-10-19-ch-alllayer-causal-screen'
DEST.mkdir(exist_ok=True)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def rounded_bf16(a):
    u=np.asarray(a,dtype=np.float32).view(np.uint32)
    return (((u+(0x7fff+((u>>16)&1)))>>16)<<16).view(np.float32)

def main():
    screen=json.loads((ROOT/'screen/results.json').read_text())
    anchors=json.loads((ROOT/'anchors/results.json').read_text())
    confirm=json.loads((ROOT/'confirm/results.json').read_text())
    fidelity=json.loads((ROOT/'fidelity/results.json').read_text())
    changed=json.loads((ROOT/'changed-answers/results.json').read_text())
    assert all(r['status']=='complete' for r in [screen,anchors,confirm,fidelity,changed])
    assets=json.loads((ROOT/'weights/manifest.json').read_text());weights={}
    for a in assets['layers']:
        path=ROOT/'weights'/a['file'];assert sha(path)==a['sha256']
        with np.load(path) as f:weights[a['layer']]={k:f[k] for k in f.files}
    with np.load(ROOT/'capture/fewshot_upper_child.npz') as a:
        native_input=(a['input_bf16'].astype(np.uint32)<<16).view(np.float32)
    audits=[]
    for phase,data in [('anchors',anchors),('screen',screen)]:
        for row in data['rows']:
            c=row['condition'];path=ROOT/phase/row['edit']['file'];assert sha(path)==row['edit']['sha256']
            with np.load(path) as v:
                assert np.array_equal(v['input'],native_input[c['layer'],c['position']])
                assert np.array_equal(rounded_bf16(v['native_mlp']+v['intended_delta']),v['edited_mlp'])
                assert np.array_equal(v['edited_mlp']-v['native_mlp'],v['actual_delta'])
                if c['kind']=='native_mlp':delta=-c['fraction']*v['native_mlp']
                else:
                    w=weights[c['layer']];z=(v['input']-w['mean'])/w['std'];delta=np.zeros(2560,dtype=np.float32)
                    for f,component in zip(c['features'],row['edit']['components']):
                        ix=w['ids'].tolist().index(f);raw=float(z@w['encoder'][ix]+w['bias'][ix])
                        gate=raw if raw>w['threshold'][ix] and raw>0 else 0.
                        assert abs(raw-component['raw'])<5e-5
                        assert (gate>0)==(component['gate']>0)
                        delta-=c['fraction']*gate*((z@w[f'V_{f}']@w[f'U_{f}'])*w['outstd'])
                err=float(np.max(np.abs(delta-v['intended_delta'])))
                assert np.allclose(delta,v['intended_delta'],atol=5e-5,rtol=5e-4),(c['id'],err)
                audits.append({'phase':phase,'condition':c['id'],'CPU_delta_max_abs_error':err})
    singles=[r for r in screen['rows'] if r['condition']['kind']=='single_molt']
    active=[r for r in singles if any(c['gate']>0 for c in r['edit']['components'])]
    repeats=[]
    for row in confirm['rows']:
        c=row['condition']
        if row['case_id']=='fewshot_upper_child' and len(c.get('terms',[]))==1 and not c.get('random_direction') and c.get('fraction')==1:
            old=next(r for r in singles if r['condition']['id']==c['id'])
            assert row['score']['metrics']['target_probability']==old['target_probability'],c['id']
            repeats.append(c['id'])
    assert len(repeats)==len(confirm['selection']['features'])
    verification={'all_phases_complete':True,'all_candidate_edits_CPU_reconstructed':True,
        'all_actual_BF16_edits_verified_exactly':True,'all_source_inputs_match_unintervened_capture':True,
        'all_selected_primary_probabilities_reproduced_exactly':True,'repeated_ids':repeats,'rows':audits}
    (DEST/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    colors={'source':'#3366aa','prediction':'#cc7733'}
    fig,grid=plt.subplots(2,2,figsize=(14,10),sharex=True);axs=grid.ravel()
    for site in ['source','prediction']:
        count=[sum(r['condition']['site']==site and r['condition']['layer']==l for r in active) for l in range(36)]
        axs[0].plot(range(36),count,'o-',color=colors[site],label='Input child' if site=='source' else 'Position predicting CH')
        ar=[r for r in anchors['rows'] if r['condition']['site']==site and r['condition']['fraction']==1]
        axs[1].plot([r['condition']['layer'] for r in ar],[r['drop_log_odds'] for r in ar],'o-',color=colors[site],ms=4)
        sr=[r for r in active if r['condition']['site']==site]
        jitter=-.13 if site=='source' else .13
        ax=axs[2] if site=='source' else axs[3]
        ax.scatter([r['condition']['layer']+jitter for r in sr],[r['drop_log_odds'] for r in sr],s=24,alpha=.65,color=colors[site])
    axs[0].set_ylabel('Active MOLTs');axs[0].legend(loc='upper left',ncol=2)
    fig.suptitle('child → CH: all 36 layers, two causal test positions',fontsize=17,weight='bold',y=.99)
    axs[0].set_title('Natural activity',loc='left',fontsize=12)
    axs[1].set_ylabel('Drop in CH log odds');axs[1].set_title('Zero the whole native MLP output at one position',loc='left',fontsize=11)
    axs[2].set_ylabel('Drop in first-token CH log odds');axs[2].set_title('Single MOLT at input child',loc='left',fontsize=12)
    axs[3].set_ylabel('Drop in first-token CH log odds');axs[3].set_title('Single MOLT at the prediction position',loc='left',fontsize=12)
    for site in ['source','prediction']:
        top=sorted([r for r in active if r['condition']['site']==site],key=lambda x:-x['drop_log_odds'])[:2]
        for i,r in enumerate(top):
            c=r['condition'];x=c['layer']+(-.13 if site=='source' else .13);y=r['drop_log_odds']
            ax=axs[2] if site=='source' else axs[3]
            ax.annotate(f"L{c['layer']}-T{c['transform']}",(x,y),xytext=(-75 if c['layer']>24 else 8,15+i*10),textcoords='offset points',fontsize=9,
                arrowprops={'arrowstyle':'-','alpha':.4},color=colors[site])
    for ax in axs:
        ax.axhline(0,color='#777',lw=.7);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylim(0,18)
    axs[2].margins(y=.18);axs[3].margins(y=.18)
    axs[2].set_xticks(range(0,36,3));axs[2].set_xlabel('Layer (zero-based)');axs[3].set_xlabel('Layer (zero-based)')
    fig.text(.08,.018,'Positive: removal weakens first-token CH. Negative: removal strengthens it. Large prediction-position effects can change format to child: CH.',fontsize=10,color='#555')
    fig.tight_layout(rect=[0,.045,1,.96]);fig.savefig(DEST/'causal-screen.png',dpi=170);fig.savefig(DEST/'causal-screen.svg');plt.close(fig)
    upper=[b for b in confirm['baselines'] if b['case']['id'].startswith('fewshot_upper_')]
    transfer=[]
    for b in upper:
        cid=b['case']['id'];r=next(r for r in confirm['rows'] if r['case_id']==cid and r['condition']['id']=='L28-T1123-source')
        transfer.append({'word':b['case']['target'],'baseline_P_CH':b['score']['metrics']['target_probability'],
            'ablated_P_CH':r['score']['metrics']['target_probability'],'baseline_answer':b['native']['answer'],
            'ablated_answer':r['generation']['answer']})
    x=np.arange(len(transfer));fig,ax=plt.subplots(figsize=(10,5.5))
    before=ax.bar(x-.18,[r['baseline_P_CH']*100 for r in transfer],.36,color='#3366aa',label='Native')
    after=ax.bar(x+.18,[r['ablated_P_CH']*100 for r in transfer],.36,color='#cc7733',label='Remove L28-T1123 at input word')
    for bars in [before,after]:
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.1f}%',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(x,[r['word']+'\n'+r['baseline_answer']+' → '+r['ablated_answer'] for r in transfer]);ax.set_ylim(0,113)
    ax.set_ylabel('Probability of first answer token CH (%)');ax.spines[['top','right']].set_visible(False);ax.yaxis.grid(alpha=.15);ax.set_axisbelow(True)
    ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(.5,1.02),frameon=False)
    ax.set_title('L28-T1123 supports CH: confidence loss on child, wrong letters on chief',fontsize=13,weight='bold',pad=35)
    fig.text(.08,.035,'Exact uppercase few-shot prompts. Labels below words show native → ablated full answers.\ncheef is a two-token nonce word; both input subtokens were edited. Native CE is already wrong.',fontsize=9,color='#555')
    fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(DEST/'L28-T1123-confirmation.png',dpi=170);fig.savefig(DEST/'L28-T1123-confirmation.svg');plt.close(fig)
    (DEST/'L28-T1123-transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
    ranked={site:sorted([r for r in active if r['condition']['site']==site],key=lambda r:-r['drop_log_odds']) for site in ['source','prediction']}
    by_changed={r['condition']['id']:r for r in changed['rows']}
    assert set(by_changed)=={r['condition']['id'] for r in singles if r['greedy_token_id']!=2149}
    recon={s:[r['edit']['reconstruction_relative_L2_error'] for r in screen['rows']
        if r['condition']['kind']=='all_active_molts' and r['condition']['site']==s] for s in ['source','prediction']}
    summary={'baseline':screen['baseline'],'active_feature_site_pairs':len(active),
        'active_unique_transforms':len({(r['condition']['layer'],r['condition']['transform']) for r in active}),
        'single_feature_tests':len(singles),'threshold_check_tests':len(singles)-len(active),
        'active_by_site':{s:len(v) for s,v in ranked.items()},
        'single_ablation_first_token_changes':sum(r['greedy_token_id']!=2149 for r in active),
        'top_by_site':{s:v[:8] for s,v in ranked.items()},'confirmation_case_count':len(confirm['baselines']),
        'confirmation_rows':len(confirm['rows']),'changed_first_token_full_answers':by_changed,
        'reconstruction_median_relative_L2_error':{s:float(np.median(v)) for s,v in recon.items()},
        'interpretation':'L28-T1123 source ablation lowers CH and raises CI while retaining greedy CH. Many large prediction-position effects change output format to child: CH. Estimated contribution subtraction in native Qwen is not proof of CH specificity or absorption. Other positions and attention were not causally screened.'}
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (DEST/'single-ablation-results.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(['layer','transform','site','cached_label','gate','actual_delta_norm','P_CH','drop_log_odds','greedy_token'])
        for r in singles:
            c=r['condition'];wr.writerow([c['layer'],c['transform'],c['site'],c['cached_label'],r['edit']['components'][0]['gate'],r['edit']['actual_bf16_delta_norm'],r['target_probability'],r['drop_log_odds'],r['greedy_token']])
    body=''
    for site,title in [('source','At input child'),('prediction','At the position predicting CH')]:
        body+=f'<h2>{title}</h2><table><tr><th>MOLT</th><th>Cached gate label</th><th>P(first CH) after removal</th><th>Drop in log odds</th><th>Next token</th><th>Full answer, if replayed</th></tr>'
        for r in ranked[site][:8]:
            c=r['condition'];full=by_changed.get(c['id'],{}).get('answer')
            if full is None:
                matched=[x for x in confirm['rows'] if x['case_id']=='fewshot_upper_child' and x['condition']['id']==c['id']]
                full=matched[0]['generation']['answer'] if matched else 'Not replayed'
            body+=f'<tr><td>L{c["layer"]}-T{c["transform"]}</td><td>{html.escape(str(c["cached_label"]))}</td><td>{r["target_probability"]:.6%}</td><td>{r["drop_log_odds"]:.3f}</td><td>{html.escape(r["greedy_token"])}</td><td>{html.escape(full)}</td></tr>'
        body+='</table>'
    body+='<h2>Confirmation: generated outputs</h2>'
    for base in confirm['baselines']:
        cid=base['case']['id'];op=' open' if cid=='fewshot_upper_child' else ''
        body+=f'<details{op}><summary>{html.escape(cid)} · baseline: {html.escape(base["native"]["answer"])}</summary><table><tr><th>Intervention</th><th>Answer</th><th>Target probability</th></tr>'
        for r in confirm['rows']:
            if r['case_id']!=cid:continue
            body+=f'<tr><td>{html.escape(r["condition"]["id"])}</td><td>{html.escape(r["generation"]["answer"])}</td><td>{r["score"]["metrics"]["target_probability"]:.6%}</td></tr>'
        body+='</table></details>'
    body+='<h2>Single-site reconstruction controls</h2><p>Replace only the tested site with the complete MOLT estimate, including output mean. Compare that reconstruction baseline with the same reconstruction after removing the selected component. These checks use an approximate single-site surrogate; they are separate from the native-model subtractions above.</p><table><tr><th>Feature</th><th>Relative output error</th><th>Reconstruction P(first CH)</th><th>After removal</th><th>Full answers, baseline → removal</th></tr>'
    for f in confirm['selection']['features']:
        pair=[r for r in fidelity['rows'] if r['feature']['id']==f['id']];base=next(r for r in pair if not r['remove_feature']);ab=next(r for r in pair if r['remove_feature'])
        body+=f'<tr><td>{html.escape(f["id"])}</td><td>{base["source_MOLT_relative_L2_error"]:.3f}</td><td>{base["P_CH"]:.6%}</td><td>{ab["P_CH"]:.6%}</td><td>{html.escape(base["answer"])} → {html.escape(ab["answer"])}</td></tr>'
    body+='</table>'
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Which MOLTs support child → CH?</title>
    <style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#1b2433}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{border-bottom:1px solid #ddd;text-align:left;padding:9px;vertical-align:top}}th{{background:#eef2f6}}code,pre{{background:#f2f4f7;padding:3px 5px}}.note{{background:#edf4ff;padding:16px;border-left:4px solid #3366aa}}</style></head><body>
    <h1>Which MOLTs support child → CH?</h1><p>19 October 2026 · Qwen3-4B · Native model with estimated MOLT contribution subtraction</p>
    <p class="note">Baseline P(CH): {screen['baseline']['target_probability']:.6%}. Tested {len(active)} active feature-position pairs across all 36 layers at the source word and final prompt position, plus one near-threshold check. {summary['single_ablation_first_token_changes']} individual active-component ablations changed the first token. Activity alone is not evidence of causal support.</p>
    <p><strong>Main result:</strong> L28-T1123 is the strongest source-word candidate: native ablation reduces first-token CH from 99.32% to 81.68% and raises CI from 0.67% to 18.23%, while the greedy answer stays CH. Its cached label is “Ch-prefix food and noun tokens.” Large prediction-position effects often change CH into the longer answer child: CH; a changed first token is not automatically a wrong-letter answer. Every changed-first-token condition was continued to a complete answer.</p>
    <p>The same source-word ablation changes chief from CH to CI, with P(CH) falling from 96.03% to 8.43%. For child, a matched-norm random direction leaves P(CH) at 99.46%. Two new uppercase child templates also lose CH confidence while retaining the right answer. The feature is inactive on both cider and table controls; those outputs are unchanged. These comparisons support a candidate role in CH letter choice, while child remains robust to the tested single-feature removal.</p>
    <img src="L28-T1123-confirmation.png" alt="Native and L28-T1123-ablated CH probabilities and full answers for five uppercase words">
    <img src="causal-screen.png" alt="All-layer activity and ablation effects at source and prediction positions">
    <h2>Exact primary prompt</h2><pre>{html.escape(json.loads((ROOT/'cases.json').read_text())[0]['user'])}</pre>
    <p>Source word: token 27, <code>child</code>. Prediction position: token 37, the final newline after the closed thinking block. The first generated token is <code>CH</code> (2149). The model's main native alternative is <code>CI</code>. Probabilities are measured with cached decoding, one prompt at a time. Log odds compare CH against the sum of all other next-token probabilities.</p>
    <h2>Scope and interpretation</h2><p>All 2,480 encoders at all 36 layers were surveyed on all processed native tokens of ten earlier uppercase/lowercase prompts. Causal screening covered only the source word and position predicting CH in the original uppercase child task. Whole-native-MLP site controls and individual/group MOLT subtractions use normal downstream attention. Selected single effects were repeated and tested on 16 prompts total, including six new capitalization/template/non-CH controls. Random controls use one direction with matched intended norm per selected primary effect.</p>
    <p>A MOLT contributes gate × ((standardized input × V) × U) × output standard deviation. The intervention subtracts that estimate from the native MLP output, retaining the native remainder. Across layers, median relative reconstruction error is {np.median(recon['source']):.3f} at the source and {np.median(recon['prediction']):.3f} at the prediction site. These substantial errors limit mechanistic interpretation. Some single-site reconstruction baselines already change response format. These are approximate decomposition interventions, not a completely MOLT-replaced model. A positive effect does not by itself establish CH selectivity, necessity across prompts, or feature absorption. Effects at other token positions, attention heads, and residual information remain open. Lowercase source interventions affect the input word only; teacher-forced echoed child tokens can provide another route to the letters.</p>
    {body}<h2>Provenance and verification</h2><p>Native generations reproduce all ten earlier outputs, including the known uppercase cheef → CE mistake. Read-only captures leave every prediction score unchanged. Full base-file hashes and released tensor slice hashes were verified. Every screen edit was independently reconstructed on CPU; BF16 rounding and actual applied edits were verified exactly. All selected primary probabilities reproduced exactly in a new model load.</p>
    <p><a href="summary.json">Summary JSON</a> · <a href="verification.json">Independent verification</a> · <a href="single-ablation-results.csv">All individual ablations</a> · <a href="screen-results.json">Screen details</a> · <a href="anchors-results.json">Native MLP controls</a> · <a href="confirmation-results.json">Full confirmation details</a> · <a href="protocol.json">Protocol</a></p>
    <p>Checkpoints: Qwen/Qwen3-4B @ {PVAL('base_revision')} and georglange/qwen3-4b-molt @ {PVAL('molt_revision')}. Gate labels come from dated authenticated-dashboard caches; labels were not used to select candidates. The absorption paper motivates measuring graded output effects and testing causality: <a href="https://arxiv.org/html/2409.14507v6">Feature Absorption in Sparse Autoencoders</a>.</p></body></html>'''
    (DEST/'index.html').write_text(text)
    for source,name in [('screen/results.json','screen-results.json'),('anchors/results.json','anchors-results.json'),('confirm/results.json','confirmation-results.json'),('fidelity/results.json','reconstruction-controls.json'),('changed-answers/results.json','changed-answers.json'),('protocol.json','protocol.json')]:shutil.copy2(ROOT/source,DEST/name)
    manifest={p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['top_by_site','changed_first_token_full_answers']},indent=2))
    for s in ranked:
        print(s,[(r['condition']['id'],r['target_probability'],r['drop_log_odds']) for r in ranked[s][:4]])

def PVAL(key):return json.loads((ROOT/'protocol.json').read_text())[key]

if __name__=='__main__':main()
