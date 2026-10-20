"""Build the narrative overview from frozen measurements; no inference or weight loading."""
import argparse,hashlib,html,json,shutil
from pathlib import Path
from plot_input_gates import build_input_gate_figure

HERE=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE.parents[1])
    a=ap.parse_args();root=a.root;out=root/'results/2026-10-19-ch-absorption-overview';out.mkdir(parents=True,exist_ok=True)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    def load(name):return json.loads((root/name).read_text())
    cross=load('results/2026-10-19-ch-crosslayer-activation/summary.json')
    probe=load('results/2026-10-19-ch-prefix-probe/summary.json')
    original=load('results/2026-10-19-ch-child-all-occurrences-ablation/summary.json')
    joint=load('results/2026-10-19-ch-child-joint-ablation/summary.json')
    screen=load('results/2026-10-19-ch-alllayer-causal-screen/summary.json')
    transfer=load('results/2026-10-19-ch-alllayer-causal-screen/L28-T1123-transfer.json')
    source_inputs=['results/2026-10-19-ch-crosslayer-activation/summary.json','results/2026-10-19-ch-prefix-probe/summary.json',
        'results/2026-10-19-ch-child-all-occurrences-ablation/summary.json','results/2026-10-19-ch-child-joint-ablation/summary.json',
        'results/2026-10-19-ch-alllayer-causal-screen/summary.json','results/2026-10-19-ch-alllayer-causal-screen/L28-T1123-transfer.json',
        'results/2026-10-19-ch-crosslayer-activation/input-word-gates.csv',
        'results/2026-10-19-ch-absorption-overview/L28-T1123-input-gates.json']
    gates=build_input_gate_figure(root,out)
    facts={'date':'2026-10-19','gates':gates,'probe':{'test':probe['test'],'splits':probe['splits'],
        'threshold':probe['threshold'],'C':probe['selected_C'],'child_diagnostics':[x for x in probe['diagnostic_unindented_lowercase_words'] if x['word']=='child']},
        'child_single':original,'joint':joint,'screen':{'active_feature_site_pairs':screen['active_feature_site_pairs'],
            'active_unique_transforms':screen['active_unique_transforms'],'active_by_site':screen['active_by_site'],
            'first_token_changes':screen['single_ablation_first_token_changes'],
            'full_changed_answers':{k:v['answer'] for k,v in screen['changed_first_token_full_answers'].items()},
            'reconstruction_median_relative_L2_error':screen['reconstruction_median_relative_L2_error']},'L28_transfer':transfer}
    assert facts['gates']['L3-T2334']['child_gates']=={'uppercase':0.,'lowercase':0.}
    assert not probe['intended_child_analysis_validated']
    assert screen['active_feature_site_pairs']==410 and screen['active_unique_transforms']==369
    assert all('CH' in v for v in facts['screen']['full_changed_answers'].values())
    facts_json=json.dumps(facts,ensure_ascii=False,indent=2)
    (out/'facts.json').write_text(facts_json+'\n')
    copies=[('results/2026-10-19-ch-prefix-probe/trained-probe/probe-validation.png','probe-validation.png'),
            ('results/2026-10-19-ch-alllayer-causal-screen/causal-screen.png','causal-screen.png'),
            ('results/2026-10-19-ch-alllayer-causal-screen/L28-T1123-confirmation.png','L28-T1123-confirmation.png')]
    for src,dst in copies:shutil.copy2(root/src,assets/dst)
    for name in ['site.css','site.js']:shutil.copy2(HERE/name,out/name)
    taskrows={}
    condition_names={'native':'No ablation','L3_only':'L3–T1372 only','L28_only':'L28–T1123 only','joint':'Both MOLTs'}
    for mode,cid in [('upper','fewshot_upper_child'),('lower','fewshot_lower_child')]:
        rows=[r for r in joint['rows'] if r['case_id']==cid and r['condition']!='zero_control']
        taskrows[mode]=[{**r,'label':condition_names[r['condition']]} for r in rows]
    facts['taskrows']=taskrows
    (out/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
    upper=''.join(f'<tr class="{ "joint-row" if r["condition"]=="joint" else ""}"><td>{r["label"]}</td><td><code>{html.escape(r["answer"])}</code></td><td>{r["target_probability"]*100:.2f}%</td></tr>' for r in taskrows['upper'])
    child_upper=next(x for x in probe['diagnostic_unindented_lowercase_words'] if x['word']=='child' and x['template']=='fewshot_upper')
    child_lower=next(x for x in probe['diagnostic_unindented_lowercase_words'] if x['word']=='child' and x['template']=='fewshot_lower')
    replacements={'{{UPPER_GATE}}':f"{cross['features']['L3-T1372']['child_gates']['uppercase']:.2f}",
        '{{INPUT_GATE_FIGURE_VERSION}}':sha(assets/'input-word-gates.png')[:12],
        '{{LOWER_GATE}}':f"{cross['features']['L3-T1372']['child_gates']['lowercase']:.2f}",
        '{{L28_UPPER_GATE}}':f"{gates['L28-T1123']['child_gates']['uppercase']:.2f}",
        '{{L28_LOWER_GATE}}':f"{gates['L28-T1123']['child_gates']['lowercase']:.2f}",
        '{{PROBE_AUROC}}':f"{probe['test']['auroc']:.3f}",'{{PROBE_BALACC}}':f"{probe['test']['balanced_accuracy']*100:.2f}%",
        '{{PROBE_CHILD_UPPER}}':f"{child_upper['probability']*100:.2f}%",'{{PROBE_CHILD_LOWER}}':f"{child_lower['probability']*100:.2f}%",
        '{{PROBE_THRESHOLD}}':f"{probe['threshold']*100:.2f}%",'{{JOINT_UPPER_ROWS}}':upper,
        '{{DATA_JSON}}':json.dumps(facts,ensure_ascii=False).replace('</','<\\/'),
        '{{SOURCE_HASHES}}':html.escape(json.dumps({s:sha(root/s) for s in source_inputs},indent=2))}
    template=(HERE/'template.html').read_text()
    for key,value in replacements.items():template=template.replace(key,value)
    assert '{{' not in template
    (out/'index.html').write_text(template)
    provenance={'inputs':{s:sha(root/s) for s in source_inputs},
        'assets':{p.name:sha(p) for p in assets.iterdir() if p.is_file()},
        'source':'experiments/ch-absorption-overview/build.py','build_uses_saved_results_only':True,
        'paper':'https://arxiv.org/html/2409.14507v6','selected_figure_sha256':sha(assets/'input-word-gates.png')}
    (out/'overview-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
    print('Overview built from frozen measurements:',out)

if __name__=='__main__':main()
