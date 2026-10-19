"""Independently reconstruct held-out predictions, audit splits, and render a report."""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'remote-export'
ART=RUN/'artifacts'
FIT=ROOT/'trained-probe'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


for item in json.loads((RUN/'download-manifest.json').read_text()):
    path=RUN/item['file'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
for item in json.loads((ROOT/'extraction-source-manifest-v2.json').read_text()):
    assert sha(RUN/item['file'])==item['sha256']
D=json.loads((ART/'dataset-summary.json').read_text())
E=json.loads((ART/'extraction.json').read_text())
M=json.loads((FIT/'metrics.json').read_text())
P=json.loads((RUN/'protocol.json').read_text())
retry=json.loads((ROOT/'numerical-retry.json').read_text())
subgroups=json.loads((ROOT/'subgroup-validation.json').read_text())
cross_checks=json.loads((ROOT/'cross-experiment-activation-check.json').read_text())
assert all(c['equals_prior_ablation_native_mlp_exactly'] for c in cross_checks)
assert sha(ART/'dataset.json')==retry['frozen_dataset_sha256']==E['dataset_sha256']==M['dataset_sha256']
assert E['status']==M['status']=='complete'
assert P['extraction_batch_size']==1 and all(x['early_stop_equals_full_native_exactly'] and x['batch_vs_single_max_abs_error']==0 for x in E['native_full_forward_checks'])
assert E['native_attention_unmodified'] and E['no_interventions_or_generation']
assert M['no_alignment_or_ablation_run']
rows=json.loads((ART/'dataset.json').read_text())
with np.load(ART/'activations.npz') as arrays:
    assert arrays['row_ids'].tolist()==[r['id'] for r in rows]
    X=(arrays['mlp_bf16'].astype(np.uint32)<<16).view(np.float32)
with np.load(FIT/'probe.npz') as arrays:
    w=arrays['weight_raw'];b=float(arrays['intercept_raw'][0]);threshold=float(arrays['threshold'][0])
    scaled_w=arrays['coefficient_scaled'];scaled_b=float(arrays['intercept_scaled'][0])
    mean=arrays['training_mean'];scale=float(arrays['global_rms_scale'][0])
assert sha(FIT/'probe.npz')==M['selection']['probe_sha256']
assert sha(RUN/'train.py')==M['selection']['train_source_sha256']
assert sha(RUN/'protocol.json')==M['selection']['protocol_sha256']
score=X.astype(np.float64)@w+b
assert np.allclose(score,((X.astype(np.float64)-mean)/scale)@scaled_w+scaled_b,atol=1e-9,rtol=1e-9)
prob=expit(score);pred=prob>=threshold
with (FIT/'predictions.csv').open() as stream:predictions=list(csv.DictReader(stream))
assert len(predictions)==len(rows)
for i,(r,p) in enumerate(zip(rows,predictions)):
    assert r['id']==p['id'] and int(p['label'])==r['label']==int(r['word'].startswith('ch'))
    assert r['prompt_token_ids'][r['target_position']]==r['token_id']
    assert abs(float(p['probe_score'])-score[i])<1e-9 and abs(float(p['probability'])-prob[i])<1e-12
    assert int(p['prediction'])==int(pred[i])
    if any(r['word'].startswith(prefix) for prefix in P['diagnostic_positive_prefixes']):assert r['split']=='diagnostic'
sets={s:{r['family'] for r in rows if r['split']==s} for s in ['train','validation','test','diagnostic']}
for a,aa in sets.items():
    for bb,ss in sets.items():
        if a!=bb:assert not aa&ss
test=np.array([i for i,r in enumerate(rows) if r['split']=='test' and not r['is_transfer']])
y=np.array([rows[i]['label'] for i in test]);ts=score[test];tpred=pred[test]
counts=Counter(rows[i]['family'] for i in test)
weights=np.array([1/counts[rows[i]['family']] for i in test])
yp=y==1;yn=~yp
pairwise=(ts[yp,None]>ts[None,yn]).astype(float)+.5*(ts[yp,None]==ts[None,yn])
auc=float(np.sum(pairwise*weights[yp,None]*weights[None,yn])/(weights[yp].sum()*weights[yn].sum()))
tpr=float(np.sum(weights[yp]*tpred[yp])/weights[yp].sum())
tnr=float(np.sum(weights[yn]*(~tpred[yn]))/weights[yn].sum())
bacc=(tpr+tnr)/2
positive=np.sum(weights[(y==1)&tpred]);fp=np.sum(weights[(y==0)&tpred]);fn=np.sum(weights[(y==1)&(~tpred)])
f1=float(2*positive/(2*positive+fp+fn))
ref=M['main']['test']['family_weighted']
assert abs(auc-ref['auroc'])<1e-12 and abs(bacc-ref['balanced_accuracy'])<1e-12 and abs(f1-ref['f1'])<1e-12
proof={'status':'passed','source_download_and_dataset_hashes_verified':True,
    'dataset_identical_before_and_after_numerical_retry':True,'stem_families_disjoint_across_all_splits':True,
    'reserved_prefixes_absent_from_train_validation_test':True,'labels_recomputed_from_spelling':True,
    'all_saved_predictions_recomputed_from_portable_raw_coordinate_weights':True,
    'preprocessing_coordinate_conversion_verified':True,'native_full_forward_checks_exact':E['native_full_forward_checks'],
    'test_auroc_independent_pairwise_calculation':auc,'test_balanced_accuracy_independent_calculation':bacc,
    'test_f1_independent_calculation':f1,'probe_sha256':sha(FIT/'probe.npz'),
    'prior_ablation_native_activations_match_exactly':cross_checks,
    'scope':'Dataset, native activation collection, probe training and validation only.'}
save('verification.json',proof)


def esc(value):return html.escape(str(value))
def pct(value):return f'{100*value:.2f}%'
def table(headers,values):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in values)+'</tbody></table></div>'


split_table=table(['Split','Prompt examples','Distinct words','Stem groups','CH-positive groups'],
    [[s,D['main'][s]['rows'],D['main'][s]['words'],D['main'][s]['families'],D['main'][s]['positive_families']] for s in ['train','validation','test','diagnostic']])
metrics_table=table(['Split','AUROC','Balanced accuracy','Precision','Recall','F1'],
    [[s,f"{M['main'][s]['family_weighted']['auroc']:.4f}",*[pct(M['main'][s]['family_weighted'][k]) for k in ['balanced_accuracy','precision','recall','f1']]] for s in ['train','validation','test']])
ci_table=table(['Held-out metric','Estimate','95% interval, resampling stem groups'],
    [[k,f"{ref[k]:.4f}",f"{interval[0]:.4f} – {interval[1]:.4f}"] for k,interval in M['test_family_bootstrap_95ci'].items()])
group_table=table(['Negative group','AUROC vs CH positives','Negative false-positive rate'],
    [[cat,f"{m['auroc']:.4f}",pct(m['negative_false_positive_rate'])] for cat,m in M['hard_negative_groups'].items()])
template_table=table(['Held-out prompt template','AUROC','Balanced accuracy'],
    [[name,f"{m['auroc']:.4f}",pct(m['balanced_accuracy'])] for name,m in M['per_template'].items()])
selected=[]
for word in ['child','chin','chair','chief','cheef','infant','baby','kid','parent','adult']:
    for template in ['fewshot_upper','fewshot_lower']:
        found=[p for p in M['diagnostic_predictions'] if p['word']==word and p['surface']==word and p['template']==template]
        for p in found:selected.append([word,template,p['label'],f"{p['probability']:.5f}",p['prediction']])
diagnostic_table=table(['Reserved word','Prompt','True CH label','Probe output probability','Prediction'],selected)
shuffle_table=table(['Family-label shuffle','AUROC on true test labels','Balanced accuracy'],
    [[r['iteration']+1,f"{r['metrics_against_true_test_labels']['auroc']:.4f}",pct(r['metrics_against_true_test_labels']['balanced_accuracy'])] for r in M['shuffled_family_label_controls']])
subgroup_table=table(['Post-fit test subset','Rows','CH-positive stem groups','AUROC','Balanced accuracy'],
    [[name,g['rows'],g['positive_families'],f"{g['auroc']:.4f}" if g['auroc'] is not None else 'not applicable',pct(g['balanced_accuracy']) if g['balanced_accuracy'] is not None else 'not applicable'] for name,g in subgroups['groups'].items()])
errors=json.loads((FIT/'test-errors.json').read_text())
error_words=sorted({r['word'] for r in errors})
error_table=table(['Word','Family','Label','Surface','Template','Probe probability'],
    [[r['word'],r['family'],r['label'],repr(r['surface']),r['template'],f"{r['probability']:.5f}"] for r in errors[:50]])
source=json.loads((RUN/'sources/source.json').read_text())
headline=f"The probe learns a CH signal overall (test AUROC {auc:.4f}), but fails on the reserved child cases. It is not validated for the intended Child-MOLT analysis."
child_diagnostic=[p for p in M['diagnostic_predictions'] if p['word']=='child' and p['surface']=='child' and p['template'] in P['templates_train']]
assert len(child_diagnostic)==2 and all(p['prediction']==0 for p in child_diagnostic)
critical='The exact unindented child token receives probe output '+', '.join(p['template']+': '+f"{p['probability']:.5f}" for p in child_diagnostic)+f", both below the frozen threshold {M['selection']['selected_threshold']:.5f}. It also misses chin. These are diagnostic failures, not evidence that CH information is absent from the model or from a MOLT. The probe has not been retrained to correct these selected cases."
limitations='''<p>The test set contains only 16 CH-positive stem groups, so the intervals matter. Multiple capitalization, whitespace and prompt variants are correlated; the primary metrics and bootstrap account for this by grouping them. Snowball stemming is a reproducible approximation to word families, not a guarantee of semantic or irregular morphological independence. The single-token dictionary restriction limits generalization to multi-token words and arbitrary spellings.</p>
<p>The diagnostic set contains words selected during our earlier investigation; it is kept separate from the randomly held-out test. The probe is a linear readout trained with class balancing, so its probabilities are not calibrated for natural text prevalence. Strong classification does not establish that the model uses this direction or that a MOLT has absorbed it. No MOLT alignment, component ablation or later-layer tracking was performed here.</p>'''
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CH probe: dataset and held-out validation</title>
<style>body{max-width:1120px;margin:32px auto;padding:0 20px;font:16px/1.55 system-ui,sans-serif;color:#172f3b;background:#fbfcfd}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4dfe6;text-align:left;padding:8px;white-space:pre-wrap}th{background:#eaf1f5}pre{white-space:pre-wrap;background:#eef3f6;padding:12px}.tablewrap{overflow-x:auto;margin:12px 0}img{max-width:100%;height:auto}details{margin:14px 0;padding:12px;background:white;border:1px solid #d4dfe6}summary{cursor:pointer;font-weight:650}</style>
<h1>CH probe: dataset and held-out validation</h1><p>Qwen3-4B · layer 3 native MLP output · input word token · 19 October 2026</p><p><strong>'''+esc(headline)+'''</strong></p>
<p>'''+esc(critical)+'''</p>
<p>A linear logistic-regression readout predicts whether a complete single-token dictionary word starts with CH, ignoring capitalization. It receives the 2,560-dimensional native layer-3 MLP output at that word. Neither MOLT activations nor the target answer are probe inputs. The model is unmodified.</p>
<h2>Frozen dataset</h2>'''+split_table+'''<p>Each main example uses an uppercase or lowercase few-shot template. Case and leading-space variants remain in their stem group. An additional '''+str(D['transfer_rows'])+''' examples evaluate a different prompt template on held-out test and diagnostic words. Negatives cover C without H, H-initial words, internal CH, and other words; representative word lengths are matched at group selection.</p>
<p>Source: '''+esc(source['dictionary'])+''', Debian package '''+esc(source['package_version'])+'''. The original dictionary, its license notice, eligibility rules, source hash and complete split membership are preserved. The prefixes child/chin/chair/chief/cheef and the reserved semantic controls never enter training or validation.</p>
<h2>Training and model selection</h2><p>L2-regularized logistic regression; equal total class weight and equal family weight within a class. Centering and one global RMS scale are fitted on training data only. No per-coordinate scaling changes the meaning of Euclidean angles. Seven regularization values are compared on validation AUROC; ties prefer stronger regularization. The classification threshold maximizes validation balanced accuracy, with ties preferring proximity to 0.5.</p>
<p>Selected C: '''+esc(M['selection']['selected_C'])+''' · frozen threshold: '''+esc(M['selection']['selected_threshold'])+'''. The selected probe and threshold were saved before test scoring. Portable coefficients are saved in native raw-MLP coordinates.</p>
<h2>Held-out results</h2>'''+metrics_table+'''<img src="trained-probe/probe-validation.png" alt="Validation and held-out ROC and precision-recall curves">'''+ci_table+'''
<h2>Hard negatives and prompt transfer</h2>'''+group_table+template_table+'''
<h2>Capitalization and whitespace checks</h2><p>These subgroup checks were added after fitting and did not change the model or threshold. The no-leading-space subset, most relevant to the original child task, has only five CH-positive test stem groups. The all-uppercase subset has no CH-positive test groups and cannot establish uppercase generalization.</p>'''+subgroup_table+'''
<h2>Reserved diagnostics</h2><p>These probe predictions are separate from the randomly held-out test metrics. A probe output is a classification readout, not the model's probability of answering CH.</p>'''+diagnostic_table+'''
<p>The nonce spelling cheef was reserved but has no eligible single-token form in these prompts, so it is excluded from this single-token probe experiment.</p>
<h2>Shuffled-label controls</h2><p>Five training-label permutations at the whole-family level, using the already selected regularization. Scores below evaluate the true held-out labels. Chance AUROC is 0.5; controls are all reported rather than selecting a favorable permutation.</p>'''+shuffle_table+'''
<h2>Errors and limits</h2><p>Held-out main-set errors: '''+str(len(errors))+''' prompt examples involving '''+str(len(error_words))+''' distinct words: '''+esc(', '.join(error_words))+'''.</p>'''+limitations+'''<details><summary>First 50 test errors</summary>'''+error_table+'''</details>
<h2>Activation and result verification</h2><p>The initial batch-32 extraction failed its predeclared single-prompt native-reference check (1.56% relative difference on one audit vector) and was rejected before fitting. The accepted extraction uses batch size one; early-stop collection exactly matches complete native forwards on all audit examples. The frozen dataset hash stayed identical. Saved predictions, preprocessing conversion, and held-out AUROC/balanced-accuracy/F1 were independently recomputed from the portable weights and exact BF16 activation bits.</p>
<p><a href="verification.json">Verification</a> · <a href="trained-probe/metrics.json">Complete metrics</a> · <a href="trained-probe/probe.npz">Portable probe</a> · <a href="trained-probe/train.csv">Training split</a> · <a href="trained-probe/validation.csv">Validation split</a> · <a href="trained-probe/test.csv">Test split</a> · <a href="trained-probe/word-splits.json">Word lists</a> · <a href="trained-probe/predictions.csv">All predictions</a> · <a href="remote-export/artifacts/dataset.json">Exact prompts and token positions</a></p></html>'''
(ROOT/'index.html').write_text(page)
save('summary.json',{'test':ref,'test_family_bootstrap_95ci':M['test_family_bootstrap_95ci'],
    'splits':D['main'],'selected_C':M['selection']['selected_C'],'threshold':threshold,
    'diagnostic_unindented_lowercase_words':[p for p in M['diagnostic_predictions'] if p['surface']==p['word'] and p['template'] in P['templates_train']],
    'verification':'passed','intended_child_analysis_validated':False,
    'key_diagnostic_failure':'Frozen probe misclassifies exact input child in both original prompt types and all examined capitalization/leading-space variants; do not interpret its angle as evidence against absorbed CH information.',
    'no_retraining_after_diagnostic_inspection':True,
    'limitations':'Only 16 CH-positive test stem groups; stemming heuristic; linear readout is not causal evidence.'})
print(json.dumps({'verification':'passed','test_auroc':auc,'test_balanced_accuracy':bacc,'test_f1':f1,'test_error_words':error_words},indent=2))
