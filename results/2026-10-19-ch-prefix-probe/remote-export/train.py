"""Fit a frozen CH readout; choose settings on validation before held-out scoring."""
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,
    f1_score,precision_score,recall_score,roc_auc_score,roc_curve,precision_recall_curve)
from threadpoolctl import threadpool_limits

START=time.monotonic()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def family_weights(rows,balance_classes=False,labels=None):
    family_counts=Counter(r['family'] for r in rows)
    family_label={r['family']:int(labels[i] if labels is not None else r['label']) for i,r in enumerate(rows)}
    class_families=Counter(family_label.values())
    w=np.array([1/family_counts[r['family']] for r in rows],dtype=np.float64)
    if balance_classes:
        w/=np.array([class_families[family_label[r['family']]] for r in rows])
    return w*len(rows)/w.sum()


def metrics(labels,prob,threshold,weights):
    pred=prob>=threshold
    two_classes=len(np.unique(labels))==2
    return {'auroc':float(roc_auc_score(labels,prob,sample_weight=weights)) if two_classes else None,
        'average_precision':float(average_precision_score(labels,prob,sample_weight=weights)) if two_classes else None,
        'balanced_accuracy':float(balanced_accuracy_score(labels,pred,sample_weight=weights)) if two_classes else None,
        'accuracy':float(accuracy_score(labels,pred,sample_weight=weights)),
        'f1':float(f1_score(labels,pred,sample_weight=weights,zero_division=0)),
        'precision':float(precision_score(labels,pred,sample_weight=weights,zero_division=0)),
        'recall':float(recall_score(labels,pred,sample_weight=weights,zero_division=0)),
        'positive_prediction_rate':float(np.average(pred,weights=weights)),
        'positive_prevalence':float(np.average(labels,weights=weights)),
        'rows':len(labels)}


def fit_probe(X,y,C,weights):
    probe=LogisticRegression(C=C,solver='lbfgs',max_iter=3000,tol=1e-7,fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        probe.fit(X,y,sample_weight=weights)
    return probe


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--result-dir',type=Path,required=True);args=ap.parse_args()
    src=Path(__file__).resolve().parent;out=args.result_dir;out.mkdir(exist_ok=True)
    assert not (out/'metrics.json').exists(),'Preserve existing results before rerunning.'
    P=json.loads((src/'protocol.json').read_text())
    rows=json.loads((args.data_dir/'dataset.json').read_text())
    extract=json.loads((args.data_dir/'extraction.json').read_text())
    assert extract['status']=='complete' and extract['dataset_sha256']==sha(args.data_dir/'dataset.json')
    assert extract['activations_sha256']==sha(args.data_dir/'activations.npz')
    with np.load(args.data_dir/'activations.npz') as a:
        assert a['row_ids'].tolist()==[r['id'] for r in rows]
        X=(a['mlp_bf16'].astype(np.uint32)<<16).view(np.float32)
    assert X.shape==(len(rows),2560) and np.isfinite(X).all()
    y=np.array([r['label'] for r in rows],dtype=np.int64)
    assert all(r['label']==int(r['word'].startswith('ch')) for r in rows)
    main_rows=np.array([not r['is_transfer'] for r in rows])
    idx={s:np.array([i for i,r in enumerate(rows) if r['split']==s and not r['is_transfer']],dtype=int)
         for s in ['train','validation','test','diagnostic']}
    family_sets={s:{rows[i]['family'] for i in ii} for s,ii in idx.items()}
    for s,ss in family_sets.items():
        for t,tt in family_sets.items():
            if s!=t:assert not ss&tt
    train_rows=[rows[i] for i in idx['train']]
    weights=family_weights(train_rows,True)
    mean=np.average(X[idx['train']].astype(np.float64),axis=0,weights=weights)
    centered=X[idx['train']].astype(np.float64)-mean
    scale=float(np.sqrt(np.average(np.mean(centered**2,axis=1),weights=weights)))
    assert scale>0
    XT=centered/scale
    XV=(X[idx['validation']].astype(np.float64)-mean)/scale
    val_rows=[rows[i] for i in idx['validation']];vw=family_weights(val_rows)
    selection=[];models=[]
    with threadpool_limits(limits=4):
        for C in P['regularization_C']:
            model=fit_probe(XT,y[idx['train']],C,weights)
            prob=model.predict_proba(XV)[:,1]
            auc=float(roc_auc_score(y[idx['validation']],prob,sample_weight=vw))
            selection.append({'C':C,'validation_family_weighted_auroc':auc,'iterations':int(model.n_iter_[0])})
            models.append(model)
            print('VALIDATION '+json.dumps(selection[-1]),flush=True)
    best_auc=max(r['validation_family_weighted_auroc'] for r in selection)
    best=next(i for i,r in enumerate(selection) if r['validation_family_weighted_auroc']>=best_auc-1e-8)
    model=models[best];C=selection[best]['C']
    vp=model.predict_proba(XV)[:,1]
    unique=np.unique(vp)
    thresholds=np.unique(np.concatenate(([0.,.5,1.],unique,(unique[:-1]+unique[1:])/2)))
    candidates=[]
    for t in thresholds:
        bacc=float(balanced_accuracy_score(y[idx['validation']],vp>=t,sample_weight=vw))
        candidates.append((bacc,abs(float(t)-.5),float(t)))
    _,_,threshold=min(candidates,key=lambda a:(-round(a[0],12),a[1],a[2]))
    w=model.coef_[0]/scale;b=float(model.intercept_[0]-np.dot(w,mean))
    assert np.allclose(XV@model.coef_[0]+model.intercept_[0],X[idx['validation']]@w+b,atol=1e-9,rtol=1e-9)
    np.savez_compressed(out/'probe.npz',weight_raw=w,intercept_raw=np.array([b]),
        training_mean=mean,global_rms_scale=np.array([scale]),coefficient_scaled=model.coef_[0],
        intercept_scaled=model.intercept_,C=np.array([C]),threshold=np.array([threshold]))
    locked={'locked_at_utc':datetime.now(timezone.utc).isoformat(),'selected_C':C,
        'selected_threshold':threshold,'selection':selection,'probe_sha256':sha(out/'probe.npz'),
        'training_rows':len(idx['train']),'validation_rows':len(idx['validation']),
        'test_not_used_for_fitting_or_selection':True,'family_split_disjoint':True,
        'training_preprocessing_uses_only_train':True,'global_rms_scale':scale,
        'train_source_sha256':sha(src/'train.py'),'protocol_sha256':sha(src/'protocol.json')}
    save(out/'selection-locked-before-test.json',locked)
    # The selected model and threshold are frozen before computing held-out scores.
    scores=X.astype(np.float64)@w+b;prob=expit(scores);pred=prob>=threshold
    result={'status':'complete','selection':locked,'primary_metric_weighting':'Equal weight per stem family; variants and both prompt templates share that weight.',
        'probability_interpretation':'Class-balanced training readout; not calibrated for natural CH prevalence.',
        'dataset_sha256':sha(args.data_dir/'dataset.json'),'activations_sha256':sha(args.data_dir/'activations.npz'),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'main':{},'per_template':{},'hard_negative_groups':{}}
    for s,ii in idx.items():
        rs=[rows[i] for i in ii];fw=family_weights(rs)
        result['main'][s]={'family_weighted':metrics(y[ii],prob[ii],threshold,fw),
            'row_weighted':metrics(y[ii],prob[ii],threshold,np.ones(len(ii))),
            'words':len({r['word'] for r in rs}),'families':len({r['family'] for r in rs})}
    for template in P['templates_train']+P['templates_transfer']:
        ii=np.array([i for i,r in enumerate(rows) if r['split']=='test' and r['template']==template])
        rs=[rows[i] for i in ii]
        result['per_template'][template]=metrics(y[ii],prob[ii],threshold,family_weights(rs))
    for cat in P['negative_categories']:
        ii=np.array([i for i in idx['test'] if rows[i]['label']==1 or rows[i]['category']==cat])
        neg=np.array([i for i in ii if not rows[i]['label']])
        result['hard_negative_groups'][cat]=metrics(y[ii],prob[ii],threshold,family_weights([rows[i] for i in ii]))
        result['hard_negative_groups'][cat]['negative_false_positive_rate']=float(np.average(pred[neg],weights=family_weights([rows[i] for i in neg])))
    # Family-stratified bootstrap preserves dependence among case/space/template variants.
    test=idx['test'];tr=[rows[i] for i in test];tw=family_weights(tr)
    f_to_local={f:np.array([j for j,r in enumerate(tr) if r['family']==f]) for f in sorted({r['family'] for r in tr})}
    f_label={r['family']:r['label'] for r in tr}
    posf=[f for f in f_to_local if f_label[f]];negf=[f for f in f_to_local if not f_label[f]]
    rng=np.random.default_rng(P['seed']+1);boot=[]
    for _ in range(1000):
        counts=Counter(rng.choice(posf,len(posf),replace=True).tolist()+rng.choice(negf,len(negf),replace=True).tolist())
        bw=np.zeros(len(test))
        for f,count in counts.items():bw[f_to_local[f]]=count/len(f_to_local[f])
        active=bw>0
        boot.append(metrics(y[test][active],prob[test][active],threshold,bw[active]))
    result['test_family_bootstrap_95ci']={k:[float(np.percentile([r[k] for r in boot],2.5)),float(np.percentile([r[k] for r in boot],97.5))]
        for k in ['auroc','average_precision','balanced_accuracy','f1','precision','recall']}
    # Permute at family level, preserving the number of positive training families.
    rng=np.random.default_rng(P['seed']+2);trainf=sorted({r['family'] for r in train_rows})
    fl=np.array([next(r['label'] for r in train_rows if r['family']==f) for f in trainf])
    shuffled=[]
    with threadpool_limits(limits=4):
        for iteration in range(5):
            mapping=dict(zip(trainf,rng.permutation(fl).tolist()))
            sy=np.array([mapping[r['family']] for r in train_rows])
            sw=family_weights(train_rows,True,sy)
            control=fit_probe(XT,sy,C,sw)
            cp=control.predict_proba((X[test].astype(np.float64)-mean)/scale)[:,1]
            sm=metrics(y[test],cp,.5,tw)
            shuffled.append({'iteration':iteration,'metrics_against_true_test_labels':sm})
            print('SHUFFLE '+json.dumps({'iteration':iteration,'test_auroc':sm['auroc']}),flush=True)
    result['shuffled_family_label_controls']=shuffled
    result['diagnostic_predictions']=[{'word':r['word'],'surface':r['surface'],'template':r['template'],
        'label':r['label'],'probability':float(prob[i]),'score':float(scores[i]),'prediction':int(pred[i])}
        for i,r in enumerate(rows) if r['split']=='diagnostic']
    fields=['id','word','surface','family','label','category','split','template','case','leading_space','token_id','target_position']
    for split,ii in idx.items():
        with (out/f'{split}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows({k:rows[i][k] for k in fields} for i in ii)
    with (out/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+['probe_score','probability','prediction','correct']);writer.writeheader()
        writer.writerows({**{k:r[k] for k in fields},'probe_score':float(scores[i]),'probability':float(prob[i]),
            'prediction':int(pred[i]),'correct':bool(pred[i]==y[i])} for i,r in enumerate(rows))
    save(out/'test-errors.json',[{**rows[i],'probe_score':float(scores[i]),'probability':float(prob[i]),'prediction':int(pred[i])}
        for i in test if pred[i]!=y[i]])
    save(out/'word-splits.json',{s:sorted({rows[i]['word'] for i in ii}) for s,ii in idx.items()})
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for split,color in [('validation','#8575b5'),('test','#087f8c')]:
        ii=idx[split];fw=family_weights([rows[i] for i in ii]);m=result['main'][split]['family_weighted']
        fpr,tpr,_=roc_curve(y[ii],prob[ii],sample_weight=fw)
        axes[0].plot(fpr,tpr,label=f"{split.title()} AUROC {m['auroc']:.3f}",color=color,lw=2)
        precision,recall,_=precision_recall_curve(y[ii],prob[ii],sample_weight=fw)
        axes[1].plot(recall,precision,label=f"{split.title()} AP {m['average_precision']:.3f}",color=color,lw=2)
    axes[0].plot([0,1],[0,1],color='#999',ls='--',lw=1)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Starts with CH: held-out families',xlim=(-.02,1.02),ylim=(-.02,1.02))
    prevalence=result['main']['test']['family_weighted']['positive_prevalence']
    axes[1].axhline(prevalence,color='#999',ls='--',lw=1,label=f'Test prevalence {prevalence:.2f}')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Family-weighted precision–recall',xlim=(-.02,1.02),ylim=(-.02,1.02))
    for ax in axes:ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15)
    fig.savefig(out/'probe-validation.png',dpi=180);fig.savefig(out/'probe-validation.svg');plt.close(fig)
    result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START,
        held_out_test_errors=int(np.sum(pred[test]!=y[test])),no_alignment_or_ablation_run=True)
    save(out/'metrics.json',result)
    print('TRAINING_COMPLETE '+json.dumps({'C':C,'threshold':threshold,'test':result['main']['test'],
        'ci':result['test_family_bootstrap_95ci'],'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
