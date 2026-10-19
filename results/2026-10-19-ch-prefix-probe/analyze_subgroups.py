"""Post-fit subgroup checks; do not fit or select any model settings."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'trained-probe/predictions.csv').open()))
summary={}
for key,value in [('leading_space','True'),('leading_space','False'),('case','lower'),('case','title'),('case','upper')]:
    rs=[r for r in rows if r['split']=='test' and r['template']!='transfer_upper' and r[key]==value]
    counts=Counter(r['family'] for r in rs)
    w=np.array([1/counts[r['family']] for r in rs])
    y=np.array([int(r['label']) for r in rs])
    prob=np.array([float(r['probability']) for r in rs])
    pred=np.array([int(r['prediction']) for r in rs])
    both=len(np.unique(y))==2
    summary[key+'='+value]={'rows':len(rs),'families':len(counts),
        'positive_families':len({r['family'] for r in rs if r['label']=='1'}),
        'auroc':float(roc_auc_score(y,prob,sample_weight=w)) if both else None,
        'balanced_accuracy':float(balanced_accuracy_score(y,pred,sample_weight=w)) if both else None,
        'positive_prediction_rate':float(np.average(pred,weights=w)),
        'note':None if both else 'Only one class is present; AUROC and balanced accuracy are not applicable.'}
(ROOT/'subgroup-validation.json').write_text(json.dumps({'post_fit_diagnostic_only':True,'no_retraining_or_threshold_changes':True,'groups':summary},indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
