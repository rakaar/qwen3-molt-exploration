"""Show answer confidence after each independent natural contribution removal."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/2026-11-03-eleven-active-controls'
d=json.loads((OUT/'summary.json').read_text())
rows=[d['native']]+d['runs'];labels=['Native']+[f'T{r["transform"]} off · L2 {r["contribution_l2"]:.2f}' for r in d['runs']]
y=np.arange(len(rows));fig,ax=plt.subplots(figsize=(9,5.8))
fig.subplots_adjust(left=.23,right=.98,bottom=.18,top=.82)
for name,offset,color in [('november',-.17,'#286450'),('december',.17,'#bd681e')]:
    values=[100*r['month_probabilities'][name] for r in rows]
    bars=ax.barh(y+offset,values,height=.31,color=color,label=name.title())
    ax.bar_label(bars,labels=[f'{x:.2f}%' for x in values],padding=3,fontsize=8)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,105)
ax.set_xlabel('Exact answer-prefix probability (%)');ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.0),ncols=2)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('Eleventh-month prompt: ablate each active L29 MOLT separately',fontsize=11,y=.97)
fig.supxlabel('Only final input “ is” edited; native contribution magnitudes differ. Lowercase/title-case variants summed.',fontsize=8)
for ext in ['png','svg']:fig.savefig(OUT/('active-control-probabilities.'+ext),dpi=180)
plt.close(fig)
print(OUT/'active-control-probabilities.png')
