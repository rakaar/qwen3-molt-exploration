"""Plot measured next-token probabilities for the predeclared scalar grid."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/2026-11-03-eleven-to-five-scalar'
d=json.loads((OUT/'summary.json').read_text())
rows=d['runs']
x=[r['target_gate'] for r in rows]
fig,ax=plt.subplots(figsize=(8.5,4.7))
ax.plot(x,[100*r['p_may_case_sum'] for r in rows],'-o',color='#23688c',label='May (lowercase + title case)',markersize=4)
ax.plot(x,[100*r['p_november_case_sum'] for r in rows],'-o',color='#b56d22',label='November (lowercase + title case)',markersize=4)
ax.axvline(d['protocol']['donor_gate'],color='#64748b',linestyle=':',label='Natural eleventh donor gate: 1.1586')
ax.set(xlabel='Fixed scalar gate at final input token “ is”',ylabel='First-token probability (%)',ylim=(-2,100),xlim=(-1,92))
ax.grid(alpha=.18)
ax.legend(frameon=False,fontsize=9)
fig.suptitle('Fifth-month prompt: scalar-only L29–T2375 steering',x=.09,ha='left',fontsize=14,weight='bold')
fig.text(.09,.885,'Recipient operator retained · final input position 6 only · no generated-token edits',fontsize=10,color='#475569')
fig.text(.09,.03,'Probabilities sum the exact leading-space lowercase/title-case one-token forms; punctuation and EOS excluded.',fontsize=8.6,color='#475569')
fig.subplots_adjust(left=.10,right=.98,top=.81,bottom=.17)
for ext in ['png','svg']:fig.savefig(OUT/('probability-sweep.'+ext),dpi=180)
