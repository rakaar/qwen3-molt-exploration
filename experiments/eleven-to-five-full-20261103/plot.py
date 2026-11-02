"""Compare measured donor-operator and scalar-only November next-token mass."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/2026-11-03-eleven-to-five-full'
d=json.loads((OUT/'summary.json').read_text())
r=d['scalar_only_comparison']
fig,ax=plt.subplots(figsize=(8.5,4.7))
x=[a['gate'] for a in r]
ax.plot(x,[100*a['full_p_november_case_sum'] for a in r],'-o',color='#b56d22',label='Donor UVx + fixed gate',markersize=4)
ax.plot(x,[100*a['scalar_p_november_case_sum'] for a in r],'--o',color='#23688c',label='Recipient UVx + fixed gate (prior scalar sweep)',markersize=4)
ax.axvline(d['protocol']['donor_gate'],color='#64748b',linestyle=':',label='Natural donor gate: 1.1586')
ax.set(xlabel='Fixed scalar gate at recipient final input token “ is”',ylabel='P(November next-token variants), %',xlim=(-1,92),ylim=(0,100))
ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
fig.suptitle('Fifth-month prompt: transplanting the donor operator',x=.09,ha='left',fontsize=14,weight='bold')
fig.text(.09,.885,'L29–T2375 · input position 6 only · native recipient MLP remainder retained',fontsize=10,color='#475569')
fig.text(.09,.035,'Same gate does not mean equal contribution norm: donor UVx is about 2.5× larger. Equal-norm controls are separate.',fontsize=8.6,color='#475569')
fig.subplots_adjust(left=.10,right=.98,top=.81,bottom=.17)
for ext in ['png','svg']:fig.savefig(OUT/('operator-comparison.'+ext),dpi=180)
