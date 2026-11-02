"""Plot all tested scalar strengths, preserving small target probabilities."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/2026-11-03-eleven-fifth-element-scalar'
d=json.loads((OUT/'summary.json').read_text())
rows=d['runs'];g=[r['target_gate'] for r in rows]
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
colors={'p_boron':'#286450','p_sodium':'#bd681e','p_aluminum_spellings':'#6952a0'}
for ax in axes:
    for key,label in [('p_boron','Boron (5th)'),('p_sodium','Sodium (11th)'),('p_aluminum_spellings','Aluminum / aluminium (13th)')]:
        ax.plot(g,[100*r[key] for r in rows],'.-',color=colors[key],label=label,markersize=5)
    ax.set_xlabel('Fixed gate at final input “ is”');ax.set_ylabel('Exact answer-prefix probability (%)')
    ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax.axvline(d['protocol']['reference_month_gate'],ls=':',color='#6b7280',lw=1)
axes[0].set_ylim(0,100);axes[0].set_title('Absolute probability');axes[0].legend(frameon=False,fontsize=8)
axes[1].set_yscale('log');axes[1].set_title('Log scale exposes small target probabilities')
fig.suptitle('L29–T2375 scalar-only sweep: “the fifth element in the periodic table is”',fontsize=12)
fig.supxlabel('Recipient UVx retained; only final input edited. Lowercase/title-case prefixes summed; 32-token greedy responses saved.',fontsize=8)
for ext in ['png','svg']:fig.savefig(OUT/('probability-sweep.'+ext),dpi=180)
plt.close(fig)
print(OUT/'probability-sweep.png')
