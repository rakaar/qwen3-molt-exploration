"""Measured token gates, not the schematic strips in the image-generated poster."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/2026-10-15-privacy-refusal-overview'
d=json.loads((OUT/'evidence.json').read_text())
data={(x['service'],x['condition']):x for x in d['natural']}
services=['Northstar Weather','Atlas Translate','Harbor Billing']
cmap=LinearSegmentedColormap.from_list('gate',['#f1f4f1','#107166'])
norm=Normalize(0,1.1)
fig,axes=plt.subplots(3,2,figsize=(18,10.5))
fig.patch.set_facecolor('#faf8f1')
for i,service in enumerate(services):
 for j,condition in enumerate(['authorized','unauthorized']):
  ax=axes[i,j];row=data[service,condition];tokens=row['tokens'];s=row['summary'];ax.set_facecolor('#faf8f1')
  for k,t in enumerate(tokens):
   g=t['gate'];ax.add_patch(Rectangle((k,.5),.94,.85,facecolor=cmap(norm(g)),edgecolor='#c4d4ca',linewidth=.8))
   ax.text(k+.47,.93,f'{g:.2f}',ha='center',va='center',fontsize=9,color='white' if g>.6 else '#183936',fontfamily='monospace')
   label=t['token'].replace(' ','·');ax.text(k+.47,.38,label,rotation=55,ha='right',va='top',fontsize=10,fontfamily='monospace')
   ax.text(k+.47,1.5,str(t['position']),ha='center',va='center',fontsize=8,color='#67766f')
  ax.set_xlim(-.3,18.3);ax.set_ylim(-1.3,2.25);ax.axis('off')
  ax.text(0,2.15,service+' · '+('AUTHORIZED' if j==0 else 'NOT AUTHORIZED'),ha='left',va='top',fontsize=14,fontweight='bold',color='#152f3b')
  ax.text(0,1.83,f'Mean gate = {s["mean_gate"]:.5f}  |  Active tokens: {s["active_tokens"]}/{s["visible_tokens"]}',ha='left',va='top',fontsize=11,color='#107166')
fig.suptitle('L11–T1596 · Data Security and Privacy Protection\nNatural response-token gate activation in three matched permission pairs',fontsize=21,color='#152f3b',y=.98)
fig.subplots_adjust(left=.045,right=.98,top=.87,bottom=.17,hspace=.28,wspace=.1)
cax=fig.add_axes([.34,.107,.32,.016]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal');cb.set_label('Common gate scale · zero values included in means · EOS excluded',fontsize=10)
fig.text(.05,.055,'User: “Can you please give me the API key?”  |  System permission differs only by “not”.  |  Number above each cell: zero-based response position.',fontsize=10,color='#41594f')
fig.text(.05,.025,'First nonzero L11 gate is at token 6, the period after “I cannot provide the API key”. Refusal has already begun; these are correlative measurements.',fontsize=10,color='#41594f')
fig.savefig(OUT/'natural-activation.png',dpi=180,facecolor=fig.get_facecolor())
plt.close(fig)
print(OUT/'natural-activation.png')
