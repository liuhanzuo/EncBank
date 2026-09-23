import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;s=json.loads((R/'scores.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharey=True,layout='constrained')
styles=[('comem','Original COMem','#111111','o','-'),('comem_split','True-KV split control','#888888','x','--'),
    ('kd256','KD 256 updates','#66a7cf','s','-'),('kd_selected','KD 640 (validation-selected)','#126da3','o','-'),('kd2048','KD 2048 endpoint','#cc652c','^','-')]
for ax,task,title in zip(axes,['single','multikey','vt'],['Single-needle NIAH','Multi-key NIAH','Variable tracking (COMem version)']):
    for arm,label,color,marker,line in styles:
        y=[s['cells'][task+length]['score'][arm] for length in ['8k','32k','128k']]
        ax.plot([0,1,2],y,color=color,marker=marker,linestyle=line,label=label,markersize=6,alpha=.9)
    ax.set(title=title,xticks=[0,1,2],xticklabels=['8K','32K','128K'],xlabel='Source context length',ylim=(-3,103))
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Reference-answer recall (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=10)
fig.suptitle('Hidden-to-KV projection: paired quality evaluation',fontsize=16)
fig.savefig(R/'benchmark_scores.png',dpi=170,bbox_inches='tight');plt.close(fig)
