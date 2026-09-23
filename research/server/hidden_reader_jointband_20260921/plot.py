import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
screen=json.loads((R/'screen_scores.json').read_text());confirm=json.loads((R/'confirmation_scores.json').read_text())
rt=json.loads((R/'runtime_summary.json').read_text());sel=json.loads((R/'selection.json').read_text())
depths=[12,14,16,18,20,24,28,32,36]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for task,color,label in [('single','#37909b','Single NIAH'),('multikey','#bd7135','Multi-key NIAH'),('vt','#5269b7','Variable tracking (Encbank)')]:
    values=[screen['tasks'][task]['score']['n'+str(n)] for n in depths]
    axes[0].plot(depths,values,'o-',color=color,label=label)
    n=sel['selected_n'];axes[0].scatter([n],[confirm['tasks'][task]['score']['n'+str(n)]],marker='X',color=color,s=110,edgecolors='black',linewidths=.6,zorder=5)
axes[0].set(title='Quality vs joint depth',xlabel='Last joint block n',ylabel='Reference-answer recall (%)',ylim=(-2,102))
axes[0].legend(frameon=False,fontsize=9);axes[0].grid(axis='y',alpha=.2)
for chunks,color in [(1,'#8b9196'),(4,'#53957b'),(12,'#126da3')]:
    values=[next(x['median_paired_ratio_to_n36'] for x in rt if x['chunks']==chunks and x['n']==n) for n in depths]
    axes[1].plot(depths,values,'o-',color=color,label=f'{chunks} history chunks')
axes[1].axhline(1,color='gray',linestyle='--',linewidth=1)
axes[1].set(title='Measured history KV reconstruction time',xlabel='Last joint block n',ylabel='Wall time / n36 wall time (lower is better)')
axes[1].legend(frameon=False);axes[1].grid(axis='y',alpha=.2)
fig.suptitle('Joint causal attention followed by independent chunk attention',fontsize=14)
fig.text(.5,-.03,'Circles: first32/cell screening. X: selected depth on remaining68/cell. Original model projections and MLPs run at every layer.',ha='center',fontsize=9)
fig.savefig(R/'joint_depth_tradeoff.png',dpi=180,bbox_inches='tight');plt.close(fig)
