import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
for arm,color,label in [('cosine3e6','#1e6ca3','Peak LR 3e-6'),('cosine1e5','#cc652c','Peak LR 1e-5')]:
    val=json.loads((R/arm/'results/distill_validation.json').read_text())
    x=[r['step']+256 for r in val];y=[r['kl'] for r in val]
    ax.plot(x,y,'o-',markersize=4,color=color,label=label)
    i=min(range(len(val)),key=lambda i:y[i]);ax.scatter([x[i]],[y[i]],s=130,facecolors='none',edgecolors=color,lw=2,zorder=3)
ax.axhline(.05914738681167364,color='#777777',lw=1,ls='--',label='256-step starting checkpoint')
ax.set(xlabel='Cumulative training updates',ylabel='Held-out validation KL (lower is better)',
    title='Longer hidden-to-KV distillation: validation improvement saturates')
ax.set_xticks([256,512,640,1024,1536,2048]);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
fig.text(.5,-.03,'128 training documents; 4 validation documents. Rings mark validation-selected checkpoints. No RULER data used.',ha='center',fontsize=9)
fig.savefig(R/'training_validation.png',dpi=180,bbox_inches='tight');plt.close(fig)
