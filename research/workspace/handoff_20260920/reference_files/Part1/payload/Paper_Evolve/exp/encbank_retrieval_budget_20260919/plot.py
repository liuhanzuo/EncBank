import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'summary.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
for col,c in enumerate([1,4,16]):
    for method,label,color in [('raw','Text replay','#64748b'),('h_gpu','GPU hidden','#059669')]:
        rows=sorted((r for r in data['points'] if r['status']=='ok' and r['concurrency']==c and r['method']==method),key=lambda x:x['k'])
        x=[r['k'] for r in rows]
        axes[0,col].plot(x,[r['output_tokens_per_s'] for r in rows],'-o',label=label,color=color)
        axes[1,col].plot(x,[r['peak_allocated_bytes']/2**30 for r in rows],'-o',color=color)
    axes[0,col].set_title(f'Concurrency {c}')
    for row in [0,1]:
        axes[row,col].set_xticks([12,24,32,48]);axes[row,col].set_xlabel('Retrieved chunks (512 positions each)');axes[row,col].grid(alpha=.2)
        axes[row,col].spines[['top','right']].set_visible(False)
for row in [0,1]:
    axes[row,2].annotate('k=48: both OOM\n(128 GiB allocator cap)', xy=(0.97,0.83 if row==0 else 0.30), xycoords='axes fraction', ha='right', va='top', fontsize=9, color='#b45309')
axes[0,0].set_ylabel('Output tokens / second');axes[1,0].set_ylabel('Peak allocated GPU memory (GiB)')
axes[0,0].legend(frameon=False)
fig.suptitle('Qwen3-8B | fixed j=12 adapter | retrieval-budget systems pilot')
for ext in ['png','pdf','svg']:fig.savefig(ROOT/('budget_comparison.'+ext),dpi=180)
