import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'summary.json').read_text())
methods={'raw':('Text replay','#64748b'),'raw_kv':('Text + prefix KV','#2563eb'),'h_cpu':('CPU hidden','#d97706'),'h_gpu':('GPU hidden','#059669'),'hybrid':('GPU hidden + upper KV','#9333ea')}
fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
for col,profile in enumerate(['hot','diverse']):
    for method,(label,color) in methods.items():
        rows=sorted((r for r in data['points'] if r['status']=='ok' and r['profile']==profile and r['method']==method),key=lambda r:r['concurrency'])
        x=[r['concurrency'] for r in rows]
        axes[0,col].plot(x,[r['output_tokens_per_s'] for r in rows],marker='o',markersize=4,color=color,label=label)
        axes[1,col].plot(x,[r['peak_allocated_bytes']/2**30 for r in rows],marker='o',markersize=4,color=color)
    axes[0,col].set_title('8 repeated evidence prefixes' if profile=='hot' else '84 evidence prefixes / 96 queries')
    for row in [0,1]:
        ax=axes[row,col];ax.set_xscale('log',base=2);ax.set_xticks([1,4,16],['1','4','16']);ax.set_xlabel('Concurrent requests');ax.grid(True,alpha=.18)
        ax.spines[['top','right']].set_visible(False)
axes[0,0].set_ylabel('Output tokens / second');axes[1,0].set_ylabel('Peak allocated GPU memory (GiB)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Qwen3-8B | one GPU (driver: L20D, sm_103) | 8 GiB persistent-cache budget',fontsize=12)
for suffix in ['png','pdf','svg']:fig.savefig(ROOT/('deployment_comparison.'+suffix),dpi=180)
print(str(ROOT/'deployment_comparison.png'))
