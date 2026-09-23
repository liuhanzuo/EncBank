import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=[r for r in json.loads((ROOT/'summary.json').read_text())['points'] if r['status']=='ok']
fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained',sharey=True)
y=list(range(len(rows)));names=[r['arm'] for r in rows]
for ax,vals,label in [(axes[0],[r['mean_chunks'] for r in rows],'Mean retrieved chunks'),(axes[1],[r['ttft_ms']['p50'] for r in rows],'Median TTFT (ms)')]:
    ax.barh(y,vals,color=['#059669' if 'qk' in r['arm'] else '#64748b' for r in rows]);ax.set_xlabel(label);ax.set_yticks(y,names,fontsize=8);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis();fig.suptitle('Qwen3-8B | adaptive chunk retrieval pilot | costs only, no task quality')
for ext in ['png','pdf','svg']:fig.savefig(ROOT/('selector_comparison.'+ext),dpi=180)
