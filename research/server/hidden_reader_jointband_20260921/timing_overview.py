import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
groups=[]
for j in json.loads((R/'submissions_screen.json').read_text()):
    p=R/j['run']/'results/runtime.json'
    if p.exists():groups.append(json.loads(p.read_text()))
for chunks in [1,4,12]:
    summary=[]
    for n in [12,14,16,18,20,24,28,32,36]:
        times=[];ratios=[]
        for group in groups:
            x=next(x for x in group if x['chunks']==chunks and x['n']==n)
            ref=next(x for x in group if x['chunks']==chunks and x['n']==36)
            times.append(x['median_wall_ms']);ratios.append(x['median_wall_ms']/ref['median_wall_ms'])
        summary.append(dict(n=n,median_ms=round(statistics.median(times),2),ratio=round(statistics.median(ratios),4)))
    print(json.dumps(dict(cells=len(groups),chunks=chunks,values=summary)))
