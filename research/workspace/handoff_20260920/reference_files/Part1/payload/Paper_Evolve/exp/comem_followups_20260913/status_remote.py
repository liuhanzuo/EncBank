from pathlib import Path
import json,subprocess
h=Path(__file__).resolve().parent
out={}
for name in ('kv_quality','clean_quality'):
    shards=[]
    for i in range(4):
        p=h/name/f'shard_{i:02d}';d={'shard':i,'complete':(p/'complete.json').exists()}
        if (p/'progress.json').exists():d.update(json.loads((p/'progress.json').read_text()))
        shards.append(d)
    out[name]={'complete':all(d['complete'] for d in shards),'shards':shards}
out['jobs']=subprocess.run(['sacct','-j','28250,28251,28261','--noheader','--parsable2','--format=JobID,State,ExitCode'],text=True,capture_output=True).stdout
print(json.dumps(out))
