from pathlib import Path
import json
root=Path(__file__).resolve().parent
status={}
for label,sub in [('dense_long','results'),('paired_short','short_results')]:
    status[label]={}
    for p in (1,2,3):
        folder=root/sub/f'process_{p:02d}'
        path=folder/('complete.json' if (folder/'complete.json').exists() else 'progress.json')
        status[label][p]=json.loads(path.read_text()) if path.exists() else {'state':'starting' if folder.exists() else 'pending'}
print(json.dumps(status,indent=2))
