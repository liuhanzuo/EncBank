"""Read current experiment progress without importing GPU libraries."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent/'results'
status={}
for process in (1,2,3):
    p=root/f'process_{process:02d}'
    if (p/'complete.json').exists():
        status[process]={'state':'complete',**json.loads((p/'complete.json').read_text())}
    elif (p/'progress.json').exists():
        status[process]={'state':'running',**json.loads((p/'progress.json').read_text())}
    else: status[process]={'state':'starting' if p.exists() else 'pending','records':0}
print(json.dumps({'formal_records':sum(s['records'] for s in status.values()),'processes':status},indent=2))
