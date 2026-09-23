"""Select by declared resources only; never read solutions or verifier code."""
import tomllib
from pathlib import Path

def select(task_root,names):
    inventory=[]
    for name in sorted(names):
        data=tomllib.loads((Path(task_root)/name/'task.toml').read_text())['environment']
        inventory.append(dict(task=name,memory_mb=data['memory_mb'],cpus=data['cpus'],gpus=data.get('gpus',0)))
    priority=['cancel-async-tasks','custom-memory-heap-crash','large-scale-text-editing','log-summary-date-ranges',
        'query-optimize','sqlite-db-truncate','regex-chess','vulnerable-secret']
    eligible=[x for x in inventory if x['memory_mb']<=2048 and x['cpus']<=1 and x['gpus']==0]
    by={x['task']:x for x in eligible};assert all(x in by for x in priority)
    tasks=(priority+[x['task'] for x in eligible if x['task'] not in priority])[:32]
    assert len(tasks)==32
    return tasks,inventory
