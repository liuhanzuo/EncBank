"""Read-only candidate/frontier observation; does not launch jobs."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
R=Path('/data/liuhanzuo/comem_v2_20260908');B=R/'workspace/exp/comem_v2_benchmarks_20260908';sys.path.insert(0,str(B))
from babi16_priority_bootstrap import read,launch_allowed
from native_priority_receipt import completed
plan=read(B/'full_plan.json');state=read(R/'outputs/queue_v2/full/state.json');selected=[]
for index,job in enumerate(plan['jobs']):
    if job['benchmark']!='babilong' or job['arm'] not in ('pub','pub_sink','cbos'):continue
    cell=job['cells'][0]
    if not ((cell['task'] in ('qa2','qa3') and cell['length']=='8k') or (cell['task'] in ('qa1','qa5') and cell['length']=='32k')):continue
    allowed,reason,gap=launch_allowed(job,plan,state,completed)
    selected.append({'id':job['id'],'index':index,'main_status':state['jobs'][job['id']]['status'],'canonical_complete':completed(job),'allowed':allowed,'reason':reason,'unfinished_canonical_jobs_before_target':gap})
report={'timestamp':datetime.now(timezone.utc).isoformat(),'execution':'remote CPU read-only; no queue or model started','main_queue_pid':state['queue_pid'],'candidates':selected}
(R/'outputs/diagnostics/heartbeat_priority_candidates_20260909_0631.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
