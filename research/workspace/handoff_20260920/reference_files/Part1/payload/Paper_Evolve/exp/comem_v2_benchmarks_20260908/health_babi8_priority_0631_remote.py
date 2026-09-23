"""One post-launch process/lock/progress check; no model work."""
import json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from babi8_priority_bootstrap import R,B,read,alive
state=read(R/'outputs/bootstrap_babi8_priority/status.json')
assert state['status']=='running' and alive(state['pid'])
qpath=Path(state['active_queue_state']);queue=read(qpath);entry=queue['jobs'][state['active_job']]
assert entry['status']=='running' and alive(queue['queue_pid']) and alive(entry['pid'])
plan=read(B/'babi8_priority_full_plan.json');job=next(j for j in plan['jobs'] if j['id']==state['active_job']);out=Path(job['output'])
lock=read(out/'RUNNING.lock');assert lock['pid']==entry['pid']
log=Path(entry['log']);lines=log.read_text(errors='replace').splitlines()
main=read(R/'outputs/queue_v2/full/state.json');active=[]
for k,v in main['jobs'].items():
    if v['status']=='running':
        lp=Path(v['log']);tail=lp.read_text(errors='replace').splitlines()[-3:]
        active.append({'job':k,'pid':v['pid'],'alive':alive(v['pid']),'gpu':v['gpu'],'log_mtime_utc':datetime.fromtimestamp(lp.stat().st_mtime,timezone.utc).isoformat(),'tail':tail})
gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert [int(x) for x in gpu.splitlines() if x.strip()]==[entry['pid']]
report={'timestamp':datetime.now(timezone.utc).isoformat(),'bootstrap':state,'bootstrap_alive':alive(state['pid']),'queue_pid':queue['queue_pid'],'queue_alive':alive(queue['queue_pid']),'active_job':entry,'model_alive':alive(entry['pid']),'canonical_job':job['id'],'canonical_output':str(out),'canonical_lock':lock,'first_progress_log':str(log),'first_progress_tail':lines[-10:],'main_queue_pid':main['queue_pid'],'main_alive':alive(main['queue_pid']),'main_counts':dict(Counter(v['status'] for v in main['jobs'].values())),'main_active':active,'gpu1_only_expected_model_pid':entry['pid'],'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),'compute_snapshot':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)}
for p in out.glob('attempts/*/native/*.csv'):
    import csv
    with p.open(newline='') as f:rows=list(csv.DictReader(f))
    report['partial_progress']={'source':str(p),'rows':len(rows),'last_index':rows[-1]['index'] if rows else None,'not_a_complete_score':True}
assert report['main_alive'] and all(x['alive'] for x in active)
(R/'outputs/diagnostics/heartbeat_babi8_priority_20260909_0631_health.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('bootstrap','active_job')},indent=2))
