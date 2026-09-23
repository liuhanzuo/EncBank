"""One bounded receipt snapshot and actual process health; no model work."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import json,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
R=Path('/data/liuhanzuo/encbank_v2_20260908');B=R/'workspace/exp/encbank_v2_benchmarks_20260908';D=R/'outputs/diagnostics'
sys.path.insert(0,str(B))
from native_priority_receipt import completed
def read(p):return json.loads(p.read_text())
def alive(p):
    if not p:return False
    try:return Path(f'/proc/{int(p)}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def proc(p):
    return {'pid':p,'alive':alive(p),'command':Path(f'/proc/{p}/cmdline').read_bytes().replace(b'\0',b' ').decode() if alive(p) else None}
def stamp(p):return datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()
plan=read(B/'full_plan.json');prior=read(D/'heartbeat_babilong_20260909_1032.json')
old={(x['task'],x['length'],x['arm']) for x in prior['carried_forward_baseline_cells']+prior['writeback']}
assert len(old)==58
snapshot={'timestamp':datetime.now(timezone.utc).isoformat(),'host':os.uname().nodename,'ssh_alias':'longjing-1','prior_babi_cells':58,'completed_jobs':[],'incomplete_jobs':[],'invalid_receipts':[]}
for job in plan['jobs']:
    if job['benchmark']=='longbench':wanted=False
    elif job['benchmark']=='babilong':wanted=job['arm'] in ('pub','pub_sink','cbos') and (job['cells'][0]['task'],job['cells'][0]['length'],job['arm']) not in old
    else:wanted=False
    if not wanted:continue
    try:
        if completed(job):snapshot['completed_jobs'].append(job)
        else:snapshot['incomplete_jobs'].append(job['id'])
    except Exception as e:snapshot['invalid_receipts'].append({'job':job['id'],'error':repr(e)})
D.mkdir(parents=True,exist_ok=True);(D/'heartbeat_accuracy_20260909_1132_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
state=read(R/'outputs/queue_v2/full/state.json')
health={'timestamp':datetime.now(timezone.utc).isoformat(),'host':os.uname().nodename,'main_queue':proc(state['queue_pid']),'main_coordinator_counts':dict(Counter(x['status'] for x in state['jobs'].values())),'main_active':[]}
for key,v in state['jobs'].items():
    if v['status']!='running':continue
    p=Path(v['log']);lines=p.read_text(errors='replace').splitlines()
    useful=[x for x in lines if ('score=' in x or 'sample ' in x or 'Wrote ' in x or 'index=' in x or 'recall=' in x)]
    health['main_active'].append({'job':key,**proc(v['pid']),'gpu':v['gpu'],'log':str(p),'log_modified_utc':stamp(p),'progress':useful[-3:],'tail':lines[-3:]})
for name in ('main','trained_pub','cacheblend16'):
    p=R/'outputs/report_progress'/f'{name}_progress.json';obj=read(p)
    health.setdefault('published_progress',{})[name]={'file':str(p),'modified_utc':stamp(p),**{k:obj[k] for k in ('planned_jobs','completed_jobs','observed_predictions','all_complete','incomplete_or_invalid_jobs')}}
training=read(R/'outputs/8b_j12_pub_4k/status.json');health['training']={k:training.get(k) for k in ('step','target_steps','complete','pid','status')}
trainrows=(R/'outputs/8b_j12_pub_4k/train.jsonl').read_text().splitlines();health['training_last_log']=json.loads(trainrows[-1])
for name in ('trained_pub','cacheblend16'):
    p=R/'outputs'/f'bootstrap_{name}'/'status.json';obj=read(p)
    health.setdefault('strong_baseline_bootstrap',{})[name]={'file':str(p),'status':obj.get('status'),'pid':obj.get('pid'),'pid_alive':alive(obj.get('pid')),'child_pid':obj.get('child_pid'),'child_alive':alive(obj.get('child_pid'))}
for name in ('babi_q15_16k_priority',):
    obj=read(R/'outputs'/f'bootstrap_{name}'/'status.json')
    health[name]={'status':obj.get('status'),'pid':obj.get('pid'),'pid_alive':alive(obj.get('pid')),'child_pid':obj.get('child_pid'),'child_alive':alive(obj.get('child_pid')),'completed_jobs':obj.get('completed_jobs'),'completed_predictions':obj.get('completed_predictions'),'jobs_count':len(obj.get('jobs',{})),'state':obj}

health['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
health['compute']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
(D/'heartbeat_accuracy_20260909_1132_health.json').write_text(json.dumps(health,indent=2)+'\n')
print(json.dumps({'snapshot':snapshot['timestamp'],'new_receipt_jobs':[j['id'] for j in snapshot['completed_jobs']],'invalid':snapshot['invalid_receipts'],'main':health['main_coordinator_counts'],'active':health['main_active'],'training':health['training'],'baseline_status':health['strong_baseline_bootstrap'],'babi_q15_16k':{k:v for k,v in health['babi_q15_16k_priority'].items() if k!='state'},'published':health['published_progress']},indent=2))





