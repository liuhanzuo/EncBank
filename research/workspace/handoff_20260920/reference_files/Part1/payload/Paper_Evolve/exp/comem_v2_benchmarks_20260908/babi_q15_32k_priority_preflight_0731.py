"""CPU boundaries plus current predecessor/frontier/GPU admission; optional launch."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import argparse,copy,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from babi_q15_32k_priority_bootstrap import R,B,read,alive,verify_scope,verify_predecessor,launch_allowed
from native_priority_receipt import completed
from remote_queue import gpu_idle
ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');args=ap.parse_args()
plan=read(B/'babi_q15_32k_priority_full_plan.json');main=read(B/'full_plan.json');state=read(R/'outputs/queue_v2/full/state.json')
verify_scope(plan,main);checks=['six_exact_canonical_jobs_600']
for what in ('task','length','argv'):
    changed=copy.deepcopy(plan)
    if what=='argv':changed['jobs'][0]['argv'].append('--unexpected')
    else:changed['jobs'][0]['cells'][0][what]='wrong'
    try:verify_scope(changed,main)
    except RuntimeError:checks.append('reject_changed_'+what)
    else:raise AssertionError('scope alteration passed')
small={'jobs':[{'id':f'j{i}'} for i in range(41)]};entry={'jobs':{j['id']:{'status':'pending'} for j in small['jobs']}}
assert launch_allowed(small['jobs'][40],small,entry,lambda j:False)==(True,'far_from_main_frontier',40);checks.append('distance40_admitted')
assert not launch_allowed(small['jobs'][39],small,entry,lambda j:False)[0];checks.append('distance39_blocked')
entry['jobs']['j40']['status']='running';assert not launch_allowed(small['jobs'][40],small,entry,lambda j:False)[0];checks.append('main_active_blocked')
entry['jobs']['j40']['status']='pending';assert launch_allowed(small['jobs'][40],small,entry,lambda j:True)[1]=='canonical_complete_reuse_before_model_load';checks.append('canonical_complete_preload_reuse')
assert alive(state['queue_pid']);proof=verify_predecessor();idle,reading=gpu_idle(1,512)
frontier=[]
for job in plan['jobs']:
    allowed,reason,gap=launch_allowed(job,main,state,completed)
    frontier.append({'job':job['id'],'allowed':allowed,'reason':reason,'main_unfinished_canonical_jobs_before_target':gap,'canonical_complete':completed(job),'main_status':state['jobs'][job['id']]['status']})
report={'timestamp':datetime.now(timezone.utc).isoformat(),'cpu_only':True,'checks':checks,'main_pid':state['queue_pid'],'predecessor':proof,'gpu1_idle':idle,'gpu1':reading,'frontier':frontier,'launched':False}
assert idle and all(x['allowed'] and not x['canonical_complete'] for x in frontier)
if args.launch:
    out=R/'outputs/bootstrap_babi_q15_32k_priority';out.mkdir(parents=True,exist_ok=True)
    old=read(out/'status.json');assert not old,'retain existing bootstrap instead of duplicate launch'
    argv=[sys.executable,'-u',str(B/'babi_q15_32k_priority_bootstrap.py')]
    attempt=1
    logpath=out/'bootstrap.log'
    while logpath.exists():
        attempt+=1;logpath=out/f'bootstrap_attempt{attempt}.log'
    with logpath.open('x') as log:
        child=subprocess.Popen(argv,cwd=R/'workspace',env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    report.update(launched=True,attempt=attempt,bootstrap_log=str(logpath),bootstrap_pid=child.pid,argv=argv,bootstrap_dir=str(out),status_file=str(out/'status.json'))
name='heartbeat_babi_q15_32k_priority_20260909_0731_launch.json' if args.launch else 'heartbeat_babi_q15_32k_priority_20260909_0731_preflight.json'
dest=R/'outputs/diagnostics'/name
if args.launch and dest.exists():
    history=dest.with_name(dest.stem+'_attempt1.json')
    if not history.exists():history.write_text(dest.read_text())
dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
