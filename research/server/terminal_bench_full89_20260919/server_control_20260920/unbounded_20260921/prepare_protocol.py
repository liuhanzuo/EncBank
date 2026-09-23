"""Materialize new immutable protocol roots from templates and an audited task list."""
import argparse,hashlib,json,shutil,tomllib
from pathlib import Path

B=Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919')
S=B/'server_control_20260920'; U=S/'unbounded_20260921'
R=Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(arm,name,tasks,qualification=False):
    h=S/name;assert not h.exists(), 'A prepared run is immutable'
    source=U/'templates'/arm;h.mkdir()
    for f in source.iterdir():
        if f.is_file() and f.suffix in {'.py','.json','.slurm'} and f.name not in {
            'submission.json','server_source_manifest.json','container_qualification.json',
            'qualification_submission.json','server_preflight.json','qualification_cpu.json'}:
            shutil.copy2(f,h/f.name)
    p=json.loads((source/'plan.json').read_text());old=p['remote_root'];runtime=R/name
    p.update(remote_root=str(h),tasks=tasks,concurrent_tasks=8,max_new_tokens=None,
       per_call_transport_watchdog_seconds=None,bootstrap_timeout_seconds=None,
       task_walltime_seconds=None,gpu_allocation_walltime_seconds=None,
       official_task_timeouts=False,server_only=True,automatic_scientific_retries=0,
       host_memory_budget_mb=204800,host_admission_root=str(R/'unbounded_host_admission'),
       job_name='qcomem-tb-'+arm+'-unbounded',evidence_id='E-TB21-UNBOUNDED-20260921-'+arm,
       ipc_root=str(R.parent/('t89ub'+arm)),
       timeout_protocol='No task, request, verifier, setup, idle, or Slurm deadline; elapsed time recorded. Native position capacity and physical memory limits remain explicit.',
       token_protocol='No configured generation-token cap. Stop on EOS; report native-context capacity as incomplete, never ordinary quality zero.',
       selection='User-authorized token/time-truncated replay plus interrupted and never-started tasks, reconciled with final old-machine handoff.',
       container_qualified_hostnames=['gpu-host'],qualification_only=qualification)
    for k,child in [('rpc_root','rpc'),('results_root','results'),('controller_tmp','tmp'),('controller_cache','cache')]:
        p[k]=str(runtime/child)
    inventory=[]
    for task in tasks:
        t=tomllib.loads((Path(p['task_root'])/task/'task.toml').read_text())['environment']
        inventory.append(dict(task=task,memory_mb=int(t.get('memory_mb',2048)),cpus=int(t.get('cpus',1))))
    p['resource_inventory']=inventory
    if arm=='dense':
        p.update(max_concurrent_requests=8,max_num_seqs=8,request_token_budget=8*p['context_tokens'])
    else:
        p.update(max_live_sessions=8,decode_batch_size=8,native_allocator_cap_gib=248,
                 device_cap_gib=256,kv_bytes_per_token=65536,decode_workspace_gib=32)
    save(h/'plan.json',p)
    kind=p['arm'];t=json.loads((h/(kind+'_harbor_template.json')).read_text())
    t.update(job_name=name,jobs_dir=p['results_root'],tasks=[{'path':str(Path(p['task_root'])/n)} for n in tasks])
    save(h/(kind+'_harbor_template.json'),t)
    slurm=(source/'server.slurm').read_text().replace(old,str(h))
    oldipc=json.loads((source/'plan.json').read_text())['ipc_root'];slurm=slurm.replace(oldipc,p['ipc_root'])
    lines=[]
    for l in slurm.splitlines():
        if l.startswith('#SBATCH --job-name='):l='#SBATCH --job-name='+p['job_name']
        elif l.startswith('#SBATCH --cpus-per-task='):l='#SBATCH --cpus-per-task=32'
        elif l.startswith('#SBATCH --mem='):l='#SBATCH --mem=160G'
        elif l.startswith('#SBATCH --time='):l='#SBATCH --time=0'
        elif l.startswith('#SBATCH --exclude='):continue
        lines.append(l)
    (h/'server.slurm').write_text('\n'.join(lines)+'\n')
    if qualification:
        shutil.copy2(U/'qualify_all_environments.py',h/'qualify_all_environments.py')
        q=f'''#!/bin/bash
#SBATCH --job-name=qcomem-unbounded-envcheck
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output={h}/qualification-%j.log
set -euo pipefail
umask 077
cd {h}
export PYTHONPATH={h} PYTHONUNBUFFERED=1 LITELLM_LOCAL_MODEL_COST_MAP=True
{p['harbor_python']} -u qualify_all_environments.py
'''
        (h/'qualification.slurm').write_text(q)
    manifest={f.name:sha(f) for f in h.iterdir() if f.is_file()}
    save(h/'server_source_manifest.json',manifest)
    print(json.dumps(dict(root=str(h),tasks=len(tasks),concurrency=8,manifest_sha256=sha(h/'server_source_manifest.json'))))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--qualification',action='store_true');a.add_argument('--selection');args=a.parse_args()
    if args.qualification:
        tasks=sorted(x.name for x in (R/'tasks').iterdir() if (x/'task.toml').exists());assert len(tasks)==89
        tasks.sort(key=lambda n:(tomllib.loads((R/'tasks'/n/'task.toml').read_text())['environment'].get('memory_mb',2048),n))
        prepare('dense','unbounded_env_qualification_20260921',tasks,True)
    else:
        assert args.selection,'Final audited selection required'
        selection=json.loads(Path(args.selection).read_text());assert selection['status']=='PASS'
        for arm in ['dense','k12','k48']:prepare(arm,arm+'_unbounded_20260921',selection['arms'][arm]['tasks'])
