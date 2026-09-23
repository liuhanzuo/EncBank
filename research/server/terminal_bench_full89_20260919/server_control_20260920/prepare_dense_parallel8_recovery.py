"""Prepare a fresh, audited recovery of zero-model environment failures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

B = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919')
S = B / 'server_control_20260920'
OLD = S / 'dense_parallel4_20260921'
H = S / 'dense_parallel8_recovery_20260921'
R = Path('/srv/encbank/qencbank_runtime_20260911/server_control_20260920')
TASKS = ['mcmc-sampling-stan', 'rstan-to-pystan', 'torch-pipeline-parallelism', 'torch-tensor-parallelism']

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v): p.write_text(json.dumps(v, indent=2) + '\n')

QUALIFY = r'''import asyncio,hashlib,json,os,time,traceback
from pathlib import Path
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig
H=Path(__file__).resolve().parent
P=json.loads((H/'plan.json').read_text())
T=json.loads((H/'dense_harbor_template.json').read_text())
OUT=H/'qualification'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
async def check(task,sem):
 async with sem:
  started=time.time()
  cfg=TrialConfig.model_validate(dict(task={'path':str(Path(P['task_root'])/task)},trial_name='qualification-'+task,
    trials_dir=str(OUT),install_only=True,agent=T['agents'][0],environment=T['environment'],
    verifier={},artifacts=[],extra_instruction_paths=[],extra_instructions=[]))
  trial=await Trial.create(cfg)
  policy=trial.agent_environment._network_policy.model_dump(mode='json')
  result=await trial.run()
  env=trial.agent_environment
  row=dict(task=task,status='FAIL',started_epoch=started,model_calls=0,
    network_policy=policy,legacy_allow_internet=env.task_env_config.allow_internet,
    exception_type=result.exception_info.exception_type if result.exception_info else None,
    exception_message=result.exception_info.exception_message if result.exception_info else None)
  try:
   assert result.exception_info is None,row['exception_message']
   assert result.agent_execution is None and result.verifier_result is None
   spec=json.loads((env._root/'spec.json').read_text())
   service=json.loads((env._root/'service.json').read_text())
   closure=json.loads((env._root/'closure.json').read_text())
   network=json.loads((env._root/'network_diagnostic.json').read_text())
   assert spec['internet'] is True and policy['network_mode']=='public'
   assert 'tap0' in network['stdout']
   assert int(service['memory_max'])==8192*2**20 and closure['cgroup_empty']
   row.update(status='PASS',service_root=str(env._root),closure=closure,
     resolved_internet=spec['internet'],memory_max=service['memory_max'],affinity=service['affinity'])
  except Exception:
   row['audit_error']=traceback.format_exc()
  row.update(ended_epoch=time.time(),elapsed_seconds=time.time()-started)
  save(OUT/(task+'.json'),row)
  print(json.dumps(row),flush=True)
  return row
async def main():
 OUT.mkdir(exist_ok=False)
 from server_preflight import check as preflight
 save(H/'qualification_cpu.json',preflight(check_container=False))
 sem=asyncio.Semaphore(P['concurrent_tasks'])
 rows=await asyncio.gather(*(check(t,sem) for t in P['tasks']))
 assert not list(Path(P['rpc_root']).glob('**/*.request.json'))
 report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
  scope='all_plan_task_environments',plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
  tasks=P['tasks'],checks=rows,task_concurrency=P['concurrent_tasks'],qualified_simultaneous_tasks=len(rows),
  model_calls=0,benchmark_attempts=0,qualification='Actual Harbor Trial.create and install_only Trial.run path; no model or verifier calls')
 save(H/'container_qualification.json',report)
 print(json.dumps({'status':report['status'],'configured_ceiling':P['concurrent_tasks'],'qualified_tasks':len(rows)}),flush=True)
 if report['status']!='PASS':raise SystemExit(1)
asyncio.run(main())
'''

def main():
    assert not H.exists(), 'Never overwrite a prepared or used run'
    done = read(OLD/'server_job_receipt.json')
    assert done['actual_parent_waits'] and done['owner_exit'] == done['holder_exit'] == 0
    p0 = read(OLD/'plan.json')
    all_requests = [read(f) for f in (Path(p0['rpc_root'])/'dense').glob('*.request.json')]
    evidence = []
    for task in TASKS:
        receipt_path = OLD/'execution/receipts'/(task+'.json')
        receipt = read(receipt_path)
        outcome = read(OLD/'execution/task_outcomes'/(task+'.json'))
        assert receipt['actual_parent_wait'] and receipt['requests'] == 0
        assert outcome['model_requests'] == 0 and not outcome['valid_result']
        assert not any(q['task'] == task for q in all_requests)
        results = [dict(path=x, sha256=sha(Path(x))) for x in receipt['result_paths']]
        evidence.append(dict(task=task,receipt_sha256=sha(receipt_path),results=results,classification='PRE_AGENT_ENVIRONMENT_FAILURE',model_calls=0))
    H.mkdir()
    for name, digest in read(OLD/'server_source_manifest.json').items():
        assert sha(OLD/name) == digest, name
        if name.endswith(('.py','.json','.slurm')) and not name.startswith('historical_'):
            shutil.copy2(OLD/name,H/name)
    old_root = str(OLD)
    runtime = R/H.name
    p = read(H/'plan.json')
    p.update(remote_root=str(H), tasks=TASKS, concurrent_tasks=8, host_memory_budget_mb=73728,
             evidence_id='E-TB21-DENSE-P8-ENVRECOVERY-20260921', job_name='qencbank-tb-dense-p8-recovery',
             predecessor_job_id='114849', predecessor_root=old_root,
             ipc_root=str(R.parent/'t89dp8r'), selection='Only four audited zero-model environment failures from114849',
             host_admission='Requested ceiling8; four eligible8GiB tasks;72GiB reservations with2GiB fresh headroom',
             authorization='User explicitly requested supplement task concurrency8 on2026-09-21',
             qualification_path='Harbor Trial.create plus install_only Trial.run, identical network/mount setup to execution')
    p['resource_inventory'] = [r for r in p['resource_inventory'] if r['task'] in TASKS]
    for key, child in [('rpc_root','rpc'),('results_root','results'),('controller_tmp','tmp'),('controller_cache','cache')]:
        p[key] = str(runtime/child)
    save(H/'plan.json',p)
    t=read(H/'dense_harbor_template.json')
    t.update(job_name=H.name,jobs_dir=p['results_root'],tasks=[{'path':str(Path(p['task_root'])/n)} for n in TASKS])
    save(H/'dense_harbor_template.json',t)
    env=(H/'apptainer_environment.py').read_text()
    env=env.replace("affinity = sorted(os.sched_getaffinity(0))[:max(1,int(self._effective_cpus or 1))]", """available = sorted(os.sched_getaffinity(0))
        plan = json.loads(Path(__file__).with_name('plan.json').read_text())
        task_name = self.environment_dir.parent.name
        inventory = plan['resource_inventory']
        offset = sum(int(r['cpus']) for r in inventory[:next(i for i,r in enumerate(inventory) if r['task']==task_name)])
        count = max(1,int(self._effective_cpus or 1))
        assert offset + count <= len(available), 'Insufficient allocated CPU affinity for these task environments'
        affinity = available[offset:offset+count]
        mode = self._network_policy.network_mode.value
        assert mode in {'public','no-network'}, 'Unsupported network policy; never broaden an allowlist'
        internet = mode == 'public'""")
    assert 'internet=self.task_env_config.allow_internet' in env
    env=env.replace('internet=self.task_env_config.allow_internet','internet=internet,network_policy=self._network_policy.model_dump(mode="json"),\n                    startup_lock=str(Path(__file__).with_name("mount_startup.lock"))')
    (H/'apptainer_environment.py').write_text(env)
    service=(H/'apptainer_service.py').read_text().replace('import json\n','import json\nimport fcntl\n')
    assert '        cli(cmd+[spec[\'image\'],name])' in service
    service=service.replace("        cli(cmd+[spec['image'],name])", """        # Serialize mount startup only; model requests and task execution remain concurrent.
        with Path(spec['startup_lock']).open('a') as startup_lock:
            fcntl.flock(startup_lock,fcntl.LOCK_EX)
            cli(cmd+[spec['image'],name])""")
    (H/'apptainer_service.py').write_text(service)
    slurm=(H/'server.slurm').read_text().replace(old_root,str(H)).replace(p0['ipc_root'],p['ipc_root'])
    slurm=slurm.replace(p0['job_name'],p['job_name'])
    (H/'server.slurm').write_text(slurm)
    (H/'qualify_recovery.py').write_text(QUALIFY)
    submit=(H/'submit_dense_parallel4.py').read_text().replace('task_concurrency=4','task_concurrency=8')
    submit=submit.replace("'113514'","'114849'").replace("predecessor.startswith('CANCELLED')","predecessor.startswith('COMPLETED')")
    (H/'submit_recovery.py').write_text(submit)
    save(H/'recovery_evidence.json',dict(epoch=time.time(),predecessor='114849',parent_exit=done,tasks=evidence,
         mteb_leaderboard='Excluded from automatic replay:8model calls and reward0; network-contract mismatch requires separate adjudication',
         network_fix='Use resolved runtime NetworkPolicy instead of cleared legacy allow_internet field',
         startup_fix='Disjoint CPU affinities for the four selected tasks and serialized mount startup; no model/concurrent-task serialization',
         no_model_artifacts_downloaded=True))
    qualify=f'''#!/bin/bash
#SBATCH --job-name=qencbank-p8-recovery-qualification
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output={H}/qualification-%j.log
set -euo pipefail
umask 077
cd {H}
export PYTHONPATH={H} LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
export TMPDIR={runtime}/tmp APPTAINER_CACHEDIR={R}/apptainer_cache APPTAINER_TMPDIR={runtime}/tmp
mkdir -p "$TMPDIR"
{p['harbor_python']} {H}/qualify_recovery.py
'''
    (H/'qualification.slurm').write_text(qualify)
    save(H/'server_source_manifest.json',{f.name:sha(f) for f in H.iterdir() if f.is_file() and f.name!='server_source_manifest.json'})
    result=subprocess.run(['sbatch','--parsable',str(H/'qualification.slurm')],capture_output=True,text=True)
    save(H/'qualification_submission.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time(),model_calls=0))
    assert result.returncode==0,result.stderr
    print(json.dumps(dict(prepared=str(H),qualification_job=result.stdout.strip(),concurrent_tasks=8,eligible_tasks=len(TASKS),model_calls=0)))

if __name__=='__main__':main()
