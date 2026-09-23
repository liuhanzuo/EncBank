"""Fresh research arms; no alteration or resubmission of the full89 owners."""
import ast,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from task_selection import select
SRC=Path(__file__).resolve().parent;B=SRC.parent
Q=Path(sys.argv[1]);T=B/'tf27b_hot_live_20260921'
assert Q.is_relative_to(B) and not T.exists(),'Fresh result root required'
proof=json.loads((Q/'qualification_complete.json').read_text())
assert proof['passed'] and proof['numerical'] and proof['batch']
assert json.loads((Q/'parent_exit.json').read_text())['exit_code']==0
for name,value in json.loads((Q/'source_manifest.json').read_text()).items():
    assert hashlib.sha256((Q/name).read_bytes()).hexdigest()==value,name
high=max(x['batch'] for x in proof['stress'] if x['arm']=='hot' and x['history_tokens']==131072 and x['passed'])
assert next(x for x in proof['stress'] if x['arm']=='dense' and x['batch']==8 and x['history_tokens']==131072)['passed']
P=json.loads((Q/'plan.json').read_text());old=B/'terminal_bench_full89_20260919/server_control_20260920/k12_unbounded_20260921'
# Resource-only deterministic subset, no scores or verifier contents consulted.
manifest=json.loads((old/'task_manifest.json').read_text())
tasks,inventory=select(P['task_root'],manifest['tasks'])
T.mkdir();(T/'qualification_link.json').write_text(json.dumps(dict(root=str(Q),proof=proof),indent=2)+'\n')
backend=['common.py','controller.py','parallel_trials.py','run_trial.py','tb_agent.py','apptainer_environment.py',
    'apptainer_service.py','apptainer_executor.py','cpu_slots.py','host_admission.py','server_transport.py',
    'harbor_unbounded.py','environment_template.json','worker.py','launch_trial_group.py']
core=['native_common.py','hybrid_reader.py','hybrid_hot.py','batch_cache.py','memory_selectors.py','runtime_identity.py','model_setup.py']
jobs=[]
arms=[('dense8','dense',8),('cold8','cold',8),('hot8','hot',8)]
if high>8:arms.append(('hot'+str(high),'hot',high))
for label,arm,concurrency in arms:
    R=T/label;R.mkdir()
    for n in backend:shutil.copy2(SRC/n,R/n)
    for n in core:shutil.copy2(Q/n,R/n)
    plan=dict(P,experiment='Qwen3.8-27B Transformers real Terminal-Bench hot-prefix comparison',
        tasks=tasks,resource_inventory=inventory,arm=arm,arms=[arm],task_concurrency=concurrency,decode_batch_size=concurrency,
        hot_chunks=24,qualification_root=str(Q),host_memory_budget_mb=204800,temperature=1.0,
        task_root='/srv/encbank/qcomem_runtime_20260911/server_control_20260920/tasks',
        task_selection='8 declared representative tasks then alphabetically fill to32 among <=2GiB/1CPU tasks; independent of scores',
        cache_semantics='exact ordered H21-prefix KV+DeltaNet snapshot; online H preserved identically in cold/hot',
        hot_promotion='capture newly completed 512-token online H and upper state; only identical prefix reusable',
        benchmark_accounting='independent research arms, never merged into existing full89 continuation',
        max_new_tokens=None,request_timeout=None,task_timeout=None)
    (R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    runtime=Path(P['task_root']).parent
    filtered={str(runtime/x['path']):x['sha256'] for x in manifest['files'] if any('/'+t+'/' in '/'+x['path'] for t in tasks)}
    assert filtered
    for path,expected in filtered.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    (R/'task_manifest.json').write_text(json.dumps(filtered,indent=2)+'\n')
    for d in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs','worker']:(R/d).mkdir()
    for task in tasks:
        (R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
    for p in R.glob('*.py'):ast.parse(p.read_text())
    env=os.environ.copy();env.update(PYTHONPATH=str(R),PYTHONDONTWRITEBYTECODE='1',LITELLM_LOCAL_MODEL_COST_MAP='True',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    check=subprocess.run([P['harbor_python'],'-B','-c','import controller,tb_agent,apptainer_environment'],env=env,cwd=R,capture_output=True,text=True)
    assert check.returncode==0,check.stderr
    files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file()}
    (R/'source_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    jobs.append(dict(label=label,arm=arm,concurrency=concurrency,root=str(R)))
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
assert 'tf27b-tb-' not in queue
(T/'ownership_before_submit.json').write_text(json.dumps(dict(queue=queue,epoch=time.time()),indent=2)+'\n')
for row in jobs:
    R=Path(row['root']);cpus=2*row['concurrency']+8
    cmd=['sbatch','--parsable','--job-name=tf27b-tb-'+row['label'],'--partition=gpu','--time=UNLIMITED','--no-requeue',
        '--cpus-per-task='+str(cpus),'--mem=160G','--gres=gpu:nvidia_l20d:1','--exclude=gpu-node6',
        '--chdir='+str(R),'--output='+str(R/'logs/slurm-%j.out'),'--error='+str(R/'logs/slurm-%j.err'),
        '--wrap',P['engine_python']+' -B '+str(R/'launch_trial_group.py')]
    res=subprocess.run(cmd,capture_output=True,text=True);assert res.returncode==0,res.stderr
    row.update(job=res.stdout.strip().split(';')[0],argv=cmd)
    (T/'submissions.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps(dict(tasks=tasks,jobs=jobs),indent=2))
