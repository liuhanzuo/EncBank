import ast,hashlib,json,os,shutil,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;OLD=A.with_name('hidden_reader_terminal_r4_20260921');R=A.with_name('hidden_reader_terminal_r5_20260921')
assert not R.exists()
# Close only the known incomplete R4 dispatcher and its not-yet-started worker.
pid=json.loads((A/'dispatch_started.json').read_text())['pid']
try:
    args=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
    assert str(A/'dispatch_steps.py').encode() in args
    os.kill(pid,15)
except FileNotFoundError:pass
for j in json.loads((OLD/'jobs/submissions_run.json').read_text()):
    if j['kind']!='worker':continue
    task=j['task'];box=OLD/'mailbox'/task
    assert not list(box.glob('*.request.json')),'Unexpected live trial; review before retirement'
    state=subprocess.check_output(['squeue','-j',j['job'],'-h','-o','%T'],text=True).strip()
    if state=='PENDING':subprocess.run(['scancel',j['job']],check=True)
    elif state:
        assert state in ['RUNNING','COMPLETING'],state
        (box/'stop.json').write_text(json.dumps(dict(origin='known incomplete controller setup; zero live model requests')))
P=json.loads((OLD/'plan.json').read_text());R.mkdir();roots={}
for task in P['tasks']:
    C=R/task;C.mkdir();roots[task]=str(C)
    for p in OLD.iterdir():
        if p.is_file() and (p.suffix=='.py' or p.name in ['environment_template.json','task_manifest.json','dependency_check.json']):shutil.copy2(p,C/p.name)
    for name in ['phase_child.py','integrated_launch.py']:shutil.copy2(A/name,C/name)
    shutil.copytree(OLD/'vendor',C/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
    plan=dict(P,tasks=[task],predecessor=str(OLD),revision_reason='Integrated Slurm ownership; unchanged R4 model code. Qualify actual task environment before loading model; separate physical CPU cores. Complete backend dependency copy.')
    (C/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(C/n).mkdir()
    for n in ['mailbox','pairs']:(C/n/task).mkdir()
    for p in C.rglob('*.py'):ast.parse(p.read_text())
    env=os.environ.copy();env.update(PYTHONPATH=str(C),LITELLM_LOCAL_MODEL_COST_MAP='True',PYTHONDONTWRITEBYTECODE='1')
    check=subprocess.run([P['harbor_python'],'-B','-c','import controller, tb_agent, apptainer_environment; print("all CPU imports passed")'],cwd=C,env=env,capture_output=True,text=True)
    (C/'dependency_import_check.json').write_text(json.dumps(dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr),indent=2)+'\n')
    assert check.returncode==0,check.stderr
    manifest={str(p.relative_to(C)):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.rglob('*') if p.is_file() and (p.suffix=='.py' or p.name in ['plan.json','environment_template.json','task_manifest.json'])}
    (C/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'roots.json').write_text(json.dumps(roots,indent=2)+'\n')
jobs=[]
for task,path in roots.items():
    C=Path(path);args=['sbatch','--parsable','--job-name=bandtb5-'+task,'--partition=gpu','--time=UNLIMITED','--no-requeue',
        '--cpus-per-task=8','--mem=64G','--gres=gpu:nvidia_l20d:1','--exclude=gpu-node1,gpu-node2,gpu-node5,gpu-node6,gpu-node7',
        '--chdir='+path,'--output='+str(C/'logs'/'slurm-%j.out'),'--error='+str(C/'logs'/'slurm-%j.err'),
        '--wrap',P['engine_python']+' -B '+str(C/'integrated_launch.py')]
    res=subprocess.run(args,capture_output=True,text=True);assert res.returncode==0,res.stderr
    jobs.append(dict(task=task,root=path,job=res.stdout.strip().split(';')[0],argv=args))
    (R/'submissions.json').write_text(json.dumps(jobs,indent=2)+'\n')
print(json.dumps(jobs,indent=2))
