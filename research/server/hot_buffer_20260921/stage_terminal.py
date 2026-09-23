"""Fresh real tasks; each arm has independent ownership, so one long run cannot block its controls."""
import ast,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent;T=B/'hot_buffer_terminal_20260921';OLD=B/'hidden_reader_terminal_r5_20260921'/'query-optimize'
assert not T.exists(),'Never duplicate hot-buffer trials'
assert json.loads((H/'complete.json').read_text())['passed']
assert json.loads((H/'parent_exit.json').read_text())['exit_code']==0
for name,h in json.loads((H/'source_manifest.json').read_text()).items():assert hashlib.sha256((H/name).read_bytes()).hexdigest()==h
T.mkdir();tasks=['large-scale-text-editing','log-summary-date-ranges'];arms=['cold36','hot36_c24','hot24_c24'];jobs=[]
P=json.loads((OLD/'plan.json').read_text());backend_names=['common.py','controller.py','run_trial.py','tb_agent.py','apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','host_admission.py','server_transport.py','harbor_unbounded.py','phase_child.py','integrated_launch.py','environment_template.json','task_manifest.json']
for task in tasks:
    for arm in arms:
        R=T/(task+'--'+arm);R.mkdir()
        for name in backend_names:shutil.copy2(OLD/name,R/name)
        agent=(R/'tb_agent.py').read_text()
        agent=agent.replace('cache_tokens=0,cost_usd=0',"cache_tokens=result.get('events',[{}])[0].get('hit_chunks',0)*512,cost_usd=0")
        (R/'tb_agent.py').write_text(agent)
        for name in ['hot_memory.py','stream_session.py','band_reader.py','layout.py','gpu_experiment.py']:shutil.copy2(H/name,R/name)
        shutil.copy2(H/'tb_worker.py',R/'worker.py');shutil.copytree(H/'vendor',R/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
        plan=dict(P,tasks=[task],arms=[arm],hot_qualification_root=str(H),decode_retrieval_interval=512,scope='Real Terminal-Bench hot-buffer pilot, original verifiers, independent arm allocations')
        (R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        for n in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(R/n).mkdir()
        for n in ['pairs','mailbox']:(R/n/task).mkdir()
        for p in R.rglob('*.py'):ast.parse(p.read_text())
        env=os.environ.copy();env.update(PYTHONPATH=str(R),PYTHONDONTWRITEBYTECODE='1',LITELLM_LOCAL_MODEL_COST_MAP='True',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        check=subprocess.run([P['harbor_python'],'-B','-c','import controller,tb_agent,apptainer_environment'],cwd=R,env=env,capture_output=True,text=True);assert check.returncode==0,check.stderr
        manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*') if p.is_file() and p.suffix in ['.py','.json']}
        (R/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        jobs.append(dict(task=task,arm=arm,root=str(R)))
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True);assert 'hotbuf-tb-' not in queue
(T/'ownership_before_submit.json').write_text(json.dumps(dict(queue=queue,epoch=time.time()),indent=2)+'\n')
for row in jobs:
    R=Path(row['root']);cmd=['sbatch','--parsable','--job-name=hotbuf-tb-'+row['arm']+'-'+row['task'][:12],
        '--partition=gpu','--time=UNLIMITED','--no-requeue','--cpus-per-task=8','--mem=64G','--gres=gpu:nvidia_l20d:1',
        '--exclude=gpu-node1,gpu-node2,gpu-node5,gpu-node6,gpu-node7','--chdir='+str(R),
        '--output='+str(R/'logs/slurm-%j.out'),'--error='+str(R/'logs/slurm-%j.err'),
        '--wrap',P['engine_python']+' -B '+str(R/'integrated_launch.py')]
    res=subprocess.run(cmd,capture_output=True,text=True);assert res.returncode==0,res.stderr
    row.update(job=res.stdout.strip().split(';')[0],argv=cmd)
    (T/'submissions.json').write_text(json.dumps(jobs,indent=2)+'\n')
(T/'PROTOCOL.md').write_text('Two preselected tasks cover short and 12-chunk histories. Three variants use fresh task environments and original verifiers: cold36, hot36_c24 with online promotion, hot24_c24 with online promotion. All refresh retrieval every 512 generated tokens. Same model/LoRA, greedy non-thinking, one GPU per independent arm. GPU model is matched, device UUID may differ. This is a six-trial pilot, not a full benchmark score; earlier failed tasks were not selected for success. No artificial time/token cap; physical memory/context bounds remain explicit.\n')
print(json.dumps(jobs,indent=2))
