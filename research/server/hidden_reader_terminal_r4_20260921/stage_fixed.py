import json,shutil,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.with_name('hidden_reader_terminal_r3_20260921');R2=R.with_name('hidden_reader_terminal_r2_20260921')
Q=R2/'bandtb_node4_qualification_r2_20260921'
assert not (R/'plan.json').exists()
assert json.loads((Q/'jobs'/'qualification_complete.json').read_text())['passed']
assert json.loads((Q/'jobs'/'qualify_parent_exit.json').read_text())['exit_code']==0
oldjobs=json.loads((OLD/'jobs'/'submissions_run.json').read_text())
live=subprocess.check_output(['squeue','-j',','.join(j['job'] for j in oldjobs),'-h','-o','%i|%T'],text=True)
assert not live.strip(),live
for p in OLD.iterdir():
    if p.is_file() and p.suffix in ['.py','.md'] and not (R/p.name).exists():shutil.copy2(p,R/p.name)
shutil.copytree(OLD/'vendor',R/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
for n in ['environment_template.json','task_manifest.json','cpu_check.json','dependency_check.json']:shutil.copy2(OLD/n,R/n)
p=json.loads((R2/'plan.json').read_text());p['global_task_order']=p['tasks'];p['tasks']=[t for t in p['tasks'] if t!='cancel-async-tasks']
p['resource_inventory']=[r for r in p['resource_inventory'] if r['task'] in p['tasks']]
p['predecessors']=[str(R2),str(OLD)];p['revision_reason']='Cache audit fix only: cloned H12 bank tensors now have version counters outside inference mode, identical values. Added three-turn GPU qualification for native and n24. Retain the already valid cancel-async-tasks pair; retry only infrastructure-invalid/incomplete remaining tasks.'
for arm in ['native','n24']:
    receipt=json.loads((R2/'results'/('cancel-async-tasks--'+arm)/'execution_receipt.json').read_text())
    assert not receipt['exception'] and receipt['verifier'] is not None and receipt['closure']['cgroup_empty']
(R/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
names=['apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','harbor_unbounded.py','run_trial.py','tb_agent.py','host_admission.py']
reuse=dict(origin=str(Q),qualification_job='116157',backend_hashes={n:hashlib.sha256((Q/n).read_bytes()).hexdigest() for n in names})
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==v for n,v in reuse['backend_hashes'].items())
(R/'qualification_reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
for n in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(R/n).mkdir()
for task in p['tasks']:(R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
print(json.dumps(dict(tasks=p['tasks'],retained_pair='cancel-async-tasks',environment_qualification=str(Q))))
