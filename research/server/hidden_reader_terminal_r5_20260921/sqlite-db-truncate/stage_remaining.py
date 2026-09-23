import json,shutil,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.with_name('hidden_reader_terminal_r2_20260921')
Q=OLD/'bandtb_node4_qualification_r2_20260921'
assert not (R/'plan.json').exists()
assert json.loads((Q/'jobs'/'qualification_complete.json').read_text())['passed']
assert json.loads((Q/'jobs'/'qualify_parent_exit.json').read_text())['exit_code']==0
tasks=['log-summary-date-ranges','query-optimize','sqlite-db-truncate']
for task in tasks:
    assert json.loads((OLD/'pairs'/task/'parent_exit.json').read_text())['exit_code']==1
    assert not list((OLD/'mailbox'/task).glob('*.request.json'))
for p in OLD.iterdir():
    if p.is_file() and p.suffix in ['.py','.md'] and not (R/p.name).exists():shutil.copy2(p,R/p.name)
shutil.copytree(OLD/'vendor',R/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
for n in ['environment_template.json','task_manifest.json','cpu_check.json','dependency_check.json']:shutil.copy2(OLD/n,R/n)
p=json.loads((OLD/'plan.json').read_text());p['global_task_order']=p['tasks'];p['tasks']=tasks
p['resource_inventory']=[r for r in p['resource_inventory'] if r['task'] in tasks]
p['predecessor']=str(OLD);p['revision_reason']='Retry only three pre-model GPU-memory admission failures; zero previous requests on these tasks. Other three task pairs remain owned by revision2. Container controller uses newly qualified gpu-node4.'
(R/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
names=['apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','harbor_unbounded.py','run_trial.py','tb_agent.py','host_admission.py']
reuse=dict(origin=str(Q),qualification_job='116157',backend_hashes={n:hashlib.sha256((Q/n).read_bytes()).hexdigest() for n in names})
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==v for n,v in reuse['backend_hashes'].items())
(R/'qualification_reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
for n in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(R/n).mkdir()
for task in tasks:(R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
print(json.dumps(dict(tasks=tasks,previous_calls=0,environment_qualification=str(Q))))
