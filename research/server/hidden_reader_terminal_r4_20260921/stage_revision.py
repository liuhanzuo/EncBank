"""New immutable revision; retain every failed predecessor record."""
import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=R.with_name('hidden_reader_terminal_20260921')
assert not (R/'plan.json').exists()
for p in OLD.iterdir():
    if p.is_file() and p.suffix in ['.py','.md'] and not (R/p.name).exists():shutil.copy2(p,R/p.name)
shutil.copytree(OLD/'vendor',R/'vendor',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['plan.json','task_manifest.json','environment_template.json','cpu_check.json']:shutil.copy2(OLD/name,R/name)
plan=json.loads((R/'plan.json').read_text());plan['predecessor']=str(OLD);plan['revision_reason']='Missing PEFT import path caused six pre-model exits; zero benchmark model calls. Reuse existing qualified backend, repair only engine dependency environment.'
(R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
names=['apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','harbor_unbounded.py','run_trial.py','tb_agent.py','host_admission.py']
reuse=dict(origin=str(OLD),qualification_job='116014',backend_hashes={n:hashlib.sha256((OLD/n).read_bytes()).hexdigest() for n in names})
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==v for n,v in reuse['backend_hashes'].items())
assert not list((OLD/'mailbox').glob('**/*.request.json'))
assert all(json.loads((OLD/'pairs'/t/'parent_exit.json').read_text())['exit_code']!=0 for t in plan['tasks'])
(R/'qualification_reuse.json').write_text(json.dumps(reuse,indent=2)+'\n')
for name in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(R/name).mkdir()
for task in plan['tasks']:(R/'pairs'/task).mkdir();(R/'mailbox'/task).mkdir()
print(json.dumps(dict(staged=str(R),qualification_reused_from=str(OLD),previous_model_calls=0)))
