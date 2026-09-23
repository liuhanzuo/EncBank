import ast,hashlib,json,shutil,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.with_name('hidden_reader_terminal_r4_20260921')
P=json.loads((R/'plan.json').read_text());controls=A/'controls';assert not controls.exists()
state=subprocess.check_output(['squeue','-j','116227','-h','-o','%T'],text=True).strip()
assert state=='PENDING',state
assert not list((R/'mailbox').glob('*/*.request.json'))
subprocess.run(['scancel','116227'],check=True)
names=['common.py','controller.py','run_trial.py','tb_agent.py','apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','host_admission.py','harbor_unbounded.py','environment_template.json']
for task in P['tasks']:
    C=controls/task;C.mkdir(parents=True)
    for name in names:shutil.copy2(R/name,C/name)
    shutil.copy2(A/'controller_child.py',C/'controller_child.py')
    cp=dict(P,tasks=[task],worker_root=str(R));(C/'plan.json').write_text(json.dumps(cp,indent=2)+'\n')
    for n in ['jobs','results','mailbox','pairs','logs','configs','qualification']:(C/n).symlink_to(R/n,target_is_directory=True)
    for n in ['tmp','cache']:(C/n).mkdir()
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.iterdir() if p.is_file()}
    for p in C.glob('*.py'):ast.parse(p.read_text())
    (C/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
receipt=dict(cancelled_pending_controller='116227',no_trials_started_before_switch=True,
    reason='Standalone unlimited CPU controller cannot backfill; controllers now run in disjoint CPUs of own GPU allocations.',
    controllers=[str(controls/t) for t in P['tasks']],worker_sources_unchanged=True,
    node_environment_qualification='Each actual host/task is requalified install-only before model calls.')
(A/'controller_topology.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
