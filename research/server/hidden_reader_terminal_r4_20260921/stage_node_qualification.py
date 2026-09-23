import json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;Q=R/'bandtb_node4_qualification_r2_20260921';Q.mkdir()
names=['common.py','run_trial.py','tb_agent.py','controller.py','launch.py','server_transport.py','host_admission.py','apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','harbor_unbounded.py','plan.json','environment_template.json','task_manifest.json']
for n in names:shutil.copy2(R/n,Q/n)
for n in ['jobs','results','qualification','mailbox','logs','tmp','cache','pairs','configs']:(Q/n).mkdir()
for t in json.loads((Q/'plan.json').read_text())['tasks']:(Q/'pairs'/t).mkdir();(Q/'mailbox'/t).mkdir()
(Q/'source_manifest.json').write_text(json.dumps({n:hashlib.sha256((Q/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
print(str(Q))
