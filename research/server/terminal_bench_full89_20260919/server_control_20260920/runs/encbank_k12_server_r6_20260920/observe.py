"""Read server-side task, controller and worker status without importing model code."""
import json
import subprocess
import time
from pathlib import Path

from host_admission import identity_alive

H = Path(__file__).resolve().parent
P = json.loads((H / 'plan.json').read_text())
report = {'epoch': time.time(), 'run': H.name, 'server_only': True}
for name in ['submission.json', 'server_job_receipt.json', 'server_cpu_preflight.json', 'server_preflight.json',
             'execution/owner_registration.json', 'execution/status.json', 'execution/controller_failure.json',
             'execution/owner_complete.json', 'execution/transport_snapshot.json',
             'run_' + P['arm'] + '/worker_ready.json', 'run_' + P['arm'] + '/worker_failure.json',
             'run_' + P['arm'] + '/process_receipt.json']:
    if (H / name).exists():
        report[name] = json.loads((H / name).read_text())
owner = report.get('execution/owner_registration.json')
if owner:
    report['owner_alive_on_this_host'] = identity_alive(owner['identity']) if owner['hostname'] == __import__('socket').gethostname() else None
report['delivered_responses'] = len(list((Path(P['rpc_root']) / P['arm']).glob('*.broker.json')))
report['closed_tasks'] = len(list((H / 'execution/receipts').glob('*.json')))
report['queue'] = subprocess.check_output(['squeue', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%N'], text=True)
print(json.dumps(report, indent=2))
