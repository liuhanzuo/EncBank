import json,subprocess
from common import ROOT,PLAN
def read(path):return json.loads(path.read_text()) if path.exists() else None
rows=[]
for task in PLAN['tasks']:
    row=dict(task=task,status=read(ROOT/'pairs'/task/'status.json'),worker_exit=read(ROOT/'pairs'/task/'parent_exit.json'),worker_failure=read(ROOT/'pairs'/task/'worker_failure.json'),arms={})
    for arm in ['qualify','native','n24']:
        directory=ROOT/('qualification' if arm=='qualify' else 'results')/(task+'--'+arm)
        row['arms'][arm]=dict(receipt=read(directory/'execution_receipt.json'),wait=read(ROOT/'jobs'/('wait-'+task+'--'+arm+'.json')),calls=len(list((ROOT/'mailbox'/task).glob(arm+'_*.response.json'))))
    rows.append(row)
print(json.dumps(dict(rows=rows,qualification=read(ROOT/'jobs'/'qualification_complete.json'),controller=read(ROOT/'jobs'/'controller_complete.json'),controller_failure=read(ROOT/'jobs'/'controller_failure.json'))))
print(subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%M|%N'],text=True))
