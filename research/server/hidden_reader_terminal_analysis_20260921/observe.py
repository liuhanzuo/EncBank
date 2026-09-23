"""Compact read-only observer for this research's jobs, not other controllers."""
import json,subprocess
from pathlib import Path
B=Path('/srv/encbank/qcomem_align_codex_20260911')
R=B/'hidden_reader_terminal_r4_20260921'
def load(p):return json.loads(p.read_text()) if p.exists() else None
jobs=load(R/'jobs/submissions_run.json') or []
queue=subprocess.run(['squeue','-h','-j',','.join(j['job'] for j in jobs),'-o','%i|%T|%N|%R'],capture_output=True,text=True)
print(queue.stdout.strip())
for t in load(R/'plan.json')['tasks']:
    d=R/'pairs'/t; q=load(d/'qualification.json'); s=load(d/'status.json')
    row=dict(task=t,qualification=bool(q and q.get('passed')),status=s,
        parent=load(d/'parent_exit.json'))
    fail=load(d/'worker_failure.json')
    if fail:row['failure']=fail.get('error','')[-1200:]
    row['arms']={}
    for arm in ['native','n24']:
        receipt=load(R/'results'/(t+'--'+arm)/'execution_receipt.json')
        rs=[load(p) for p in (R/'mailbox'/t).glob(arm+'*.response.json')]
        row['arms'][arm]=dict(requests=len(list((R/'mailbox'/t).glob(arm+'*.request.json'))),responses=len(rs),
            tokens=sum(r.get('generated_tokens',0) for r in rs),max_chunks=max([r.get('selected_chunks',0) for r in rs] or [0]),
            reward=receipt.get('verifier') if receipt else None,
            exception=(receipt.get('exception') or {}).get('exception_type') if receipt else None,
            closed=receipt is not None)
    print(json.dumps(row))
print(json.dumps(dict(controller=load(R/'jobs/controller_parent_exit.json'))))
