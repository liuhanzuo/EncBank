import json,subprocess
from pathlib import Path
B=Path('/srv/encbank/qencbank_align_codex_20260911');R=B/'hidden_reader_terminal_r5_20260921'
def load(p):return json.loads(p.read_text()) if p.exists() else None
jobs=load(R/'submissions.json') or []
if jobs:print(subprocess.run(['squeue','-h','-j',','.join(j['job'] for j in jobs),'-o','%i|%T|%N|%R'],capture_output=True,text=True).stdout.strip())
for task,root in (load(R/'roots.json') or {}).items():
    C=Path(root);d=C/'pairs'/task;s=load(d/'status.json');q=load(d/'qualification.json')
    row=dict(task=task,model_qualification=bool(q and q.get('passed')),status=s,complete=load(C/'jobs/integrated_complete.json'))
    for n in ['integrated_failure.json','worker_failure.json']:
        f=load(d/n)
        if f:row[n]=f.get('error','')[-800:]
    row['arms']={}
    for arm in ['native','n24']:
        receipt=load(C/'results'/(task+'--'+arm)/'execution_receipt.json')
        rs=[load(p) for p in (C/'mailbox'/task).glob(arm+'*.response.json')]
        row['arms'][arm]=dict(requests=len(list((C/'mailbox'/task).glob(arm+'*.request.json'))),responses=len(rs),
            tokens=sum(r.get('generated_tokens',0) for r in rs),max_chunks=max([r.get('selected_chunks',0) for r in rs] or [0]),
            reward=receipt.get('verifier') if receipt else None,closed=receipt is not None)
    print(json.dumps(row))
