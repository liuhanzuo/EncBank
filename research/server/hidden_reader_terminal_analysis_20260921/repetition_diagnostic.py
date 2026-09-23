"""Read-only diagnostics; never changes, stops or scores live trials."""
import json
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.with_name('hidden_reader_terminal_r5_20260921');rows=[]
for task,root in json.loads((R/'roots.json').read_text()).items():
    C=Path(root)
    for arm in ['native','n24']:
        observed=[]
        for p in sorted((C/'mailbox'/task).glob(arm+'_*.response.json')):
            r=json.loads(p.read_text())
            try:
                answer=json.loads(r['text']);commands=answer.get('commands',[])
                if not commands:continue
                key=json.dumps(commands,sort_keys=True);observed.append((r['step'],key,commands,answer.get('task_complete')))
            except (ValueError,KeyError):pass
        if not observed:continue
        tail=[]
        for row in reversed(observed):
            if row[1]!=observed[-1][1]:break
            tail.append(row[0])
        if len(tail)>=3:rows.append(dict(task=task,arm=arm,consecutive_identical_command_responses=len(tail),
            steps=sorted(tail),commands=observed[-1][2],task_complete=observed[-1][3],action_taken='none; read-only observation'))
(A/'repetition_diagnostic.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n');print(json.dumps(rows))
