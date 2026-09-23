import json,time
from pathlib import Path
R=Path('/srv/encbank/qcomem_align_codex_20260911/hidden_reader_terminal_r5_20260921')
for task,root in json.loads((R/'roots.json').read_text()).items():
    C=Path(root);rs=sorted((C/'mailbox'/task).glob('*.response.json'),key=lambda p:p.stat().st_mtime)
    if not rs:continue
    p=rs[-1];r=json.loads(p.read_text());q=json.loads(p.with_name(p.name.replace('.response.','.request.')).read_text())
    print(json.dumps(dict(task=task,last=p.name,age_seconds=time.time()-p.stat().st_mtime,
        latest_observation=q['messages'][-1]['content'][-1200:],model_text=r['text'][-2000:])))
