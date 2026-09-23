"""Verify original TB assets and the explicitly authorized agent-timeout removal."""
import hashlib,json,tomllib
from pathlib import Path

ORIGINAL=Path('/srv/encbank/Encbank_Migration_20260920/Part2/payload/qencbank/.runtime/terminal_bench_full89_20260919/tasks')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_tasks(task_root,manifest,tasks):
    root=Path(task_root);files={};changes=[]
    for entry in manifest['files']:
        parts=Path(entry['path']).parts
        if len(parts)<3 or parts[0]!='tasks' or parts[1] not in tasks:continue
        relative=Path(*parts[1:]);current=root/relative
        actual=sha(current);expected=entry['sha256']
        if actual!=expected:
            assert relative.name=='task.toml',('Unexpected asset modification',str(relative))
            original=ORIGINAL/relative
            assert sha(original)==expected,('Original source identity mismatch',str(relative))
            before=tomllib.loads(original.read_text());after=tomllib.loads(current.read_text())
            timeout=before.get('agent',{}).pop('timeout_sec',None)
            assert timeout is not None and before==after,('Changes exceed agent timeout removal',str(relative))
            changes.append(dict(path=str(relative),original_sha256=expected,runtime_sha256=actual,
                removed_agent_timeout_sec=timeout,authorization='2026-09-21 user requests removal of task/request time limits'))
        files[str(current)]=actual
    assert files and {Path(p).relative_to(root).parts[0] for p in files}==set(tasks)
    return files,dict(passed=True,files=len(files),task_count=len(tasks),authorized_changes=changes)

if __name__=='__main__':
    from task_selection import select
    S=Path(__file__).resolve().parent
    old=S.parent/'terminal_bench_full89_20260919/server_control_20260920/k12_unbounded_20260921'
    plan=json.loads((old/'plan.json').read_text());manifest=json.loads((old/'task_manifest.json').read_text())
    tasks,_=select(plan['task_root'],manifest['tasks'])
    _,proof=verify_tasks(plan['task_root'],manifest,tasks)
    (S/'task_integrity_preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='authorized_changes'}|dict(timeout_removals=len(proof['authorized_changes']))))
