"""Serial, model-free checks of frozen task images on the qualified server node."""
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

R=Path('/srv/encbank/qencbank_runtime_20260911/server_control_20260920')


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=R/('matrix-'+os.environ['SLURM_JOB_ID'])
    root.mkdir(exist_ok=False)
    rows=[]
    tasks=sorted(p.name for p in (R/'tasks').iterdir() if (p/'task.toml').exists())
    for task in tasks:
        env=tomllib.loads((R/'tasks'/task/'task.toml').read_text())['environment']
        assert env.get('memory_mb',2048)<=16384
        output=root/task
        with (root/(task+'.log')).open('w') as log:
            child=subprocess.run([str(R/'harbor_env/bin/python'),str(R/'container_probe.py'),
                                  '--managed','--task',task,'--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
        report=output/'report.json'
        d=json.loads(report.read_text()) if report.exists() else {}
        rows.append(dict(task=task,exit_code=child.returncode,actual_parent_wait=True,
                         status=d.get('status','NO_REPORT'),cleanup_verified=d.get('cleanup_verified',False),
                         report=str(report),error=d.get('error')))
        snapshot=dict(state='running',rows=rows,pending=tasks[len(rows):],model_calls=0,benchmark_attempts=0,
                      epoch=time.time(),server_only=True)
        (root/'status.json').write_text(json.dumps(snapshot,indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)
        if d and not d.get('cleanup_verified'):
            raise RuntimeError('Cleanup is not proven; stop before creating another container')
    snapshot.update(state='completed',passed=sum(r['status']=='BASIC_PASS' for r in rows))
    (root/'status.json').write_text(json.dumps(snapshot,indent=2)+'\n')


if __name__=='__main__':main()
