"""Submit one bounded diagnostic only after checking this task has no allocation."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from slurm_gpu_guard import confined
from submit_slurm_sparse import check_other_jobs


def main():
    root=confined('/srv/encbank/comem_sparse_slurm_20260912')
    code=confined(Path(__file__).parent,root)
    worker=code/'route_probe_20260913_slurm.py'
    out=root/'outputs/route_probe_20260913'
    python='/srv/encbank/Paper_Evolve/.venv/bin/python'
    with confined('/srv/encbank/.codex-comem-sparse-l20d.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_other_jobs(root,worker)
        accounting=subprocess.run(['sacct','-j','24577','--noheader','--parsable2','--format','JobID,State,ExitCode'],
            text=True,capture_output=True,timeout=30,check=True)
        if '24577|COMPLETED|0:0' not in accounting.stdout.splitlines():
            raise RuntimeError('Prior training completion is not confirmed')
        if out.exists() or (root/'logs/route_probe_20260913_submission.json').exists():
            raise RuntimeError('Diagnostic was already submitted or has artifacts; inspect before retry')
        command=['sbatch','--parsable','--job-name','comem-route-probe','--partition','gpu',
            '--gres','gpu:nvidia_l20d:1','--nodes','1','--ntasks','1','--cpus-per-task','4',
            '--mem','64G','--time','01:45:00','--chdir',str(root),
            '--output',str(root/'logs/route-probe-%j.out'),'--error',str(root/'logs/route-probe-%j.err'),
            '--signal','B:TERM@120']
        argv=[python,'-B','-u',str(worker),'--task-root',str(root),'--out',str(out)]
        script='#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexec '+shlex.join(argv)+'\n'
        receipt=dict(command=command,stdin_script=script,pre_submit_training_accounting=accounting.stdout,
            no_other_task_allocation=True,submitted_utc=datetime.now(timezone.utc).isoformat(),
            cpus=4,gpu_count=1,formal_inference_timing=False,output=str(out))
        result=subprocess.run(command,input=script,text=True,capture_output=True,timeout=60,check=True)
        raw=result.stdout.strip()
        if not re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',raw):
            raise RuntimeError('Ambiguous submission result; do not retry: '+raw)
        receipt.update(job_id=raw.split(';')[0],stdout=result.stdout,stderr=result.stderr)
        (root/'logs/route_probe_20260913_submission.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
