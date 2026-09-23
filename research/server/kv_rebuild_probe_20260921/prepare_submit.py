"""Run with system Python on gpu-node1; immutable staging and singleton submission."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parent
home=Path('/srv/encbank').resolve()
assert home in root.resolve().parents
assert not (root/'submission.json').exists(), 'Inspect existing submission; never blindly resubmit'
archive=home/'COMem_Migration_20260920/Part1/payload/Paper_Evolve/exp/comem_deployment_b300_20260919/payload.tar'
vendor=root/'vendor';vendor.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        name=member.name
        allowed=(name in ['engine.py','bm25_index.py','prefix_store.py','workloads.json'] or
            (name.startswith('comem/') and name.endswith('.py')))
        if not allowed:continue
        assert member.isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
        dest=vendor/name;dest.parent.mkdir(parents=True,exist_ok=True)
        payload=tar.extractfile(member).read()
        if dest.exists():assert dest.read_bytes()==payload
        else:dest.write_bytes(payload)
manifest={}
for f in sorted(root.rglob('*')):
    if f.is_file():
        if f.suffix=='.py':compile(f.read_text(),str(f),'exec')
        manifest[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with (root.parent/'kv_rebuild_submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%N'],text=True)
    assert 'qcm-kvrebuild-8b' not in queue
    own=[line for line in queue.splitlines() if any(x in line for x in ['qcomem-tb-','qcm-kvrebuild-'])]
    assert len(own)<4, own
    command=['sbatch','--parsable','--job-name=qcm-kvrebuild-8b','--dependency=singleton',
        '--partition=gpu','--nodelist=gpu-node1','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1',
        '--cpus-per-task=8','--mem=64G','--time=01:00:00','--chdir='+str(root),
        '--output='+str(root/'slurm-%j.out'),'--error='+str(root/'slurm-%j.err')]
    script='#!/bin/bash\nset -euo pipefail\nexec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u '+str(root/'launch.py')+'\n'
    (root/'job.sh').write_text(script)
    (root/'submission_intent.json').write_text(json.dumps(dict(command=command,queue=queue,
        at=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
    p=subprocess.run(command,input=script,text=True,capture_output=True,timeout=45)
    record=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        at=datetime.datetime.now().astimezone().isoformat())
    if p.returncode==0:record['job']=p.stdout.strip().split(';')[0]
    (root/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    assert p.returncode==0,p.stderr
