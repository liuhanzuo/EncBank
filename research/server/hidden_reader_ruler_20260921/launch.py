"""Server-owned parent process, bounded environment, actual child exit receipt."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
arm=sys.argv[1]
assert arm in json.loads((root/'configs.json').read_text())
assert Path('/srv/encbank') in root.resolve().parents
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
    PYTHONPATH=str(root/'vendor')+':/srv/encbank/encbank_infra_recheck_20260912/deps',
    CPATH='/srv/encbank/.cache/python/include/python3.12')
for key,name in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),
                 ('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg'),('HF_HOME','hf')]:
    p=root/arm/'cache'/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
out=root/arm/'results';out.mkdir(exist_ok=True)
def dump(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
dump('owner.json',dict(pid=os.getpid(),job=os.environ['SLURM_JOB_ID'],root=str(root),
    at=datetime.datetime.now().astimezone().isoformat(),role='paired RULER hidden Reader evaluation',arm=arm))
with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
    child=subprocess.Popen(['/srv/encbank/Paper_Evolve/.venv/bin/python','-B','-u',str(root/'experiment.py'),arm],
        cwd=root,env=env,stdout=stdout,stderr=stderr)
    dump('child.json',dict(pid=child.pid,job=os.environ['SLURM_JOB_ID']))
    result=child.wait()
dump('parent_exit.json',dict(actual_wait=True,returncode=result,child_pid=child.pid,
    job=os.environ['SLURM_JOB_ID'],at=datetime.datetime.now().astimezone().isoformat()))
sys.exit(result)
