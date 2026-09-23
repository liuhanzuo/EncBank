"""Node-local output and actual scientific-child exit record; one Slurm GPU."""
import datetime,json,os,shutil,subprocess,tarfile
from pathlib import Path
job=os.environ['SLURM_JOB_ID'];assert os.getuid()==20021
control=Path('/tmp/qcm-deploy-control-liuhanzuo-20260919-formal');root=Path('/tmp/qcm-deploy-liuhanzuo-'+job)
assert shutil.disk_usage('/tmp').free>2*2**30
root.mkdir(mode=0o700)
with tarfile.open(control/'payload.tar') as tar:
    for m in tar.getmembers():
        assert not m.name.startswith('/') and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
    tar.extractall(str(root))
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',CPATH='/srv/encbank/.cache/python/include/python3.12')
env['PYTHONPATH']=str(root)+':/srv/encbank/encbank_infra_recheck_20260912/deps'
for k,n in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
    p=root/n;p.mkdir();env[k]=str(p)
def dump(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
dump('launch.json',dict(job=job,root=str(root),at=datetime.datetime.utcnow().isoformat()+'Z'))
with (root/'stdout.log').open('ab') as o,(root/'stderr.log').open('ab') as e:
    child=subprocess.Popen(['/srv/encbank/Paper_Evolve/.venv/bin/python','-B','-u',str(root/'benchmark.py')],cwd=str(root),env=env,stdout=o,stderr=e)
    dump('child.json',dict(pid=child.pid));code=child.wait()
dump('parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=child.pid,job=job,at=datetime.datetime.utcnow().isoformat()+'Z'))
raise SystemExit(code)
