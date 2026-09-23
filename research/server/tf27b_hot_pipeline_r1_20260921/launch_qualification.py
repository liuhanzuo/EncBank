import json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,PLAN as P,save,verify_sources
verify_sources();env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    CPATH='/srv/encbank/.cache/python/include/python3.12',TOKENIZERS_PARALLELISM='false',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'cache'),
    TORCHINDUCTOR_CACHE_DIR=str(ROOT/'compile_cache'))
start=time.time();p=subprocess.Popen([P['engine_python'],'-B',str(ROOT/'qualify_gpu.py')],cwd=ROOT,env=env)
save(ROOT/'child.json',dict(pid=p.pid,job=os.environ['SLURM_JOB_ID'],start_epoch=start))
code=p.wait();save(ROOT/'parent_exit.json',dict(exit_code=code,actual_parent_wait=True,pid=p.pid,
    job=os.environ['SLURM_JOB_ID'],start_epoch=start,end_epoch=time.time()));sys.exit(code)
