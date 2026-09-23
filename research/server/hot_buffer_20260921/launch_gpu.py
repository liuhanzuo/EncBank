import os,subprocess,time,sys
from common import ROOT,PLAN as P,save,verify_sources
verify_sources();env=os.environ.copy()
env.update(PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor')+':/srv/encbank/comem_infra_recheck_20260912/deps',
    PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'cache'),
    CPATH='/srv/encbank/.cache/python/include/python3.12',TORCHINDUCTOR_CACHE_DIR=str(ROOT/'compile_cache'))
start=time.time()
with (ROOT/'worker.stdout.log').open('wb') as out,(ROOT/'worker.stderr.log').open('wb') as err:
    child=subprocess.Popen([P['engine_python'],'-B',str(ROOT/'gpu_experiment.py')],cwd=ROOT,env=env,stdout=out,stderr=err)
    save(ROOT/'child.json',dict(pid=child.pid,job=os.environ.get('SLURM_JOB_ID'),start_epoch=start))
    code=child.wait()
save(ROOT/'parent_exit.json',dict(exit_code=code,actual_parent_wait=True,pid=child.pid,job=os.environ.get('SLURM_JOB_ID'),start_epoch=start,end_epoch=time.time()))
sys.exit(code)
