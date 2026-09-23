"""Slurm parent owns one worker/controller process and records its real wait."""
import os,subprocess,sys,time
from common import ROOT,PLAN as P,save,verify_sources
kind=sys.argv[1];task=sys.argv[2] if len(sys.argv)>2 else None
verify_sources();env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor'))
env.setdefault('XDG_RUNTIME_DIR','/run/user/'+str(os.getuid()));env.setdefault('DBUS_SESSION_BUS_ADDRESS','unix:path='+env['XDG_RUNTIME_DIR']+'/bus')
if kind=='worker':
    env.update(PYTHONPATH=env['PYTHONPATH']+':/srv/encbank/encbank_infra_recheck_20260912/deps',
        PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        CPATH='/srv/encbank/.cache/python/include/python3.12')
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'cache'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'pairs'/task/'compile_cache'))
    command=[P['engine_python'],str(ROOT/'worker.py'),task];directory=ROOT/'pairs'/task
else:
    command=[P['harbor_python'],str(ROOT/'controller.py')]+(['--qualify'] if kind=='qualify' else [])
    directory=ROOT/'jobs';env.pop('HF_HUB_OFFLINE',None);env.pop('TRANSFORMERS_OFFLINE',None)
start=time.time()
with (directory/(kind+'.stdout.log')).open('wb') as out,(directory/(kind+'.stderr.log')).open('wb') as err:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err);code=child.wait()
save(directory/('parent_exit.json' if kind=='worker' else kind+'_parent_exit.json'),dict(exit_code=code,actual_parent_wait=True,job=os.environ.get('SLURM_JOB_ID'),pid=child.pid,start_epoch=start,end_epoch=time.time()))
sys.exit(code)
