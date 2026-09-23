"""Run local cost processes serially; each worker acquires the shared GPU gate."""
from pathlib import Path
import subprocess,sys,os,json
HERE=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
base=[sys.executable,'-u',str(HERE/'run_cost.py'),'--model','/srv/encbank/legacy_workspace/models/Qwen3-8B','--adapter',str(HERE.parent/'encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final')]
for process,smoke in ((0,True),(1,False),(2,False),(3,False)):
    args=base+['--process',str(process)]+(['--smoke'] if smoke else [])
    log=HERE/f'local_{process:02d}.log'
    with log.open('wb') as f:
        result=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps({'process':process,'smoke':smoke,'exit_code':result.returncode,'log':str(log)}),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
