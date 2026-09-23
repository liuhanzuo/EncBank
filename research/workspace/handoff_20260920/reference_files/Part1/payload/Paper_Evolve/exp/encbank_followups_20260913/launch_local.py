"""Three sequential fresh processes; each worker uses the shared GPU lock."""
from pathlib import Path
import json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
for number in range(4):
    complete=HERE/('kv_cost_smoke' if number==0 else 'kv_cost')/f'process_{number:02d}'/'complete.json'
    if complete.exists():continue
    args=[sys.executable,'-u',str(HERE/'run_kv_cost.py'),'--model','/srv/encbank/legacy_workspace/models/Qwen3-8B','--adapter',str(HERE.parent/'encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final'),'--process',str(number)]
    if number==0:args.append('--smoke')
    with (HERE/f'local_{number:02d}.log').open('wb') as log:r=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'process':number,'exit':r.returncode}),flush=True)
    if r.returncode:raise SystemExit(r.returncode)
