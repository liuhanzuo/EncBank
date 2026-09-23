import os,subprocess,sys
from pathlib import Path
root=Path('F:/Paper_Evolve')
out=root/'exp/encbank_infra_128k_20260912'
for process in (1,2,3):
    dest=out/'results'/f'process_{process:02d}'
    if (dest/'complete.json').exists():
        continue
    if (dest/'records.jsonl').exists():
        raise RuntimeError('Refusing to append to interrupted process '+str(dest))
    cmd=[str(root/'.venv/Scripts/python.exe'),'-u',str(root/'exp/encbank_infra_20260912/bench_local.py'),'--model','/srv/encbank/legacy_workspace/models/Qwen3-8B','--adapter',str(root/'exp/encbank_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final'),'--source','/srv/encbank/legacy_workspace/data/pg19_train_64.jsonl','--out',str(dest),'--process',str(process),'--source-length','131072']
    with (out/f'process_{process:02d}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if result.returncode:
        raise SystemExit(result.returncode)
