from pathlib import Path
import argparse,os,subprocess
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
for proc in ([0] if a.smoke else [1,2,3]):
    dest=HERE/('short_smoke' if a.smoke else 'short_results')/f'process_{proc:02d}'
    if (dest/'complete.json').exists(): continue
    assert not dest.exists(),'Inspect incomplete output before retrying: '+str(dest)
    cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(HERE/'bench_short.py'),
         '--model','/srv/encbank/legacy_workspace/models/Qwen3-8B',
         '--adapter',str(ROOT/'exp/comem_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final'),
         '--process',str(proc)]
    if a.smoke: cmd.append('--smoke')
    with (HERE/f'short_process_{proc:02d}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if result.returncode: raise SystemExit(result.returncode)
