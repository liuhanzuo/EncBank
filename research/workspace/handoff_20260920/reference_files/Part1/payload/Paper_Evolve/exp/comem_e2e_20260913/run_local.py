"""Run three independent processes sequentially through the existing GPU gate."""
from pathlib import Path
import argparse, os, subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
for process in ([0] if a.smoke else [1,2,3]):
    dest=HERE/('smoke' if a.smoke else 'results')/f'process_{process:02d}'
    if (dest/'complete.json').exists(): continue
    assert not dest.exists(), 'Inspect the incomplete process directory before rerunning: '+str(dest)
    cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-u',str(HERE/'bench_e2e.py'),
        '--model','/srv/encbank/legacy_workspace/models/Qwen3-8B',
        '--adapter',str(ROOT/'exp/comem_v2_benchmarks_20260908/checkpoints/8b_j12_pub_4k/final'),
        '--process',str(process)]
    if a.smoke: cmd.append('--smoke')
    with (HERE/f'process_{process:02d}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
             env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    if result.returncode: raise SystemExit(result.returncode)
