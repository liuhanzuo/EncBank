"""Launch the authorized one-shot GPU queues; never touch other GPU processes."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
HERE = ROOT / 'workspace/exp/comem_v2_benchmarks_20260908'


def launch(name, smoke, full, gpu, extra=()):
    folder = ROOT / 'outputs' / ('bootstrap_' + name)
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / 'status.json'
    if status.exists():
        previous = json.loads(status.read_text())
        if previous.get('status') == 'completed':
            print(json.dumps({'name': name, 'status': 'already completed'}), flush=True)
            return
        pid = previous.get('pid')
        if pid and Path(f'/proc/{pid}/cmdline').exists():
            command_line = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'benchmark_remote_bootstrap.py' in command_line:
                print(json.dumps({'name': name, 'pid': pid, 'status': previous['status']}), flush=True)
                return
    command = [sys.executable, '-u', str(HERE / 'benchmark_remote_bootstrap.py'),
        '--smoke-plan', str(HERE / smoke), '--full-plan', str(HERE / full),
        '--gpus', gpu, '--status-dir', str(folder), *map(str, extra)]
    with (folder / 'bootstrap.log').open('a') as log, open(os.devnull) as devnull:
        process = subprocess.Popen(command, cwd=HERE, stdin=devnull, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    print(json.dumps({'name': name, 'pid': process.pid, 'gpu': gpu, 'log': str(folder / 'bootstrap.log')}), flush=True)


if __name__ == '__main__':
    launch('main', 'smoke_plan.json', 'full_plan.json', '2,3')
    launch('trained_pub', 'trained_pub_smoke_plan.json', 'trained_pub_full_plan.json', '0',
        ['--wait-training-status', ROOT / 'outputs/8b_j12_pub_4k/status.json'])
    # CacheBlend uses stock weights and has no trained-baseline dependency.
    # GPUs 1/3 were verified free; the queue rechecks memory and compute contexts.
    launch('cacheblend16', 'cacheblend16_smoke_plan.json', 'cacheblend16_full_plan.json', '1,3')
    # User requirement, 2026-09-08: remote 3090s are for quality and training only.
    # All timing and serving-memory measurements run on the local RTX 5090.
    print(json.dumps({'name': 'serving', 'state': 'moved_to_local_rtx5090',
                      'launcher': 'serving_local_bootstrap.py'}), flush=True)
