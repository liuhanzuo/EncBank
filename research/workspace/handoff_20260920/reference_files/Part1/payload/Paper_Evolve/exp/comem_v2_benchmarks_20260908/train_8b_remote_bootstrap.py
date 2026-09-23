"""One-shot GPU-0 launcher: wait for model transfer, measure one step, resume 4000.

This does not discover or borrow other GPUs. It may be launched with nohup; each
phase writes a separate log and the durable status file identifies failures.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
WORK = ROOT / 'workspace'
MODEL = ROOT / 'models/Qwen3-8B'
OUT = ROOT / 'outputs/8b_j12_pub_4k'
LOGS = ROOT / 'logs'
SIZES = {
    'config.json': 728,
    'generation_config.json': 239,
    'merges.txt': 1671853,
    'model-00001-of-00005.safetensors': 3996250744,
    'model-00002-of-00005.safetensors': 3993160032,
    'model-00003-of-00005.safetensors': 3959604768,
    'model-00004-of-00005.safetensors': 3187841392,
    'model-00005-of-00005.safetensors': 1244659840,
    'model.safetensors.index.json': 32878,
    'tokenizer_config.json': 9732,
    'tokenizer.json': 11422654,
    'vocab.json': 2776833,
}


def state(phase, **extra):
    row = {'phase': phase, 'pid': os.getpid(), 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'gpu_physical': 0, 'output': str(OUT), **extra}
    dest = LOGS / 'train_8b_bootstrap_status.json'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2), encoding='utf-8')
    tmp.replace(dest)
    print(json.dumps(row), flush=True)


def run_phase(name, extra_args):
    args = [sys.executable, '-u', str(WORK / 'exp/comem_v2_benchmarks_20260908/train_8b_baseline.py'),
            '--model', str(MODEL), '--data', str(WORK / 'exp/data/pg19_train_64.jsonl'),
            '--out', str(OUT), '--devices', 'cuda:0', '--steps', '4000',
            '--loss', 'published', '--adapter-dtype', 'float32', *extra_args]
    path = LOGS / f'train_8b_{name}.log'
    state(name, command=args, log=str(path))
    with path.open('a', encoding='utf-8') as log:
        rc = subprocess.call(args, cwd=WORK, env={**os.environ,
            'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}, stdout=log, stderr=subprocess.STDOUT)
    if rc:
        state('failed', failed_phase=name, returncode=rc, log=str(path))
        raise SystemExit(rc)


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / 'train_8b_bootstrap.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another training bootstrap owns the lock.')
    deadline = time.monotonic() + 24 * 3600
    while True:
        missing = {name: {'expected': size, 'actual': (MODEL / name).stat().st_size if (MODEL / name).exists() else 0}
                   for name, size in SIZES.items() if not (MODEL / name).exists() or (MODEL / name).stat().st_size != size}
        if not missing:
            break
        state('waiting_for_model', incomplete_files=missing)
        if time.monotonic() > deadline:
            state('failed', reason='model transfer deadline')
            raise SystemExit(2)
        time.sleep(30)
    # Read every safetensors header and all shape metadata before the GPU run.
    from safetensors import safe_open
    index = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(index.values())):
        expected = {key for key, value in index.items() if value == shard}
        with safe_open(str(MODEL / shard), framework='pt', device='cpu') as handle:
            if set(handle.keys()) != expected:
                raise ValueError(f'Shard/index key mismatch: {shard}')
            for key in expected:
                handle.get_slice(key).get_shape()
    marker = MODEL / 'TRANSFER_COMPLETE.json'
    marker_tmp = marker.with_suffix('.tmp')
    marker_tmp.write_text(json.dumps({'files': SIZES, 'safetensors_header_keys_match_index': True,
        'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=2), encoding='utf-8')
    marker_tmp.replace(marker)
    # Never start on GPU 0 while another compute process occupies it.
    occupants = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupants:
        state('failed', reason='GPU 0 acquired by another compute process', occupants=occupants)
        raise SystemExit(3)
    checkpoint = OUT / 'last.pt'
    if not checkpoint.exists():
        run_phase('first_step', ['--stop-after', '1'])
    saved = json.loads((OUT / 'status.json').read_text())
    if saved.get('complete') and saved.get('step') == 4000:
        state('complete', training_status=saved)
        return
    run_phase('full_4000', ['--resume', str(checkpoint)])
    saved = json.loads((OUT / 'status.json').read_text())
    if saved.get('step') != 4000 or not saved.get('complete'):
        state('failed', reason='trainer returned without 4000 completed steps', training_status=saved)
        raise SystemExit(4)
    state('complete', training_status=saved)


if __name__ == '__main__':
    main()
