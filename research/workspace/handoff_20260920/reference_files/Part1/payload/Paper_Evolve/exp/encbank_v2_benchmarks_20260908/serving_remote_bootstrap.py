"""Retired remote timing entry point: serving measurements require local RTX 5090."""
from pathlib import Path
import json
import os
import time

ROOT = Path('/data/liuhanzuo/encbank_v2_20260908')


def main():
    folder = ROOT / 'outputs/bootstrap_serving'
    folder.mkdir(parents=True, exist_ok=True)
    state = {'status': 'moved_to_local_rtx5090', 'pid': os.getpid(),
             'updated_at': time.time(), 'required_gpu': 'NVIDIA GeForce RTX 5090',
             'reason': 'User requires one local GPU for all comparable timing and serving-memory results.',
             'launcher': 'serving_local_bootstrap.py', 'remote_timing_allowed': False}
    temp = folder / 'status.tmp'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(folder / 'status.json')
    print(json.dumps(state), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
