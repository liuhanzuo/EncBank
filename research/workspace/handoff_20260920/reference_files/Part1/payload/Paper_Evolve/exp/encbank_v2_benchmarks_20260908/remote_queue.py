"""Run the explicit Encbank experiment plan on otherwise idle allocated GPUs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def gpu_idle(index, max_mib):
    result = subprocess.run(['nvidia-smi', '-i', str(index),
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    memory, util = [int(x.strip()) for x in result.stdout.strip().split(',')]
    contexts = subprocess.run(['nvidia-smi', '-i', str(index),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    compute_pids = [int(x.strip()) for x in contexts.stdout.splitlines() if x.strip()]
    return memory <= max_mib and util <= 5 and not compute_pids, {
        'memory_mib': memory, 'utilization': util, 'compute_pids': compute_pids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--gpus', default='0,2')
    parser.add_argument('--max-idle-mib', type=int, default=512)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan['state_dir'])
    out.mkdir(parents=True, exist_ok=True)
    guard = (out / 'queue.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = out / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}}
    for job in plan['jobs']:
        previous = state['jobs'].get(job['id'], {})
        if previous.get('status') == 'running':
            pid = previous['pid']
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                previous['status'] = 'interrupted'
            else:
                raise RuntimeError(f'Recorded child {pid} is still running; do not duplicate it')
        state['jobs'].setdefault(job['id'], {'status': 'pending', 'attempts': 0})
    running = {}
    gpu_indices = [int(x) for x in args.gpus.split(',')]
    state.update(queue_pid=os.getpid(), gpus=gpu_indices, plan=str(args.plan))
    atomic_json(state_path, state)
    while True:
        for gpu, active in list(running.items()):
            process, log, job_id = active
            code = process.poll()
            if code is None:
                continue
            log.close()
            state['jobs'][job_id].update(status='completed' if code == 0 else 'failed',
                                         exit_code=code, finished_at=time.time())
            print(f'FINISHED {job_id} gpu={gpu} exit={code}', flush=True)
            del running[gpu]
        for gpu in gpu_indices:
            if gpu in running:
                continue
            pending = [j for j in plan['jobs']
                if state['jobs'][j['id']]['status'] in {'pending', 'interrupted'}
                and all(state['jobs'].get(dep, {}).get('status') == 'completed'
                        for dep in j.get('depends_on', []))]
            if not pending:
                continue
            idle, reading = gpu_idle(gpu, args.max_idle_mib)
            state.setdefault('gpu_checks', {})[str(gpu)] = reading
            if not idle:
                continue
            job = pending[0]
            entry = state['jobs'][job['id']]
            entry['attempts'] += 1
            log_path = out / f"{job['id']}.attempt{entry['attempts']}.log"
            log = log_path.open('w', encoding='utf-8')
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), ENCBANK_REMOTE_QUEUE='1', PYTHONHASHSEED='0',
                PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            env.update(plan.get('env', {}))
            command = [sys.executable] + job['argv']
            process = subprocess.Popen(command, cwd=plan['cwd'], env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            entry.update(status='running', pid=process.pid, gpu=gpu, log=str(log_path),
                         command=command, started_at=time.time())
            running[gpu] = process, log, job['id']
            print(f"STARTED {job['id']} gpu={gpu} pid={process.pid}", flush=True)
            atomic_json(state_path, state)
        state['updated_at'] = time.time()
        atomic_json(state_path, state)
        unfinished = [j for j in state['jobs'].values()
                      if j['status'] in {'pending', 'interrupted', 'running'}]
        if not unfinished:
            state['finished_at'] = time.time()
            atomic_json(state_path, state)
            return 1 if any(j['status'] == 'failed' for j in state['jobs'].values()) else 0
        if not running and all(j.get('depends_on') for j in plan['jobs']
                if state['jobs'][j['id']]['status'] in {'pending', 'interrupted'}):
            blocked = [j for j in plan['jobs'] if state['jobs'][j['id']]['status'] in {'pending', 'interrupted'}
                and any(state['jobs'].get(d, {}).get('status') == 'failed' for d in j.get('depends_on', []))]
            for job in blocked:
                state['jobs'][job['id']].update(status='blocked', reason='dependency failed')
        time.sleep(10)


if __name__ == '__main__':
    raise SystemExit(main())
