"""Reassign one queue at a natural model-job boundary without rerunning that job."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from remote_queue import atomic_json, gpu_idle
from summarize_runs import summarize


def proc_info(pid):
    try:
        # Fields after comm begin with state (3); exit_code is Linux field 52.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'ppid': int(fields[1]), 'wait_status': int(fields[49])}
    except FileNotFoundError:
        return None


def sole_running_job(state, plan):
    running = [(key, value) for key, value in state['jobs'].items()
               if value['status'] == 'running']
    if len(running) != 1:
        raise RuntimeError('Boundary handoff requires exactly one recorded active job')
    key, entry = running[0]
    jobs = {job['id']: job for job in plan['jobs']}
    if key not in jobs:
        raise RuntimeError('Active job is absent from the unchanged plan')
    return key, entry, jobs[key]


def wait_unlocked(path):
    with path.open('a') as lock:
        for _ in range(100):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                return
            except BlockingIOError:
                time.sleep(.1)
    raise RuntimeError(f'Old coordinator did not release {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--status-dir', required=True, type=Path)
    parser.add_argument('--bootstrap-pid', required=True, type=int)
    parser.add_argument('--queue-pid', required=True, type=int)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    folder = args.status_dir.resolve()
    handoff_lock = (folder / 'reassign.lock').open('a')
    fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bootstrap = json.loads((folder / 'status.json').read_text())
    if bootstrap['pid'] != args.bootstrap_pid or bootstrap['status'] != 'running' or bootstrap['phase'] != 'full':
        raise RuntimeError('Bootstrap identity or phase changed; no action taken')
    plan = json.loads(Path(bootstrap['full_plan']).read_text())
    queue_dir = Path(plan['state_dir'])
    state_path = queue_dir / 'state.json'
    command = [part.decode() for part in Path(f'/proc/{args.bootstrap_pid}/cmdline').read_bytes().split(b'\0') if part]
    if not any(part.endswith('/benchmark_remote_bootstrap.py') for part in command):
        raise RuntimeError('Unexpected bootstrap command')
    if '--status-dir' not in command or Path(command[command.index('--status-dir') + 1]).resolve() != folder:
        raise RuntimeError('Bootstrap belongs to another workflow')
    queue_command = Path(f'/proc/{args.queue_pid}/cmdline').read_bytes().split(b'\0')
    if not any(part.endswith(b'/remote_queue.py') for part in queue_command):
        raise RuntimeError('Unexpected queue command')
    if Path(f'/proc/{args.queue_pid}').stat().st_uid != os.getuid():
        raise RuntimeError('Queue belongs to another user')
    if proc_info(args.queue_pid)['ppid'] != args.bootstrap_pid:
        raise RuntimeError('Queue is not a child of the expected bootstrap')
    new_gpus = [int(value) for value in args.gpus.split(',')]
    if len(new_gpus) != len(set(new_gpus)):
        raise RuntimeError('Duplicate GPU indices')
    initial_state = json.loads(state_path.read_text())
    if initial_state['queue_pid'] != args.queue_pid:
        raise RuntimeError('Queue identity changed before admission')
    for gpu in set(new_gpus) - set(initial_state['gpus']):
        idle, reading = gpu_idle(gpu, 512)
        if not idle:
            raise RuntimeError(f'Additional GPU {gpu} is occupied; no schedulers paused: {reading}')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    history = folder / 'history' / ('reassign_' + stamp)
    history.mkdir(parents=True, exist_ok=False)
    paused = []
    retired = False
    def interrupted(signum, frame):
        raise RuntimeError(f'Boundary handoff interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        # Only schedulers are paused. Their independently sessioned model child runs on.
        for pid in (args.bootstrap_pid, args.queue_pid):
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        for _ in range(100):
            if all(proc_info(pid)['state'] in ('T', 't') for pid in paused):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Coordinators did not pause')
        state = json.loads(state_path.read_text())
        if state['queue_pid'] != args.queue_pid:
            raise RuntimeError('Queue identity changed')
        key, entry, job = sole_running_job(state, plan)
        child = entry['pid']
        children = Path(f'/proc/{args.queue_pid}/task/{args.queue_pid}/children').read_text().split()
        if children != [str(child)] or proc_info(child)['ppid'] != args.queue_pid:
            raise RuntimeError('Process children differ from the recorded active job')
        (history / 'bootstrap_before.json').write_text(json.dumps(bootstrap, indent=2))
        (history / 'queue_before.json').write_text(json.dumps(state, indent=2))
        atomic_json(history / 'handoff.json', {'status': 'waiting_for_natural_child_exit', 'job': key,
                    'child_pid': child, 'bootstrap_pid': args.bootstrap_pid, 'queue_pid': args.queue_pid,
                    'gpus': new_gpus, 'started_at': time.time()})
        print(json.dumps({'status': 'waiting_for_natural_child_exit', 'job': key, 'child_pid': child}), flush=True)
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            info = proc_info(child)
            if info and info['state'] == 'Z':
                if info['wait_status'] != 0:
                    raise RuntimeError(f'Model child exited unsuccessfully: {info}')
                break
            if info is None:
                raise RuntimeError('Child exit status unavailable; original queue will resume')
            time.sleep(1)
        else:
            raise TimeoutError('Natural job boundary not reached in time')
        summary = summarize({**plan, 'jobs': [job], 'reused_results': []})
        if not summary['all_complete'] or summary['completed_jobs'] != 1:
            raise RuntimeError(f'Finished job is not complete: {summary.get("diagnostics")}')
        (history / 'finished_job_summary.json').write_text(json.dumps(summary, indent=2))
        # Allow driver accounting to settle after the completed child releases CUDA.
        gpu_deadline = time.monotonic() + 30
        while True:
            readings, all_idle = {}, True
            for gpu in new_gpus:
                idle, reading = gpu_idle(gpu, 512)
                readings[str(gpu)] = reading
                all_idle = all_idle and idle
            if all_idle:
                break
            if time.monotonic() >= gpu_deadline:
                raise RuntimeError(f'GPU occupied at boundary; original queue will resume: {readings}')
            time.sleep(1)
        # The model has exited successfully and its full output has been checked.
        # Retire only its two paused schedulers and retain the completed attempt.
        for pid in (args.queue_pid, args.bootstrap_pid):
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        retired = True
        wait_unlocked(queue_dir / 'queue.lock')
        wait_unlocked(folder / 'bootstrap.lock')
        entry.update(status='completed', exit_code=0, finished_at=time.time(),
                     boundary_handoff=str(history))
        state['updated_at'] = time.time()
        atomic_json(state_path, state)
        command[command.index('--gpus') + 1] = args.gpus
        with (folder / 'bootstrap.log').open('a') as log, open(os.devnull) as devnull:
            replacement = subprocess.Popen(command, cwd=Path(__file__).resolve().parent,
                stdin=devnull, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        result = {'status': 'reassigned', 'replacement_pid': replacement.pid, 'completed_job': key,
                  'child_pid': child, 'gpus': new_gpus, 'gpu_readings': readings,
                  'history': str(history), 'finished_at': time.time()}
        atomic_json(history / 'handoff.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        atomic_json(history / 'handoff_error.json', {'error': repr(exc), 'retired': retired,
                    'time': time.time()})
        raise
    finally:
        if not retired:
            for pid in reversed(paused):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    main()
