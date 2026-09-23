"""Linux Harbor controller. Slurm owns its lifetime; the client computer is optional."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import host_admission
from control_plane_recovery import ControlPlane
from server_transport import Transport, save
from startup_wait import wait_for_worker

H = Path(__file__).resolve().parent
P = json.loads((H / 'plan.json').read_text())
O = H / 'execution'
BOX = Path(P['rpc_root']) / P['arm']
RESULTS = Path(P['results_root'])
DRAIN = False


def drain(signum, _frame):
    global DRAIN
    DRAIN = True
    save(O / 'drain_requested.json', {'signal': signum, 'epoch': time.time()})


def start_task(row):
    name = row['task']
    assert name in P['tasks'] and Path(name).name == name
    template = H / (P['arm'] + '_harbor_template.json')
    config = json.loads(template.read_text())
    config.update(job_name=name, jobs_dir=str(RESULTS))
    config['tasks'] = [{'path': str(Path(P['task_root']) / name)}]
    config_path = O / 'configs' / (name + '.json')
    assert not config_path.exists() and not (RESULTS / name).exists(), 'Task already launched'
    save(config_path, config)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(H), LITELLM_LOCAL_MODEL_COST_MAP='True',
               DOCKER_HOST=P['docker_host'], PYTHONUNBUFFERED='1',
               TMPDIR=P['controller_tmp'], XDG_CACHE_HOME=P['controller_cache'],
               APPTAINER_CACHEDIR=P['container_cache'], APPTAINER_TMPDIR=P['controller_tmp'],
               SINGULARITY_CACHEDIR=P['container_cache'], SINGULARITY_TMPDIR=P['controller_tmp'])
    command = [str(Path(P['harbor_python']).with_name('harbor')), 'run', '-c', str(config_path)]
    output = (O / (name + '.stdout.log')).open('wb')
    error = (O / (name + '.stderr.log')).open('wb')
    try:
        process = subprocess.Popen(command, cwd=H, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=error, start_new_session=True)
    except BaseException:
        output.close(); error.close()
        raise
    save(O / 'launches' / (name + '.json'), dict(row, pid=process.pid, argv=command,
         identity=host_admission.process_identity(process.pid), epoch=time.time(),
         container_backend=P['container_backend'], server_only=True))
    return dict(row, child=process, out=output, err=error)


def main():
    assert sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID')
    assert H == Path(P['remote_root']).resolve()
    O.mkdir(exist_ok=True); BOX.mkdir(parents=True, exist_ok=True)
    lock = (O / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (O / 'owner_registration.json').exists(), 'Prior owner exists; review its closure instead of replaying tasks'
    save(O / 'owner_registration.json', {'identity': host_admission.process_identity(os.getpid()),
         'epoch': time.time(), 'hostname': os.uname().nodename, 'job_id': os.environ['SLURM_JOB_ID'],
         'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'server_only': True})
    signal.signal(signal.SIGTERM, drain); signal.signal(signal.SIGINT, drain)
    transport = Transport(H, P, O)
    active, futures, seen, completed = {}, {}, set(), []
    pending = list(P['resource_inventory'])
    control = None
    ready = False
    try:
        status = wait_for_worker(transport.ctl, save, O, max_wait_seconds=P['bootstrap_timeout_seconds'] + 300)
        ready = True
        save(O / 'worker_ready.json', status['worker_ready.json'])
        control = ControlPlane(transport.ctl, host_admission, H.name, O, save)
        last_status = 0
        with ThreadPoolExecutor(max_workers=P['concurrent_tasks']) as pool:
            while pending or active:
                if DRAIN and not active:
                    break
                for name, info in list(active.items()):
                    child = info['child']
                    if child.poll() is None:
                        continue
                    task_futures = [future for task, future in futures.values() if task == name]
                    if any(not future.done() for future in task_futures):
                        continue
                    code = child.wait()
                    info['out'].close(); info['err'].close()
                    requests = [json.loads(path.read_text()) for path in BOX.glob('*.request.json')]
                    requests = [request for request in requests if request['task'] == name]
                    releases = []
                    if P['arm'] == 'encbank':
                        for session in sorted({request['task_id'] for request in requests}):
                            releases.append(transport.ctl('release', session))
                    results = list((RESULTS / name).glob('*/result.json'))
                    wait_record = {'task': name, 'pid': child.pid, 'exit_code': code,
                                   'actual_parent_wait': True, 'epoch': time.time()}
                    save(O / 'parent_waits' / (name + '.json'), wait_record)
                    save(O / 'receipts' / (name + '.json'), dict(wait_record,
                         requests=len(requests), release=releases, result_paths=[str(path) for path in results],
                         transport_errors=[repr(future.exception()) for future in task_futures if future.exception()]))
                    host_admission.release(H.name, name)
                    completed.append(name); del active[name]
                    if code != 0 and not results:
                        save(O / 'pre_agent_failures' / (name + '.json'), dict(wait_record,
                             model_requests=len(requests), automatic_retry=False, classification='INFRASTRUCTURE_FAILURE'))
                for path in sorted(BOX.glob('*.request.json')):
                    if path.name in seen:
                        continue
                    request = json.loads(path.read_text())
                    seen.add(path.name)
                    if request['task'] not in active:
                        save(BOX / (request['request_id'] + '.error.json'),
                             {'error': 'Request after task closure; no replay', 'no_automatic_request_retry': True})
                        continue
                    futures[path.name] = (request['task'], pool.submit(transport.handle, path,
                                          active[request['task']]['child']))
                healthy = control.tick()
                enabled = bool(pending) and len(active) < P['concurrent_tasks'] and not DRAIN and not (O / 'pause_new_tasks.json').exists()
                row = control.admit(pending[0] if enabled else None, healthy, enabled)
                if row is not None:
                    assert row == pending[0]
                    pending.pop(0)
                    try:
                        active[row['task']] = start_task(row)
                    except BaseException:
                        host_admission.release(H.name, row['task'])
                        raise
                if time.monotonic() - last_status >= 20:
                    save(O / 'status.json', {'state': 'draining' if DRAIN else 'server_tasks_running',
                         'epoch': time.time(), 'active': {name: {'pid': row['child'].pid, 'memory_mb': row['memory_mb']}
                                                       for name, row in active.items()},
                         'completed': completed, 'pending': [row['task'] for row in pending],
                         'remote_healthy': healthy, 'server_only': True})
                    last_status = time.monotonic()
                time.sleep(.2)
        save(O / 'harbor_receipt.json', {'all_task_parents_waited': not active,
             'tasks_closed': len(completed), 'pending_not_launched': [row['task'] for row in pending],
             'epoch': time.time(), 'drained': DRAIN})
    except BaseException:
        save(O / 'controller_failure.json', {'error': traceback.format_exc(), 'epoch': time.time(),
             'active': {name: row['child'].pid for name, row in active.items()},
             'classification': 'INFRASTRUCTURE_FAILURE', 'automatic_retry': False})
        # Only our own child process groups are cancelled. No task is automatically rerun.
        for row in active.values():
            if row['child'].poll() is None:
                os.killpg(row['child'].pid, signal.SIGTERM)
        for name, row in active.items():
            try:
                code = row['child'].wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(row['child'].pid, signal.SIGKILL)
                code = row['child'].wait()
            save(O / 'interrupted_parents' / (name + '.json'), {'pid': row['child'].pid, 'exit_code': code,
                 'actual_parent_wait': True, 'classification': 'INFRASTRUCTURE_INTERRUPTION'})
        raise
    finally:
        if control is not None:
            control.close()
        if ready and all(row['child'].poll() is not None for row in active.values()):
            try:
                save(O / 'stop_receipt.json', transport.ctl('stop'))
            except Exception as exc:
                save(O / 'stop_error.json', {'error': repr(exc)})
    save(O / 'owner_complete.json', {'epoch': time.time(), 'all_task_launches_accounted': True,
         'remaining_tasks': [row['task'] for row in pending], 'server_only': True})


if __name__ == '__main__':
    main()
