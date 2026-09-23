"""One explicit QA retry with untouched workers, controllers, and GPU guards.

The failed backend attempt and never-started prefix queue remain unchanged.
Only the prefix module's backend output dependency is redirected in memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
CODE = ROOT / 'workspace/exp/comem_block_attention_20260911'
OUT = ROOT / 'outputs/qa_retry_20260913_attempt1'
OLD = {'backend': ROOT / 'outputs/backend_quality_20260912',
       'prefix': ROOT / 'outputs/native_prefix_quality_20260912'}
EXPECTED_PREFIX = (3254268, 982966640)
HANDOFF = ROOT / 'outputs/qa_slurm_handoff_20260912/declaration_v2.json'
HANDOFF_SHA = 'ae1c25b7a7c099e8c189edf6de6c4b6b96a98239cfc54fe6bee068ded6841726'
SCHEMA = 'comem-one-bounded-qa-retry-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)


def _pidfd_syscall(number, *args):
    """Older Python lacks wrappers even when the Linux kernel supports pidfds."""
    import ctypes
    if sys.platform != 'linux' or os.uname().machine not in ('x86_64', 'aarch64'):
        raise RuntimeError('No verified pidfd syscall mapping on this platform')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(arg) for arg in args))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    return os.pidfd_open(pid) if hasattr(os, 'pidfd_open') else _pidfd_syscall(434, pid, 0)


def signal_pidfd(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
    else:
        _pidfd_syscall(424, fd, sig, 0, 0)


def check_old_states(states, queue, *, prefix_alive=True):
    backend, prefix = states['backend'], states['prefix']
    owner = backend.get('controller', {})
    if (not owner or queue.process_identity(owner['pid']) == owner
            or backend.get('reason') != 'failed_no_promotion'
            or set(backend.get('jobs', {})) != {'smoke', 'full'}):
        raise RuntimeError('Original backend is not the exited failed attempt')
    smoke, full = backend['jobs']['smoke'], backend['jobs']['full']
    if (smoke.get('phase') != 'failed' or smoke.get('returncode') != -15
            or smoke.get('error') != 'External GPU process appeared; stopped only the QA worker'
            or full.get('phase') != 'queued'):
        raise RuntimeError('Retry only the known resource-interrupted smoke')
    owner = prefix.get('controller', {})
    if (tuple(owner.get(k) for k in ('pid', 'start_ticks')) != EXPECTED_PREFIX
            or str(CODE / 'qa_slurm_handoff.py') not in owner.get('argv', [])
            or prefix.get('reason') != 'backend_quality_controller_not_active'
            or set(prefix.get('jobs', {})) != {'smoke', 'full'}
            or any(job.get('phase') != 'queued' for job in prefix['jobs'].values())):
        raise RuntimeError('Original prefix is not the exact untouched waiting owner')
    live = queue.process_identity(owner['pid']) == owner
    if live != prefix_alive:
        raise RuntimeError('Original prefix process identity/liveness changed')
    for kind, state in states.items():
        for mode in ('smoke', 'full'):
            if Path(state['jobs'][mode]['out']) != OLD[kind] / mode:
                raise RuntimeError('Original output mapping changed')
        for mode in ('smoke', 'full') if kind == 'prefix' else ('full',):
            if (OLD[kind] / mode).exists() or (OLD[kind] / (mode + '_control') / 'worker.log').exists():
                raise RuntimeError('An allegedly untouched QA attempt has output')


def current_plan(queue, handoff):
    handoff.validate_declaration(HANDOFF, HANDOFF_SHA, queue)
    if any(queue.live_trainers(CODE / name, ROOT / 'outputs') for name in
           ('evaluate_backend_quality.py', 'evaluate_native_prefix_quality.py')):
        raise RuntimeError('QA worker active; no retry can be installed')
    states = {kind: queue.read_json(path / 'queue.json') for kind, path in OLD.items()}
    check_old_states(states, queue)
    backend = importlib.import_module('remote_backend_quality_queue')
    prefix = importlib.import_module('remote_prefix_quality_queue')
    for kind, module in (('backend', backend), ('prefix', prefix)):
        config = states[kind]['config']
        receipt = Path(config['cpu_receipt'])
        if sha(receipt) != config['cpu_receipt_sha256']:
            raise RuntimeError('Original CPU receipt changed')
        module.validate_cpu_receipt(receipt)
    return states


def validate_retry(path, expected_sha, *, queue):
    path = Path(path).resolve()
    if path != OUT / 'declaration.json' or sha(path) != expected_sha:
        raise RuntimeError('Retry declaration changed or moved')
    value = json.loads(path.read_text())
    if (value.get('schema') != SCHEMA or value.get('bounded_attempts') != 1
            or value.get('worker_or_guard_changes') is not False
            or value.get('wrapper_sha256') != sha(__file__)
            or value.get('handoff_declaration') != str(HANDOFF)
            or value.get('handoff_sha256') != HANDOFF_SHA
            or value.get('outputs') != {kind: str(OUT / kind) for kind in OLD}):
        raise RuntimeError('Unknown or altered retry declaration')
    for kind in OLD:
        snapshot = OUT / (kind + '_original_queue.json')
        if sha(snapshot) != value['original_snapshot_sha256'][kind]:
            raise RuntimeError('Original QA snapshot changed')
        state = json.loads(snapshot.read_text())
        if sha(OLD[kind] / 'queue.json') != sha(snapshot):
            raise RuntimeError('Original QA queue changed after retry handoff')
        if state['config'] != value['original_configs'][kind]:
            raise RuntimeError('Original QA configuration changed')
    states = {kind: json.loads((OUT / (kind + '_original_queue.json')).read_text()) for kind in OLD}
    check_old_states(states, queue, prefix_alive=False)
    return value


def execute(states, queue, handoff):
    if OUT.exists():
        raise RuntimeError('Retry directory already exists; never repeat or overwrite automatically')
    OUT.mkdir()
    write_new(OUT / 'started.json', dict(utc=datetime.now(timezone.utc).isoformat(), states=states))
    owner = states['prefix']['controller']
    # A pidfd binds both signals to this process even if a numeric PID is reused.
    fd = open_pidfd(owner['pid'])
    frozen = False
    try:
        if queue.process_identity(owner['pid']) != owner:
            raise RuntimeError('Prefix identity changed before stop')
        signal_pidfd(fd, signal.SIGSTOP)
        frozen = True
        deadline = time.monotonic() + 5
        while True:
            stat = Path('/proc') / str(owner['pid']) / 'stat'
            if stat.read_text().split(') ', 1)[1].split()[0] in ('T', 't'):
                break
            if queue.process_identity(owner['pid']) != owner or time.monotonic() > deadline:
                raise RuntimeError('Could not freeze the exact waiting prefix owner')
            time.sleep(.05)
        states = current_plan(queue, handoff)
        for kind in OLD:
            with (OUT / (kind + '_original_queue.json')).open('xb') as handle:
                handle.write((OLD[kind] / 'queue.json').read_bytes())
        signal_pidfd(fd, signal.SIGTERM)
        signal_pidfd(fd, signal.SIGCONT)
        frozen = False
        deadline = time.monotonic() + 10
        while queue.process_identity(owner['pid']) == owner:
            if time.monotonic() > deadline:
                raise RuntimeError('Exact waiting prefix controller did not exit')
            time.sleep(.1)
    finally:
        if frozen:
            signal_pidfd(fd, signal.SIGCONT)
        os.close(fd)
    check_old_states(states, queue, prefix_alive=False)
    value = dict(schema=SCHEMA, bounded_attempts=1, worker_or_guard_changes=False,
        wrapper_sha256=sha(__file__), handoff_declaration=str(HANDOFF), handoff_sha256=HANDOFF_SHA,
        outputs={kind: str(OUT / kind) for kind in OLD},
        original_configs={kind: states[kind]['config'] for kind in OLD},
        original_snapshot_sha256={kind: sha(OUT / (kind + '_original_queue.json')) for kind in OLD})
    declaration = OUT / 'declaration.json'
    write_new(declaration, value)
    digest = sha(declaration)
    validate_retry(declaration, digest, queue=queue)
    handoff.validate_declaration(HANDOFF, HANDOFF_SHA, queue)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
    result = {}
    for kind in OLD:
        command = [str(ROOT / 'venv/bin/python'), '-B', '-u', str(CODE / Path(__file__).name),
            '--queue-kind', kind, '--declaration', str(declaration), '--declaration-sha256', digest]
        with (OUT / (kind + '_controller.log')).open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT / 'workspace', env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result[kind] = dict(command=command, controller=queue.process_identity(proc.pid),
                           out=str(OUT / kind))
        write_new(OUT / (kind + '_launched.json'), result[kind])
    write_new(OUT / 'launched.json', result)
    return result


def run_queue(kind, declaration, digest, queue, handoff):
    value = validate_retry(declaration, digest, queue=queue)
    backend = importlib.import_module('remote_backend_quality_queue')
    priority = handoff.install_priority(backend, queue, HANDOFF, HANDOFF_SHA,
                                      OUT / (kind + '_priority.jsonl'))
    def checked_priority():
        try:
            validate_retry(declaration, digest, queue=queue)
        except Exception as exc:
            return False, 'retry_declaration_unavailable:' + type(exc).__name__ + ':' + str(exc)
        return priority()
    backend.training_priority_clear = checked_priority
    selected = backend if kind == 'backend' else importlib.import_module('remote_prefix_quality_queue')
    if kind == 'prefix':
        if selected.backend_queue is not backend:
            raise RuntimeError('Prefix must retain the same backend priority validator')
        selected.BACKEND_OUT = Path(value['outputs']['backend'])
    config = value['original_configs'][kind]
    sys.argv = [selected.__file__, '--out', value['outputs'][kind], '--cpu-receipt', config['cpu_receipt'],
                '--poll-seconds', '30', '--run']
    selected.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--queue-kind', choices=tuple(OLD))
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--declaration-sha256')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Linux remote only')
    import remote_sparse_queue as queue
    import qa_slurm_handoff as handoff
    if args.queue_kind:
        if args.execute or not args.declaration or not args.declaration_sha256:
            raise RuntimeError('Run only one declared retry controller')
        run_queue(args.queue_kind, args.declaration, args.declaration_sha256, queue, handoff)
        return
    plan = current_plan(queue, handoff)
    print(json.dumps(execute(plan, queue, handoff) if args.execute else plan))


if __name__ == '__main__':
    main()
