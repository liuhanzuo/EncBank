"""Bounded remote RTX 3090 queue for the isolated sparse-CoMem pilot.

Four resource/correctness smokes must succeed before any full training arm starts.
Uses the pre-existing per-GPU ownership locks and never interrupts other users.
Run on longjing-1 (actual hostname longjing-2); not a local timing runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path('/data/liuhanzuo/comem_v2_20260908')
CODE = ROOT / 'workspace/exp/comem_block_attention_20260911'
ARMS = ('D0', 'A', 'B', 'D1')
DISPLAY_PROCESSES = frozenset(('Xorg', 'X', 'gnome-shell'))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def publish(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def gpu_inventory(xml):
    """Fail closed, including Python processes reported as graphics (G) clients."""
    result = []
    for index, gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        raw_used = gpu.findtext('fb_memory_usage/used', '')
        try:
            used = int(raw_used.split()[0]) if raw_used.endswith('MiB') else None
        except (ValueError, IndexError):
            used = None
        process_node = gpu.find('processes')
        processes = [] if process_node is None else [
            {c.tag: c.text for c in proc} for proc in process_node.findall('process_info')]
        # A display server may remain resident. All other C/G/M process types,
        # unknown process names, and unavailable process inventory block launch.
        known_processes = (process_node is not None and
                           (not process_node.text or not process_node.text.strip()))
        busy = [p for p in processes
                if Path(p.get('process_name') or '').name not in DISPLAY_PROCESSES
                or p.get('type') != 'G']
        name = gpu.findtext('product_name', '')
        result.append(dict(index=index, name=name, used_mib=used, processes=processes,
                           eligible=('3090' in name and used is not None and used < 512
                                     and known_processes and not busy)))
    return result


def inventory():
    return gpu_inventory(subprocess.check_output(
        ['nvidia-smi', '-q', '-x'], text=True, timeout=15))


def eligible_gpus(snapshot):
    return sorted((g['index'] for g in snapshot if g['eligible']),
                  key=lambda gpu: (gpu == 0, gpu))


def process_identity(pid):
    """PID plus Linux start ticks, so a recycled PID cannot impersonate a job."""
    try:
        base = Path('/proc') / str(pid)
        stat = (base/'stat').read_text()
        tail = stat[stat.rfind(')') + 2:].split()
        argv = [v.decode(errors='replace') for v in (base/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=int(pid), start_ticks=int(tail[19]), argv=argv)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def live_trainers(trainer, out):
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        item = process_identity(int(directory.name))
        if not item or str(trainer) not in item['argv'] or '--out' not in item['argv']:
            continue
        try:
            destination = Path(item['argv'][item['argv'].index('--out') + 1]).resolve()
        except (IndexError, OSError):
            continue
        if destination == out or out in destination.parents:
            found.append(item)
    return found


def train_command(args, arm, stage, destination):
    cmd = [sys.executable, '-u', str(args.trainer),
           '--model', str(args.model), '--init-adapter', str(args.init_adapter),
           '--train', str(args.data_dir/'train.jsonl'), '--dev', str(args.data_dir/'dev.jsonl'),
           '--out', str(destination), '--device', 'cuda:0', '--arm', arm,
           '--j', '12', '--m', '16', '--rho', '0.5']
    if stage == 'smoke':
        cmd += ['--smoke', '--steps', '1', '--grad-accum', '2',
                '--dev-limit', '1', '--eval-limit', '1', '--final-eval-limit', '1',
                '--max-new-tokens', '2', '--skip-initial-eval']
    else:
        cmd += ['--steps', str(args.steps), '--grad-accum', str(args.grad_accum)]
    return cmd


def retry_failed_smoke(jobs, arm, out):
    """One explicit new attempt; never overwrite or silently resume a failure."""
    if arm != 'D0':
        raise ValueError('Only the explicitly authorized D0 retry is supported')
    name = 'smoke/' + arm
    previous = jobs[name]
    original = Path(out) / 'smoke' / arm
    destination = Path(out) / 'smoke' / (arm + '_retry1')
    if (previous.get('phase') != 'failed' or previous.get('retry_count', 0)
            or Path(previous['out']).resolve() != original.resolve()):
        raise ValueError('D0 retry requires the original failed smoke and no previous retry')
    if destination.exists():
        raise ValueError('D0 retry output already exists; inspect it without overwriting')
    jobs[name] = {key: previous[key] for key in
                  ('stage', 'arm', 'steps', 'grad_accum', 'dev_ids')}
    jobs[name].update(phase='queued', out=str(destination), retry_count=1,
                      previous_attempts=[dict(previous)])


class LeaseRefresher:
    """One stdlib writer refreshes leases while NVIDIA queries may block."""
    def __init__(self, writer, interval=1.):
        self.writer, self.interval = writer, interval
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.entries, self.errors = {}, {}
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='sparse-gpu-lease-heartbeat')

    def add(self, name, *, path, gpu, lock_fd, worker_script, run_id):
        with self.lock:
            values = dict(path=path, gpu=gpu, lock_fd=lock_fd,
                          worker_script=worker_script, run_id=run_id)
            self.writer(**values)
            self.entries[name] = values

    def remove(self, name):
        # Call before closing the FD, so an in-flight write cannot use a reused FD.
        with self.lock:
            self.entries.pop(name, None)
            self.errors.pop(name, None)

    def refresh(self):
        with self.lock:
            for name, values in self.entries.items():
                try:
                    self.writer(**values)
                except Exception as exc:
                    self.errors[name] = f'{type(exc).__name__}: {exc}'

    def failures(self):
        with self.lock:
            return dict(self.errors)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.refresh()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.)


def guard_running_jobs(running, jobs, snapshot, heartbeat_errors, *, unexpected, terminate):
    """Fail closed and stop only bound owned workers; leave external jobs alone."""
    for name, (proc, _log, _lock) in list(running.items()):
        if proc.poll() is not None:
            continue
        reason, foreign = heartbeat_errors.get(name), None
        if not reason:
            try:
                foreign = unexpected(snapshot, jobs[name]['gpu'], {proc.pid})
                if foreign:
                    reason = 'Another GPU process appeared'
            except Exception as exc:
                reason = f'GPU monitoring unavailable: {type(exc).__name__}: {exc}'
        if reason:
            jobs[name]['guard_failure'] = dict(reason=reason, observed_utc=utc_now(),
                                               unexpected_processes=foreign, snapshot=snapshot)
            terminate(proc)


def cleanup_owned_workers(running, leases, terminate, names=None):
    """Close ownership only after each child is confirmed exited, even on errors."""
    errors = []
    for name in list(running) if names is None else names:
        if name not in running:
            continue
        proc, log, gpu_lock = running[name]
        try:
            if proc.poll() is None:
                terminate(proc)
            if proc.poll() is None:
                raise RuntimeError('Owned worker still alive after stop request')
            leases.remove(name)
            log.close()
            gpu_lock.close()  # No LOCK_UN: any inherited description retains ownership.
            del running[name]
        except BaseException as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    if errors:
        raise RuntimeError('Unable to complete owned-worker cleanup: ' + '; '.join(errors))


@contextmanager
def supervised_workers(running, leases, terminate):
    try:
        yield
    finally:
        try:
            cleanup_owned_workers(running, leases, terminate)
        finally:
            leases.stop()


def completed_job(job):
    try:
        directory = Path(job['out'])
        state = read_json(directory/'status.json')
        metadata = read_json(directory/'metadata.json')
        recipe = metadata['recipe']
        gradient = state['gradient_check']
        smoke = job['stage'] == 'smoke'
        grad_accum = 2 if smoke else job['grad_accum']
        generation_limit = 2 if smoke else 128
        eval_limit = 1 if smoke else min(100, len(job['dev_ids']))
        finite = lambda v: type(v) in (int, float) and math.isfinite(v)
        metric = lambda v: finite(v) and 0 <= v <= 1
        # Seven upper-layer projections, each with two LoRA tensors. Every
        # upper layer processes the query even when document blocks are removed.
        expected_tensors = (metadata['model']['num_hidden_layers'] - 12) * 7 * 2
        if (state.get('complete') is not True or state.get('phase') != 'complete'
                or state.get('step') != job['steps'] or state.get('target_steps') != job['steps']
                or state.get('cursor') != job['steps'] * grad_accum or state.get('arm') != job['arm']
                or not finite(gradient.get('reader_norm')) or gradient['reader_norm'] <= 0
                or gradient.get('frozen_base_has_grad') is not False
                or type(gradient.get('total_trainable_tensors')) is not int
                or gradient['total_trainable_tensors'] != expected_tensors or expected_tensors <= 0
                or type(gradient.get('trainable_tensors_with_grad')) is not int
                or gradient['trainable_tensors_with_grad'] != expected_tensors
                or not (directory/'last.pt').is_file() or (directory/'last.pt').stat().st_size == 0):
            return False
        expected_recipe = dict(arm=job['arm'], steps=job['steps'], grad_accum=grad_accum,
            j=12, m=16, rho=.5, seed=42, smoke=smoke, max_new_tokens=generation_limit,
            final_eval_limit=1 if smoke else 100, dev_ids=job['dev_ids'],
            probe_mode='block' if job['arm'] in ('A', 'B') else 'dense',
            retain_ratio=1.0 if job['arm'] in ('A', 'D0') else .5)
        if any(recipe.get(k) != v for k, v in expected_recipe.items()):
            return False
        expected_ids = sorted(job['dev_ids'], key=lambda ident: hashlib.sha256(
            f'42:{ident}'.encode()).hexdigest())[:eval_limit]
        if not expected_ids or len(set(expected_ids)) != len(expected_ids):
            return False
        evaluation = read_json(directory/f"eval_step{job['steps']}.json")
        summary, records = evaluation['summary'], evaluation['records']
        if (not isinstance(records, list) or len(records) != len(expected_ids)
                or [r['id'] for r in records] != expected_ids
                or summary.get('examples') != len(records)
                or summary.get('max_new_tokens') != generation_limit
                or summary.get('score_scale') != '0-to-1'
                or summary.get('decoding') != 'greedy-natural-eos'
                or summary.get('formal_inference_timing') is not False
                or not all(metric(summary.get(k)) for k in ('token_f1', 'exact_match'))
                or not finite(summary.get('answer_ce')) or summary['answer_ce'] < 0):
            return False
        vocab_size = metadata['model']['vocab_size']
        for row in records:
            ids = row.get('generated_ids')
            if (not all(metric(row.get(k)) for k in ('token_f1', 'exact_match'))
                    or not finite(row.get('answer_ce')) or row['answer_ce'] < 0
                    or type(row.get('answer_ce_tokens')) is not int or row['answer_ce_tokens'] <= 0
                    or not isinstance(row.get('prediction'), str)
                    or not isinstance(ids, list) or not 1 <= len(ids) <= generation_limit
                    or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
                    or row.get('finish_reason') not in ('eos', 'max_new_tokens')
                    or (row['finish_reason'] == 'max_new_tokens' and len(ids) != generation_limit)):
                return False
        # Check that reported aggregates actually describe these same records.
        for key in ('token_f1', 'exact_match'):
            if not math.isclose(summary[key], sum(r[key] for r in records)/len(records),
                                rel_tol=1e-9, abs_tol=1e-9):
                return False
        ce = sum(r['answer_ce'] * r['answer_ce_tokens'] for r in records) / sum(
            r['answer_ce_tokens'] for r in records)
        return math.isclose(summary['answer_ce'], ce, rel_tol=1e-9, abs_tol=1e-9)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def self_test():
    def sample(used, process='', process_text=''):
        return ('<nvidia_smi_log><gpu><product_name>NVIDIA GeForce RTX 3090</product_name>'
                f'<fb_memory_usage><used>{used}</used></fb_memory_usage>'
                f'<processes>{process_text}{process}</processes></gpu></nvidia_smi_log>')
    assert eligible_gpus(gpu_inventory(sample('511 MiB'))) == [0]
    assert not eligible_gpus(gpu_inventory(sample('512 MiB')))
    assert not eligible_gpus(gpu_inventory(sample('N/A')))
    assert not eligible_gpus(gpu_inventory(sample('0 MiB', process_text='N/A')))
    py = '<process_info><pid>7</pid><type>G</type><process_name>python</process_name></process_info>'
    assert not eligible_gpus(gpu_inventory(sample('10 MiB', py)))
    xorg = '<process_info><pid>8</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name></process_info>'
    assert eligible_gpus(gpu_inventory(sample('4 MiB', xorg))) == [0]
    unknown = '<process_info><pid>9</pid><type>C</type><process_name>N/A</process_name></process_info>'
    assert not eligible_gpus(gpu_inventory(sample('1 MiB', unknown)))
    print('GPU admission self-tests passed: threshold, graphics Python, unknown inventory, display exception.')
    completion_self_test()


def completion_self_test():
    """Small stdlib fixtures; no model/checkpoint loading or GPU initialization."""
    import copy
    import tempfile
    with tempfile.TemporaryDirectory(prefix='sparse-queue-test-') as temp:
        out = Path(temp)
        job = dict(out=temp, stage='smoke', arm='B', steps=1, grad_accum=2, dev_ids=['held-out-1'])
        state = dict(complete=True, phase='complete', step=1, target_steps=1, cursor=2, arm='B',
            gradient_check=dict(reader_norm=.2, frozen_base_has_grad=False,
                trainable_tensors_with_grad=336, total_trainable_tensors=336))
        metadata = dict(model=dict(num_hidden_layers=36, vocab_size=152000), recipe=dict(
            arm='B', steps=1, grad_accum=2, j=12, m=16, rho=.5, seed=42, smoke=True,
            max_new_tokens=2, final_eval_limit=1, dev_ids=['held-out-1'],
            probe_mode='block', retain_ratio=.5))
        evaluation = dict(summary=dict(examples=1, token_f1=.5, exact_match=0., answer_ce=2.,
            score_scale='0-to-1', decoding='greedy-natural-eos', max_new_tokens=2,
            formal_inference_timing=False), records=[dict(id='held-out-1', token_f1=.5,
                exact_match=0., answer_ce=2., answer_ce_tokens=2, prediction='some answer',
                generated_ids=[1, 2], finish_reason='max_new_tokens')])
        publish(out/'status.json', state)
        publish(out/'metadata.json', metadata)
        publish(out/'eval_step1.json', evaluation)
        (out/'last.pt').write_bytes(b'fixture-not-loaded')
        assert completed_job(job)
        for change in (
                lambda v: v['gradient_check'].update(reader_norm=float('nan')),
                lambda v: v['gradient_check'].update(frozen_base_has_grad=True),
                lambda v: v['gradient_check'].update(trainable_tensors_with_grad=1),
                lambda v: v.update(phase='failed'),
                lambda v: v.update(cursor=1)):
            bad = copy.deepcopy(state)
            change(bad)
            publish(out/'status.json', bad)
            assert not completed_job(job)
        publish(out/'status.json', state)
        for change in (
                lambda v: v['summary'].update(token_f1=float('nan')),
                lambda v: v['summary'].update(examples=2),
                lambda v: v['summary'].update(token_f1=.6),
                lambda v: v['records'][0].update(id='wrong-example'),
                lambda v: v['records'][0].update(generated_ids=[]),
                lambda v: v['records'][0].update(generated_ids=[1, 2, 3]),
                lambda v: v['records'][0].update(exact_match=2.)):
            bad = copy.deepcopy(evaluation)
            change(bad)
            publish(out/'eval_step1.json', bad)
            assert not completed_job(job)
        publish(out/'eval_step1.json', evaluation)
        (out/'last.pt').unlink()
        assert not completed_job(job)
    print('Completion self-tests passed: valid fixture, finite gradients, exact IDs, aggregates, generation, checkpoint.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/sparse_comem_20260911')
    parser.add_argument('--trainer', type=Path, default=CODE/'train_sparse.py')
    parser.add_argument('--model', type=Path, default=ROOT/'models/Qwen3-8B')
    parser.add_argument('--init-adapter', type=Path, default=ROOT/'outputs/8b_j12_pub_4k/final/adapter.pt')
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--allow-training', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--retry-failed-smoke', choices=['D0'],
                        help='Explicit one-time D0 retry in smoke/D0_retry1; preserve the failed attempt')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if sys.platform != 'linux':
        raise RuntimeError('This queue may run only on the remote Linux RTX 3090 server')
    import fcntl
    from remote_gpu_guard import (write_lease, unexpected_processes,
                                  bind_owned_process, terminate_owned)
    if args.data_dir is None:
        parser.error('--data-dir must name the prepared sparse pilot data directory')
    if args.steps <= 0 or args.grad_accum <= 0 or not 5 <= args.poll_seconds <= 60:
        parser.error('Require positive budgets and poll-seconds in [5, 60]')
    for field in ('data_dir', 'out', 'trainer', 'model', 'init_adapter'):
        setattr(args, field, getattr(args, field).resolve())
        value = getattr(args, field)
        if ROOT not in value.parents:
            raise ValueError(f'{field} must stay inside the named remote experiment root')
    for path in (args.trainer, args.model/'config.json', args.init_adapter,
                 args.data_dir/'train.jsonl', args.data_dir/'dev.jsonl'):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    # Stable task lock also prevents duplicate controllers with different --out.
    task_lock = (ROOT/'logs'/'sparse_comem_20260911.queue.lock').open('a')
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller_lock = (args.out/'controller.lock').open('a')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {key: str(getattr(args, key)) for key in
              ('data_dir', 'out', 'trainer', 'model', 'init_adapter')}
    config.update(steps=args.steps, grad_accum=args.grad_accum, j=12, m=16, rho=.5,
                  arms=list(ARMS), scope='3090 training and accuracy only; no timing claims')
    old = read_json(args.out/'queue.json')
    if old and old.get('config') != config:
        raise RuntimeError('Existing output belongs to a different configuration; use a new output directory')
    existing = live_trainers(args.trainer, ROOT/'outputs')
    if existing:
        raise RuntimeError(f'Existing sparse trainer(s) still alive; do not duplicate: {existing}')
    with (args.data_dir/'dev.jsonl').open(encoding='utf-8-sig') as stream:
        dev_ids = [json.loads(line)['id'] for line in stream if line.strip()]
    if not dev_ids or len(set(dev_ids)) != len(dev_ids):
        raise ValueError('Prepared dev data must have nonempty unique IDs')
    jobs = {}
    for stage in ('smoke', 'train'):
        for arm in ARMS:
            name = stage+'/'+arm
            item = dict(stage=stage, arm=arm, phase='queued',
                        out=str(args.out/stage/arm), steps=1 if stage == 'smoke' else args.steps)
            previous = old.get('jobs', {}).get(name, {})
            if previous:
                item.update(previous)
            item.update(grad_accum=2 if stage == 'smoke' else args.grad_accum,
                        dev_ids=dev_ids[:1] if stage == 'smoke' else dev_ids)
            if previous:
                if completed_job(item):
                    item['phase'] = 'complete'
                elif item['phase'] in ('running', 'launching', 'complete'):
                    item.update(phase='failed', error='Previous worker has no valid completion artifacts; inspect before rerun')
            jobs[name] = item
    if args.retry_failed_smoke:
        retry_failed_smoke(jobs, args.retry_failed_smoke, args.out)
    running = {}
    current_gpus = []
    leases = LeaseRefresher(write_lease)
    leases.start()

    def persist(reason):
        publish(args.out/'queue.json', dict(pid=os.getpid(), controller=process_identity(os.getpid()),
            host=os.uname().nodename, updated_utc=utc_now(), reason=reason,
            config=config, allow_training=args.allow_training, gpus=current_gpus, jobs=jobs,
            complete=all(j['phase'] == 'complete' for j in jobs.values())))

    with supervised_workers(running, leases, terminate_owned):
        persist('initialized')
        while True:
            for name, (proc, log, gpu_lock) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                # Do not explicitly LOCK_UN: the inherited open file description
                # must retain ownership if the trainer left a live child behind.
                leases.remove(name)
                gpu_lock.close()
                item = jobs[name]
                item.update(returncode=rc, finished_utc=utc_now())
                if rc == 0 and not item.get('guard_failure') and completed_job(item):
                    item['phase'] = 'complete'
                else:
                    state = read_json(Path(item['out'])/'status.json')
                    item.update(phase='failed', error=item.get('guard_failure', {}).get('reason') or state.get('error') or
                                'Nonzero exit or missing validated training completion')
                del running[name]
                persist('worker_finished')
            smokes = [jobs['smoke/'+arm] for arm in ARMS]
            passed = all(j['phase'] == 'complete' for j in smokes)
            failed_smoke = any(j['phase'] == 'failed' for j in smokes)
            if not running and failed_smoke:
                persist('smoke_failed_no_training_promotion')
                leases.stop()
                return
            if not running and passed and not args.allow_training:
                persist('smokes_complete_training_not_enabled')
                leases.stop()
                return
            if all(j['phase'] in ('complete', 'failed') for j in jobs.values()):
                persist('complete' if all(j['phase'] == 'complete' for j in jobs.values()) else 'finished_with_failures')
                leases.stop()
                return
            try:
                current_gpus = inventory()
            except (subprocess.SubprocessError, OSError, ValueError, ET.ParseError) as error:
                current_gpus = [{'inventory_error': str(error)}]
                guard_running_jobs(running, jobs, current_gpus, leases.failures(),
                                   unexpected=unexpected_processes, terminate=terminate_owned)
                persist('gpu_inventory_unavailable_no_launch')
                time.sleep(1. if running else args.poll_seconds)
                continue
            guard_running_jobs(running, jobs, current_gpus, leases.failures(),
                               unexpected=unexpected_processes, terminate=terminate_owned)
            if any(jobs[name].get('guard_failure') for name in running):
                persist('owned_worker_stopped_for_gpu_guard')
                time.sleep(1.)
                continue
            reserved = {jobs[name]['gpu'] for name in running}
            candidates = [g for g in eligible_gpus(current_gpus) if g not in reserved]
            for name, item in jobs.items():
                if item['phase'] != 'queued' or not candidates or failed_smoke:
                    continue
                if item['stage'] == 'train' and (not passed or not args.allow_training):
                    continue
                gpu = candidates.pop(0)
                pre_lock = dict(observed_utc=utc_now(),
                                gpu=next(g for g in current_gpus if g['index'] == gpu))
                gpu_lock = (ROOT/'logs'/f'beacon_sft_gpu{gpu}.lock').open('a')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gpu_lock.close()
                    continue
                try:
                    locked_snapshot = inventory()
                    if gpu not in eligible_gpus(locked_snapshot):
                        gpu_lock.close()
                        continue
                    locked = dict(observed_utc=utc_now(),
                                  gpu=next(g for g in locked_snapshot if g['index'] == gpu))
                except (subprocess.SubprocessError, OSError, ValueError, ET.ParseError):
                    gpu_lock.close()
                    continue
                destination = Path(item['out'])
                if live_trainers(args.trainer, destination):
                    gpu_lock.close()
                    item.update(phase='failed', error='Duplicate live trainer detected')
                    persist('duplicate_worker_not_started')
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                # No silent overwrite/resume of interrupted training artifacts.
                if (destination/'status.json').exists():
                    gpu_lock.close()
                    item.update(phase='failed', error='Output already contains trainer status; inspect before explicit rerun')
                    persist('existing_output_not_overwritten')
                    continue
                command = train_command(args, item['arm'], item['stage'], destination)
                log_path = destination/'worker.log'
                log = log_path.open('a', encoding='utf-8')
                lease_path, run_id = destination/'gpu_lease.json', uuid.uuid4().hex
                leases.add(name, path=lease_path, gpu=gpu, lock_fd=gpu_lock.fileno(),
                           worker_script=args.trainer, run_id=run_id)
                item.update(phase='launching', gpu=gpu, command=command, log=str(log_path), started_utc=utc_now(),
                            gpu_lease=str(lease_path), run_id=run_id,
                            admission=dict(pre_lock=pre_lock, locked=locked))
                persist('worker_launching')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                       'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
                       'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                       'SPARSE_COMEM_QUEUE_TASK': 'sparse_comem_20260911',
                       'SPARSE_GPU_LEASE_PATH': str(lease_path),
                       'SPARSE_GPU_LOCK_FD': str(gpu_lock.fileno())}
                proc = None
                try:
                    proc = subprocess.Popen(command, cwd=ROOT/'workspace', env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(gpu_lock.fileno(),))
                    # Register immediately: bind/save failures must not orphan a worker.
                    running[name] = proc, log, gpu_lock
                    bind_owned_process(proc)
                    item.update(phase='running', pid=proc.pid, process=process_identity(proc.pid))
                    persist('running')
                except BaseException as error:
                    if proc is None:
                        log.close()
                        leases.remove(name)
                        gpu_lock.close()
                    else:
                        cleanup_owned_workers(running, leases, terminate_owned, names=[name])
                    item.update(phase='failed', error=f'Worker launch failed: {error}')
                    persist('worker_launch_failed')
                    if proc is not None or not isinstance(error, OSError):
                        raise
                    continue
                # Return to monitoring existing workers before another slow launch check.
                break
            persist('running' if running else 'waiting_for_free_3090')
            time.sleep(1. if running else args.poll_seconds)


if __name__ == '__main__':
    main()
