"""Reconcile dead-owner host reservations on lj-gpu7 without touching live tasks.

The original JSON is preserved byte-for-byte before a guarded, atomic update.
Run on lj-gpu7; dry-run first and pass its SHA to --apply.
"""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path


SERVER = Path('/cluster/home/liuhanzuo/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
LEDGER_DIR = Path('/cluster/home/liuhanzuo/qcomem_runtime_20260911/server_control_20260920/unbounded_host_admission') / socket.gethostname()
LEDGER = LEDGER_DIR / 'reservations.json'
HOST = 'l20-instan-001'


def identity_alive(identity):
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        current = {
            'pid': identity['pid'],
            'start_ticks': int(stat[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        }
        return current == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


def process_args():
    rows = subprocess.check_output(['ps', '-u', 'liuhanzuo', '-o', 'pid=,args='], text=True).splitlines()
    result = []
    for line in rows:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != os.getpid():
            result.append((int(parts[0]), parts[1]))
    return result


def job_identity(owner):
    path = SERVER / owner / 'submission.json'
    if not path.exists():
        return None, 'missing_submission'
    row = json.loads(path.read_text())
    job_id = str(row['job_id'])
    query = subprocess.run(
        ['squeue', '-h', '-j', job_id, '-o', '%T'], text=True,
        capture_output=True, timeout=20)
    if query.returncode and 'Invalid job id specified' not in query.stderr:
        raise RuntimeError(f'squeue {job_id}: {query.stderr.strip()}')
    state = query.stdout.strip()
    return job_id, state or 'not_in_queue'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.apply and not args.expected_sha256:
        parser.error('--apply requires the SHA from the dry run')
    if socket.gethostname() != HOST:
        raise SystemExit(f'wrong node: {socket.gethostname()}')

    with (LEDGER_DIR / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = LEDGER.read_bytes()
        sha = hashlib.sha256(original).hexdigest()
        if args.expected_sha256 and args.expected_sha256 != sha:
            raise SystemExit(f'ledger changed: expected {args.expected_sha256}, actual {sha}')
        reservations = json.loads(original)
        processes = process_args()
        owners = sorted({row['owner'] for row in reservations.values()})
        evidence = {}
        for owner in owners:
            entries = [(key, row) for key, row in reservations.items() if row['owner'] == owner]
            live_identity = any(identity_alive(row['identity']) for _, row in entries)
            matching = [(pid, cmd[:240]) for pid, cmd in processes if f'/{owner}/' in cmd]
            job_id, queue_state = job_identity(owner)
            keep = live_identity or bool(matching) or queue_state != 'not_in_queue'
            evidence[owner] = {
                'job_id': job_id,
                'queue_state': queue_state,
                'stored_owner_identity_alive': live_identity,
                'matching_processes': matching,
                'entries': len(entries),
                'memory_mb': sum(row['memory_mb'] for _, row in entries),
                'keep': keep,
            }
        remove = [key for key, row in reservations.items() if not evidence[row['owner']]['keep']]
        remaining = {key: row for key, row in reservations.items() if key not in set(remove)}
        report = {
            'epoch': time.time(),
            'hostname': socket.gethostname(),
            'ledger_sha256_before': sha,
            'entry_count_before': len(reservations),
            'memory_mb_before': sum(row['memory_mb'] for row in reservations.values()),
            'entry_count_after': len(remaining),
            'memory_mb_after': sum(row['memory_mb'] for row in remaining.values()),
            'removed_keys': remove,
            'owners': evidence,
            'applied': args.apply,
        }
        if args.apply:
            assert remove, 'nothing to reconcile'
            assert any(row['keep'] for row in evidence.values()), 'unexpected: no live-owner reservations'
            backup = LEDGER_DIR / f'reservations.before_reconcile_20260925_{sha[:12]}.json'
            with backup.open('xb') as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
            updated = json.dumps(remaining, indent=2).encode() + b'\n'
            temp = LEDGER_DIR / f'reservations.json.reconcile-{os.getpid()}.tmp'
            with temp.open('xb') as stream:
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, LEDGER)
            assert LEDGER.read_bytes() == updated
            report['backup'] = str(backup)
            report['ledger_sha256_after'] = hashlib.sha256(updated).hexdigest()
            receipt = LEDGER_DIR / f'reconcile_20260925_{sha[:12]}.json'
            receipt.write_text(json.dumps(report, indent=2) + '\n')
            report['receipt'] = str(receipt)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
