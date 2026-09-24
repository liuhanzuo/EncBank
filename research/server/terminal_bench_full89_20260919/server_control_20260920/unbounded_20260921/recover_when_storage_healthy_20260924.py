"""One-shot server-side admission of prepared score-only runs after BeeGFS recovers.

This does not retry submitted jobs or scientific attempts. An uncertain submission
ends the service for manual reconciliation.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid

ROOT = '/cluster/home/USER/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920'
U = ROOT + '/unbounded_20260921'
REC = U + '/accuracy_recovery_20260924'
PY = '/cluster/home/USER/qencbank_runtime_20260911/python312/bin/python'
DRIVER = U + '/accuracy_recovery_20260924.py'
STATUS = '/tmp/encbank_accuracy_recovery_20260924_status.json'
LOCK = '/tmp/encbank_accuracy_recovery_20260924.lock'
PROBE_DIRS = [REC, ROOT + '/k12_acc_recovery_1_20260924',
              '/cluster/home/USER/qencbank_runtime_20260911/server_control_20260920']


def save(value):
    temp = STATUS + '.tmp'
    with open(temp, 'w') as out:
        json.dump(dict(epoch=time.time(), **value), out, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.rename(temp, STATUS)


def probe(base):
    path = base + '/.encbank_health_' + uuid.uuid4().hex
    data = os.urandom(4096)
    try:
        with open(path, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        with open(path, 'rb') as inp:
            assert inp.read() == data
        os.unlink(path)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def run():
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not os.path.exists(REC + '/registry_activated.json')
        streak = 0
        save({'state':'WAITING_FOR_STORAGE', 'healthy_intervals':0})
        while streak < 30:
            try:
                for base in PROBE_DIRS:
                    probe(base)
                streak += 1
                save({'state':'WAITING_FOR_STORAGE', 'healthy_intervals':streak})
            except Exception as exc:
                streak = 0
                save({'state':'WAITING_FOR_STORAGE', 'healthy_intervals':0,
                      'last_error':repr(exc)})
            time.sleep(10)
        save({'state':'ACTIVATING', 'healthy_intervals':streak})
        result = subprocess.run([PY, DRIVER, 'activate'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode:
            save({'state':'FAILED_ACTIVATION', 'returncode':result.returncode,
                  'stdout':result.stdout[-4000:], 'stderr':result.stderr[-4000:]})
            return 1
        save({'state':'SUBMITTING', 'activation_stdout':result.stdout[-2000:]})
        result = subprocess.run([PY, DRIVER, 'submit'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode:
            save({'state':'SUBMISSION_UNCERTAIN', 'returncode':result.returncode,
                  'stdout':result.stdout[-4000:], 'stderr':result.stderr[-4000:]})
            return 1
        save({'state':'SUBMITTED', 'stdout':result.stdout[-4000:]})
        return 0


if __name__ == '__main__':
    try:
        code = run()
    except Exception as exc:
        save({'state':'FAILED', 'error':repr(exc)})
        raise
    sys.exit(code)
