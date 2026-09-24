"""Wait for main admission and stable BeeGFS, then submit Hot score-only shards once."""
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid

MAIN_STATUS = '/tmp/encbank_accuracy_recovery_20260924_status.json'
STATUS = '/tmp/encbank_hot_accuracy_recovery_20260924_status.json'
LOCK = '/tmp/encbank_hot_accuracy_recovery_20260924.lock'
DRIVER = '/tmp/encbank_hot_accuracy_recovery_20260924.py'
PY = '/cluster/home/USER/qencbank_runtime_20260911/python312/bin/python'
SERIES = '/cluster/home/USER/qencbank_align_codex_20260911/tf27b_hot_live_20260921'
BASES = [SERIES, SERIES + '/hot24_peft_adaptive_full89_20260924',
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
    path = base + '/.encbank_hot_health_' + uuid.uuid4().hex
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


def call(phase):
    result = subprocess.run([PY, DRIVER, phase], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        save({'state':'FAILED_'+phase.upper(), 'returncode':result.returncode,
              'stdout':result.stdout[-4000:], 'stderr':result.stderr[-4000:]})
        return False
    save({'state':phase.upper()+'_COMPLETE', 'stdout':result.stdout[-4000:]})
    return True


def run():
    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save({'state':'WAITING_FOR_MAIN'})
        while True:
            try:
                main = json.load(open(MAIN_STATUS))
            except (IOError, ValueError):
                main = {}
            if main.get('state') == 'SUBMITTED':
                break
            if main.get('state') in ('SUBMISSION_UNCERTAIN','FAILED_ACTIVATION','FAILED'):
                save({'state':'MAIN_NEEDS_RECONCILIATION', 'main_state':main.get('state')})
                return 1
            time.sleep(10)
        streak = 0
        while streak < 30:
            try:
                for base in BASES:
                    probe(base)
                streak += 1
                save({'state':'WAITING_FOR_STORAGE', 'healthy_intervals':streak})
            except Exception as exc:
                streak = 0
                save({'state':'WAITING_FOR_STORAGE', 'healthy_intervals':0,
                      'last_error':repr(exc)})
            time.sleep(10)
        save({'state':'PREPARING'})
        if not call('prepare'):
            return 1
        save({'state':'SUBMITTING'})
        if not call('submit'):
            return 1
        save({'state':'SUBMITTED'})
        return 0


if __name__ == '__main__':
    try:
        code = run()
    except Exception as exc:
        save({'state':'FAILED', 'error':repr(exc)})
        raise
    sys.exit(code)
