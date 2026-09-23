"""One bounded recovery attempt; retries only deployment's transient I/O failures."""
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918'


def state(phase, **extra):
    path = ROOT / 'storage_resume_state.json'
    value = dict(phase=phase, pid=os.getpid(), at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)
    print(json.dumps(value), flush=True)


def main():
    previous = ROOT / 'storage_resume_state.json'
    if previous.exists():
        raise RuntimeError('Existing recovery receipt: inspect and archive before starting another recovery process.')
    for attempt in range(1, 61):
        state('CHECKING_STORAGE', attempt=attempt, maximum_attempts=60)
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'deploy_io_recovery.py')],
            text=True, capture_output=True, timeout=60)
        record = dict(attempt=attempt, returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr)
        (ROOT / 'delivery' / ('storage_deploy_attempt_%02d.json' % attempt)).write_text(
            json.dumps(record, indent=2) + '\n', encoding='utf-8')
        if result.returncode == 0:
            break
        message = result.stdout + result.stderr
        if not any('[Errno %d]' % code in message for code in (5, 70, 110, 121)):
            state('NEEDS_ATTENTION', reason='Non-transient deployment result; no automatic GPU resubmit.', **record)
            return
        if attempt == 60:
            state('WAITING_STORAGE_NEXT_HEARTBEAT', attempts=attempt,
                  reason='One-hour recovery window exhausted; saved results untouched.')
            return
        state('WAITING_STORAGE', attempt=attempt, retry_after_seconds=60)
        time.sleep(60)
    state('RECOVERY_DISPATCHED_ONCE', attempt=attempt)
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1',
      'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout -k 10s 180s '
      '/srv/encbank/Paper_Evolve/.venv/bin/python -B -u '
      + REMOTE + '/control/recover_storage_1722.py']
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=210)
    except subprocess.TimeoutExpired:
        state('NEEDS_ATTENTION', reason='Recovery SSH deadline reached; inspect remote owner and receipts, never rerun blindly.')
        return
    record = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    (ROOT / 'delivery/storage_recovery_dispatch.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    state('RECOVERY_STARTED' if result.returncode == 0 else 'NEEDS_ATTENTION', **record)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        state('NEEDS_ATTENTION', error=repr(exc))
        raise
