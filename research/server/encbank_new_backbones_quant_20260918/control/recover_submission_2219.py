"""One inspected recovery of a rejected sbatch; never retry uncertain submissions."""
import datetime
import fcntl
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
TASK = 'quant-full-m1-s3'
HISTORY = ROOT / 'maintenance_history/submission-recovery-20260918-2219'
PYTHON = '/srv/encbank/Paper_Evolve/.venv/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=25)
    assert result.returncode == 0, result.stderr
    return result.stdout


def main():
    assert not HISTORY.exists(), 'One-shot recovery already attempted; inspect evidence first'
    lock = (ROOT / 'coordinator.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    admission = (ROOT.parent / 'qencbank_gpu_admission.lock').open('a+b')
    fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = json.loads((ROOT / 'coordinator_launch.json').read_text())
    cmdline = Path('/proc') / str(launch['pid']) / 'cmdline'
    assert not cmdline.exists() or b'control/coordinator.py' not in cmdline.read_bytes()
    assert not (ROOT / 'STOP_COORDINATOR').exists()
    receipt_path = ROOT / 'runs' / TASK / 'submission.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['returncode'] == 1 and not receipt.get('job') and not receipt['stdout'].strip()
    assert receipt['stderr'].strip() == 'sbatch: error: Batch job submission failed: I/O error writing script/environment to file'
    assert receipt['at'] == '2026-09-18T14:19:05.289197+00:00'
    queue = run(['squeue', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%b'])
    name = 'qcm-q18-' + TASK
    assert not any(line.split('|')[1] == name for line in queue.splitlines())
    accounting = run(['sacct', '-u', 'liuhanzuo', '-S', '2026-09-18T00:00:00',
                      '-n', '-P', '-o', 'JobIDRaw,JobName%100,State,ExitCode'])
    assert not any(line.split('|')[1] == name for line in accounting.splitlines()), 'Scheduler accepted a matching job; do not resubmit'
    own = [line.split('|') for line in queue.splitlines() if line.split('|')[1].startswith('qcm-q18-')]
    assert sum(int(parts[3].split(':')[-1]) for parts in own) <= 3
    # Exercise the exact small-file write/rename/read operations before recovery.
    probe = ROOT / 'submission_recovery_2219_probe.tmp'
    renamed = ROOT / 'submission_recovery_2219_probe.ok'
    assert not probe.exists() and not renamed.exists()
    probe.write_text('storage-read-write-ok\n')
    probe.replace(renamed)
    assert renamed.read_text() == 'storage-read-write-ok\n'
    renamed.unlink()
    HISTORY.mkdir()
    dump(HISTORY / 'evidence.json', dict(at=now(), rejected_submission=receipt,
         previous_owner=launch, queue=queue, accounting_no_matching_job=True,
         cause='Slurm rejected script/environment storage write; no job accepted',
         unchanged_scientific_code=True, unaffected_running_tasks=own))
    (HISTORY / 'sacct_before.txt').write_text(accounting)
    for name in ('coordinator_launch.json', 'coordinator_failure.json', 'status.json'):
        path = ROOT / name
        if path.exists():
            (HISTORY / name).write_bytes(path.read_bytes())
    receipt_path.replace(HISTORY / 'rejected_submission.json')
    (ROOT / 'coordinator_failure.json').unlink()
    admission.close()
    lock.close()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / 'coordinator.stdout.log').open('ab') as out, (ROOT / 'coordinator.stderr.log').open('ab') as err:
        child = subprocess.Popen([PYTHON, '-B', '-u', str(ROOT / 'control/coordinator.py')],
              cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    fresh = dict(pid=child.pid, at=now(), remote_root=str(ROOT),
                 script=str(ROOT / 'control/coordinator.py'), maximum_gpu_requests=4,
                 recovery_evidence=str(HISTORY / 'evidence.json'))
    dump(ROOT / 'coordinator_launch.json', fresh)
    dump(HISTORY / 'new_launch.json', fresh)
    print(json.dumps(fresh), flush=True)


if __name__ == '__main__':
    main()
