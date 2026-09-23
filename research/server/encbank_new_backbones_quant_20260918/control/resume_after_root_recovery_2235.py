"""Resume the sole paused owner after measured controller disk recovery."""
import datetime, fcntl, json, os, signal, subprocess, time
from pathlib import Path

ROOT = Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
HISTORY = ROOT / 'maintenance_history/slurm-root-restored-20260918-2235'
PYTHON = '/srv/encbank/Paper_Evolve/.venv/bin/python'


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(args):
    x = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=20)
    assert x.returncode == 0, x.stderr
    return x.stdout


def main():
    assert not HISTORY.exists()
    assert (ROOT / 'PAUSE_SUBMISSIONS.json').exists()
    vfs = os.statvfs('/var/spool')
    available = vfs.f_bavail * vfs.f_frsize
    assert available >= 1024 ** 3, 'Controller disk has not recovered'
    name = 'qcm-q18-quant-full-m1-s3'
    queue = run(['squeue', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%b'])
    accounting = run(['sacct', '-u', 'liuhanzuo', '-S', '2026-09-18T00:00:00',
                      '-n', '-P', '-o', 'JobIDRaw,JobName%100,State,ExitCode'])
    assert not any(line.split('|')[1] == name for line in queue.splitlines())
    assert not any(line.split('|')[1] == name for line in accounting.splitlines())
    launch = json.loads((ROOT / 'coordinator_launch.json').read_text())
    assert launch['pid'] == 1188635 and launch['submissions_paused']
    process = Path('/proc') / str(launch['pid']) / 'cmdline'
    assert process.exists() and b'control/coordinator.py' in process.read_bytes()
    receipt_path = ROOT / 'runs/quant-full-m1-s3/submission.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['at'] == '2026-09-18T14:29:06.443582+00:00'
    assert receipt['returncode'] == 1 and not receipt.get('job') and not receipt['stdout'].strip()
    assert receipt['stderr'].strip() == 'sbatch: error: Batch job submission failed: I/O error writing script/environment to file'
    # Paused CPU owner has no pending submissions. GPU workers/Judge are separate.
    os.kill(launch['pid'], signal.SIGTERM)
    for _ in range(30):
        if not process.exists() or b'control/coordinator.py' not in process.read_bytes():
            break
        time.sleep(.2)
    assert not process.exists() or b'control/coordinator.py' not in process.read_bytes()
    lock = (ROOT / 'coordinator.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    admission = (ROOT.parent / 'qencbank_gpu_admission.lock').open('a+b')
    fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
    HISTORY.mkdir()
    dump(HISTORY / 'evidence.json', dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        controller_available_bytes=available, previous_owner=launch, queue=queue,
        rejected_submission=receipt, accounting_no_matching_job=True,
        observed_disk_recovery=True, no_files_deleted_to_free_disk=True,
        scientific_code_unchanged=True))
    (HISTORY / 'sacct_before.txt').write_text(accounting)
    for name in ['coordinator_launch.json', 'status.json']:
        (HISTORY / name).write_bytes((ROOT / name).read_bytes())
    receipt_path.replace(HISTORY / 'rejected_submission.json')
    (ROOT / 'PAUSE_SUBMISSIONS.json').replace(HISTORY / 'PAUSE_SUBMISSIONS.json')
    admission.close()
    lock.close()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / 'coordinator.stdout.log').open('ab') as out, (ROOT / 'coordinator.stderr.log').open('ab') as err:
        child = subprocess.Popen([PYTHON, '-B', '-u', str(ROOT / 'control/coordinator.py')],
          cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    fresh = dict(pid=child.pid, at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        remote_root=str(ROOT), script=str(ROOT / 'control/coordinator.py'), maximum_gpu_requests=4,
        recovery_evidence=str(HISTORY / 'evidence.json'))
    dump(ROOT / 'coordinator_launch.json', fresh)
    dump(HISTORY / 'new_launch.json', fresh)
    print(json.dumps(fresh), flush=True)


if __name__ == '__main__':
    main()
