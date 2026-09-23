"""Restore CPU progress tracking while the Slurm controller filesystem is full."""
import datetime, fcntl, json, os, socket, subprocess
from pathlib import Path

ROOT = Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
HISTORY = ROOT / 'maintenance_history/slurm-root-full-20260918-2234'
PYTHON = '/srv/encbank/Paper_Evolve/.venv/bin/python'


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    assert not HISTORY.exists(), 'Already executed; inspect recorded state'
    lock = (ROOT / 'coordinator.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = json.loads((ROOT / 'coordinator_launch.json').read_text())
    process = Path('/proc') / str(old['pid']) / 'cmdline'
    assert not process.exists() or b'control/coordinator.py' not in process.read_bytes()
    receipt = json.loads((ROOT / 'runs/quant-full-m1-s3/submission.json').read_text())
    assert receipt['returncode'] == 1 and not receipt.get('job') and not receipt['stdout'].strip()
    assert receipt['stderr'].strip() == 'sbatch: error: Batch job submission failed: I/O error writing script/environment to file'
    vfs = os.statvfs('/var/spool')
    evidence = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), hostname=socket.gethostname(),
       available_bytes=vfs.f_bavail * vfs.f_frsize, root_free_bytes=vfs.f_bfree * vfs.f_frsize,
       reason='Slurm controller root filesystem full; retain existing GPU jobs and refresh completed status for Judge',
       rejected_submission=receipt, automatic_submission_retry=False,
       resume_condition='Inspect restored controller disk space, reconcile squeue/sacct for the exact rejected job name, archive the rejected receipt, and resume the sole owner under its lock')
    assert socket.gethostname() == 'gpu-host'
    incoming = ROOT / 'control/coordinator.incoming.py'
    compile(incoming.read_text(), str(incoming), 'exec')
    HISTORY.mkdir()
    for name in ['coordinator_launch.json', 'coordinator_failure.json', 'status.json']:
        p = ROOT / name
        if p.exists():
            (HISTORY / name).write_bytes(p.read_bytes())
    (HISTORY / 'coordinator_before.py').write_bytes((ROOT / 'control/coordinator.py').read_bytes())
    dump(HISTORY / 'evidence.json', evidence)
    dump(ROOT / 'PAUSE_SUBMISSIONS.json', evidence)
    incoming.replace(ROOT / 'control/coordinator.py')
    (ROOT / 'coordinator_failure.json').unlink()
    lock.close()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with (ROOT / 'coordinator.stdout.log').open('ab') as out, (ROOT / 'coordinator.stderr.log').open('ab') as err:
        child = subprocess.Popen([PYTHON, '-B', '-u', str(ROOT / 'control/coordinator.py')],
          cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    fresh = dict(pid=child.pid, at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          remote_root=str(ROOT), script=str(ROOT / 'control/coordinator.py'), maximum_gpu_requests=4,
          recovery_evidence=str(HISTORY / 'evidence.json'), submissions_paused=True)
    dump(ROOT / 'coordinator_launch.json', fresh)
    dump(HISTORY / 'new_launch.json', fresh)
    print(json.dumps(fresh), flush=True)


if __name__ == '__main__':
    main()
