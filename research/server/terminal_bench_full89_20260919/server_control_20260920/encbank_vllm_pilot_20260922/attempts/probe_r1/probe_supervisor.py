"""Record exact inputs and the actual child wait for a synthetic kernel probe."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path

H=Path(__file__).resolve().parent
def save(name,value):
    with (H/name).open('x') as f:json.dump(value,f,indent=2)

def main():
    started=time.time()
    manifest=json.loads((H/'source_manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((H/name).read_bytes()).hexdigest()==digest,name
    save('started.json',dict(epoch=started,job_id=os.environ['SLURM_JOB_ID'],hostname=os.uname().nodename,
        source_manifest=manifest,model_calls=0,benchmark_attempts=0))
    with (H/'probe.stdout.log').open('x') as stdout,(H/'probe.stderr.log').open('x') as stderr:
        child=subprocess.Popen(['/srv/encbank/venvs/rllm/bin/python','-u',str(H/'gpu_probe.py')],stdout=stdout,stderr=stderr,cwd=H)
        rc=child.wait()
    save('parent_receipt.json',dict(started_epoch=started,ended_epoch=time.time(),pid=child.pid,exit_code=rc,
        actual_parent_wait=True,job_id=os.environ['SLURM_JOB_ID'],model_calls=0,benchmark_attempts=0))
    if rc:raise RuntimeError('Synthetic GPU probe failed; preserve logs. Exit '+str(rc))

if __name__=='__main__':
    try:main()
    except BaseException:
        save('failure.json',dict(epoch=time.time(),error=traceback.format_exc()));raise
