"""Submit this fixed array once after readiness; an uncertain submission requires inspection."""
import fcntl, json, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    with (ROOT/'submit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=ROOT/'submission.json'
        if receipt.exists():
            print(receipt.read_text()); return
        assert json.loads((ROOT/'cpu_checks.json').read_text())['passed']
        assert (ROOT/'readiness').is_dir() and (ROOT/'data/protocol.json').exists()
        intent=ROOT/'submission_intent.json'
        assert not intent.exists(), 'Inspect squeue/sacct for an uncertain earlier submission before retrying'
        intent.write_text(json.dumps(dict(at=time.time(),script='pipeline.slurm',array='0-4%4')))
        completed=subprocess.run(['sbatch','--parsable','pipeline.slurm'],cwd=ROOT,capture_output=True,text=True,check=True)
        job_id=completed.stdout.strip().split(';')[0]
        assert job_id.isdigit(), completed.stdout
        value=dict(job_id=job_id,array='0-4%4',max_concurrent_gpus=4,submitted_at=time.time(),stdout=completed.stdout.strip())
        receipt.write_text(json.dumps(value,indent=2)); print(json.dumps(value))


if __name__=='__main__':
    main()
