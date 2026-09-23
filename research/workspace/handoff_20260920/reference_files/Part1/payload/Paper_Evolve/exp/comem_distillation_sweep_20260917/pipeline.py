"""One allocation per branch, sequential training/evaluation, resumable without duplicates."""
import argparse, fcntl, json, os, subprocess, sys, time, traceback
import config
from train_support import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',type=int,required=True);args=ap.parse_args()
    arm=config.ARMS[args.arm];out=config.ROOT/'runs'/arm['name'];out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            return
        atomic_json(out/'worker.json',dict(arm=arm,job_id=os.environ['SLURM_JOB_ID'],pid=os.getpid(),phase='TRAINING',started=time.time()))
        train=out/'training'
        done=(train/'status.json').exists() and json.loads((train/'status.json').read_text())['complete']
        try:
            if not done:
                if arm['mode']=='full':
                    command=[sys.executable,'-u','train_full.py','--arm',str(args.arm)]
                else:
                    command=[sys.executable,'-u','train_support.py','--model',config.MODEL,'--data',config.DATA,
                        '--out',str(train),'--devices','cuda:0','--token-cache',config.TOKEN_CACHE,
                        '--j','12','--rank',str(arm['rank']),'--alpha',str(arm['alpha']),
                        '--steps',str(arm['steps']),'--lr',str(arm['lr']),'--save-every','1000','--log-every','10']
                    if (train/'last.pt').exists():
                        command += ['--resume',str(train/'last.pt')]
                subprocess.run(command,check=True,cwd=config.ROOT)
            atomic_json(out/'worker.json',dict(arm=arm,job_id=os.environ['SLURM_JOB_ID'],phase='EVALUATING',at=time.time()))
            subprocess.run([sys.executable,'-u','evaluate.py','--arm',str(args.arm)],check=True,cwd=config.ROOT)
            atomic_json(out/'complete.json',dict(complete=True,arm=arm,at=time.time()))
            atomic_json(out/'worker.json',dict(arm=arm,job_id=os.environ['SLURM_JOB_ID'],phase='COMPLETE',at=time.time()))
        except BaseException:
            atomic_json(out/'failure.json',dict(arm=arm,job_id=os.environ.get('SLURM_JOB_ID'),traceback=traceback.format_exc(),at=time.time()))
            raise


if __name__=='__main__':
    main()
