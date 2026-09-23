"""Submit each dependent evaluation after real training success; at most three remote GPUs."""
import datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(p,r):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
def run(args):return subprocess.check_output(args,universal_newlines=True,timeout=30)
def main():
    lock=(ROOT/'eval_owner.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        states={};finished=[]
        for j in [6,12,18]:
            rec=json.loads((ROOT/'runs'/('train-j%d.json'%j)).read_text());job=rec['job']
            lines=run(['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode']).splitlines()
            training=next((line.split('|')[1:3] for line in lines if line.startswith(job+'|')),['UNKNOWN',''])
            states[str(j)]=dict(training_job=job,training_state=training)
            path=ROOT/'runs'/('eval-j%d.json'%j)
            if path.exists():
                er=json.loads(path.read_text());assert er.get('job'),'Uncertain submission; inspect before retry'
                lines=run(['sacct','-j',er['job'],'-n','-P','-o','JobIDRaw,State,ExitCode']).splitlines()
                es=next((line.split('|')[1:3] for line in lines if line.startswith(er['job']+'|')),['UNKNOWN',''])
                states[str(j)].update(eval_job=er['job'],eval_state=es)
                if es==['COMPLETED','0:0']:
                    dest=ROOT/'quality'/('j%d_s42'%j)
                    assert json.loads((dest/'complete.json').read_text())['records']==800
                    wait=json.loads((dest/'parent_exit.json').read_text());assert wait['actual_wait'] and wait['returncode']==0
                    finished.append(j)
                elif any(es[0].startswith(x) for x in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']):
                    raise RuntimeError('Evaluation failed: j%d %s'%(j,es))
                continue
            if training!=['COMPLETED','0:0']:
                if any(training[0].startswith(x) for x in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']):
                    raise RuntimeError('Training failed: j%d %s'%(j,training))
                continue
            dest=ROOT/'training'/('j%d_s42'%j)
            wait=json.loads((dest/'training_parent_exit.json').read_text());assert wait['actual_wait'] and wait['returncode']==0
            assert json.loads((dest/'complete.json').read_text())['step']==4000
            with (ROOT/'admission.lock').open('a+b') as admission:
                fcntl.flock(admission,fcntl.LOCK_EX)
                queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'])
                own=[l for l in queue.splitlines() if '|qcm-c34-' in l or '|qcm-q18-' in l]
                if sum(int(l.split('|')[3].split(':')[-1]) for l in own)>=3:continue
                name='qcm-c34-eval-j%d'%j;assert not any('|'+name+'|' in l for l in own)
                cmd=['sbatch','--parsable','--job-name='+name,'--dependency=singleton','--partition=gpu',
                     '--gres=gpu:nvidia_l20d:1','--cpus-per-task=4','--mem=96G','--time=12:00:00',
                     '--output='+str(ROOT/'logs'/('eval-j%d-%%j.out'%j)),
                     '--error='+str(ROOT/'logs'/('eval-j%d-%%j.err'%j)),str(ROOT/'eval_job.sh'),str(j)]
                receipt=dict(command=cmd,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue_before=own)
                dump(path,receipt)
                result=subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
                receipt.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                if result.returncode==0:receipt['job']=result.stdout.strip().split(';')[0];assert receipt['job'].isdigit()
                dump(path,receipt);assert result.returncode==0
        dump(ROOT/'eval_owner_status.json',dict(phase='COMPLETE' if len(finished)==3 else 'ACTIVE',states=states,
             at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid()))
        if len(finished)==3:return
        time.sleep(30)
if __name__=='__main__':
    try:main()
    except BaseException as exc:dump(ROOT/'eval_owner_failure.json',dict(error=repr(exc)));raise
