"""One inspected recovery after 2026-09-18 storage/node failures, not blind retries."""
import datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PYTHON='/srv/encbank/Paper_Evolve/.venv/bin/python'
FAILED={'train-r128-m1':'103237','quant-full-m0-s2':'104239','quant-full-m1-s0':'104335'}

def dump(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def main():
    assert str(ROOT.resolve())=='/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918'
    lock=(ROOT/'coordinator.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'effective_plan.json').read_text())
    queue=subprocess.run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T'],text=True,capture_output=True,check=True).stdout
    assert not any('qcm-q18-' in row for row in queue.splitlines()),queue
    evidence=json.loads((ROOT/'delivery/recovery_checkpoint_20260918_1535.json').read_text())
    assert evidence['step']==5000 and evidence['rng_restorable'] and evidence['optimizer_steps']==[5000]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    history=ROOT/'maintenance_history'/('storage-recovery-'+stamp);history.mkdir(parents=True)
    (history/'effective_plan_before.json').write_bytes((ROOT/'effective_plan.json').read_bytes())
    accounting=subprocess.run(['sacct','-j',','.join(FAILED.values()),'-n','-P','-o','JobIDRaw,State,ExitCode'],text=True,capture_output=True,check=True).stdout
    for job in FAILED.values():assert any(line.startswith(job+'|FAILED|') for line in accounting.splitlines()),accounting
    (history/'sacct_original.txt').write_text(accounting)
    observed={}
    for task in plan['tasks']:
        if task['id'] not in FAILED:continue
        out=ROOT/'runs'/task['id'];receipt=json.loads((out/'submission.json').read_text())
        assert receipt['job']==FAILED[task['id']]
        assert not (ROOT/task['complete']).exists()
        worker_lock=(out/'worker.lock').open('a+b');fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if task['kind']=='evaluation':
            pred=(ROOT/task['complete']).parent/'predictions.jsonl'
            keys=set();count=0
            with pred.open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);key=(r['id'],r['arm']);assert key not in keys;keys.add(key);count+=1
            assert count==({'quant-full-m0-s2':1350,'quant-full-m1-s0':505}[task['id']])
            observed[task['id']]=dict(saved_predictions=count,unaltered_sha256=hashlib.sha256(pred.read_bytes()).hexdigest())
        archive=out/'attempt_history'/('failed-'+FAILED[task['id']]);archive.mkdir(parents=True)
        for path in list(out.iterdir()):
            if path.is_file() and path.name!='worker.lock':path.rename(archive/path.name)
        task.update(recovery_from_job=FAILED[task['id']],launcher='control/recovery.sh')
        if task['kind']=='training':
            assert '--resume' not in task['command'];task['command'].append('--resume')
            task['resume_step']=5000;task['prestart_duration_hours']=[6,8]
        worker_lock.close()
    train=ROOT/'training/Qwen3.8-27B'
    log=train/'training.jsonl';old=log.read_bytes();rows=old.splitlines(keepends=True)
    parsed=[json.loads(line) for line in rows]
    assert [r['step'] for r in parsed]==list(range(1,5242))
    (history/'training_preoutage.jsonl').write_bytes(old)
    temp=train/'training.resume.tmp';temp.write_bytes(b''.join(line for line,r in zip(rows,parsed) if r['step']<=5000));temp.replace(log)
    (train/'progress.json').rename(history/'training_progress_preoutage.json')
    dump(train/'progress.json',dict(phase='awaiting_resume',step=5000,target_steps=8000,elapsed_s=0,
        resume_from_step=5000,discarded_uncheckpointed_steps=241,original_record_preserved=str(history/'training_preoutage.jsonl')))
    dump(ROOT/'effective_plan.json',plan)
    for name in ['coordinator_launch.json','status.json','coordinator_failure.json']:
        f=ROOT/name
        if f.exists():f.rename(history/name)
    dump(history/'recovery.json',dict(reason='Inspected shared-filesystem errno70 and Slurm failed 0:53/1:0',
        original_failures=FAILED,checkpoint=evidence,original_training_log_sha256=hashlib.sha256(old).hexdigest(),
        predictions=observed,scientific_code_unchanged=True,no_completed_task_repeated=True,
        adapter_and_optimizer_and_rng_restored=True,at=stamp))
    # Release the owner lock only after all recovery receipts/config are durable.
    lock.close()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'coordinator.stdout.log').open('ab') as out,(ROOT/'coordinator.stderr.log').open('ab') as err:
        child=subprocess.Popen([PYTHON,'-B','-u',str(ROOT/'control/coordinator.py')],cwd=ROOT,env=env,
             stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    launch=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_root=str(ROOT),
                script=str(ROOT/'control/coordinator.py'),role='Independent four-GPU CPU admission owner',
                maximum_gpu_requests=4,recovery_evidence=str(history/'recovery.json'))
    dump(ROOT/'coordinator_launch.json',launch);print(json.dumps(launch))

if __name__=='__main__':main()
