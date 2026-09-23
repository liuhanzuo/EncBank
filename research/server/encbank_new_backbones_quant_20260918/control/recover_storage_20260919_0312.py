"""One inspected recovery of five evaluation jobs after the 03:12 shared I/O outage."""
import datetime,fcntl,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
FAILED={'large-final-m0-s3':'106475','large-final-m1-s0':'106477',
        'large-final-m1-s1':'106545','large-final-m1-s2':'106546','large-final-m1-s3':'106575'}
def now():return datetime.datetime.utcnow().isoformat()+'Z'
def dump(p,v):
    t=p.with_suffix('.recovery.tmp')
    with t.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def run(cmd):
    return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30).stdout
def absent(pid,needle):
    p=Path('/proc')/str(pid)/'cmdline'
    assert not(p.exists() and needle.encode() in p.read_bytes()),'Previous process still alive'
def main():
    assert str(ROOT)=='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
    history=ROOT/'maintenance_history/storage-recovery-20260919-0312'
    assert not history.exists(),'Recovery already attempted; inspect saved state instead of repeating'
    absent(548826,'control/coordinator.py');absent(2904295,'judge_watch.py')
    locks=[]
    for p in [ROOT/'coordinator.lock',ROOT/'judge_gpt6_astra/watch.lock']:
        f=p.open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T'])
    assert not any('qcm-q18-' in x for x in queue.splitlines()),queue
    acct=run(['sacct','-j',','.join(FAILED.values()),'-n','-P','-o','JobIDRaw,State,ExitCode,End'])
    for job in FAILED.values():assert any(x.startswith(job+'|FAILED|1:0|') for x in acct.splitlines()),acct
    for base in [ROOT/'control',ROOT/'results/large-final',ROOT/'judge_gpt6_astra']:
        probe=base/'health-recovery-20260919-0312.json'
        dump(probe,dict(check='write-read-fsync-rename'))
        assert json.loads(probe.read_text())['check']=='write-read-fsync-rename';probe.unlink()
    plan=json.loads((ROOT/'effective_plan.json').read_text())
    tasks={t['id']:t for t in plan['tasks']}
    observations={}
    for ident,job in FAILED.items():
        task=tasks[ident];out=ROOT/'runs'/ident;dest=(ROOT/task['complete']).parent
        assert task['kind']=='evaluation' and '--mode' in task['command'] and 'large-final' in task['command']
        assert json.loads((out/'submission.json').read_text())['job']==job
        assert not (ROOT/task['complete']).exists()
        err=(out/'child.stderr.log').read_text()
        assert '[Errno 121]' in err or ('Output file' in err and 'could not be opened' in err),ident
        f=(out/'worker.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        if dest.exists():
            f=(dest/'worker.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        p=dest/'predictions.jsonl';data=p.read_bytes() if p.exists() else b''
        rows=[json.loads(l) for l in data.splitlines()]
        assert len({(r['id'],r['arm']) for r in rows})==len(rows)
        assert all(r['status']=='ok' for r in rows)
        observations[ident]=dict(saved_predictions=len(rows),sha256=hashlib.sha256(data).hexdigest(),
                                original_parent_receipt_exists=(out/'parent_exit.json').exists())
    judge=ROOT/'judge_gpt6_astra'
    assert '[Errno 121]' in (judge/'watch.stderr.log').read_text()
    status=json.loads((judge/'status.json').read_text())
    assert status['available_complete'] and not status['errors']
    votes=[json.loads(l) for l in (judge/'judge_decisions.jsonl').read_text().splitlines()]
    assert len(votes)==9433 and len({(r['cohort'],r['arm'],r['id']) for r in votes})==9433
    history.mkdir()
    (history/'sacct_original.txt').write_text(acct)
    (history/'effective_plan_before.json').write_bytes((ROOT/'effective_plan.json').read_bytes())
    dump(history/'intent.json',dict(at=now(),failed=FAILED,predictions=observations,judge_records=len(votes)))
    for ident,job in FAILED.items():
        out=ROOT/'runs'/ident;archive=out/'attempt_history'/('failed-'+job);archive.mkdir(parents=True)
        for p in list(out.iterdir()):
            if p.is_file() and p.name!='worker.lock':p.rename(archive/p.name)
        dest=(ROOT/tasks[ident]['complete']).parent
        for name in ['failure.json','progress.json']:
            p=dest/name
            if p.exists():p.rename(archive/('evaluation_'+name))
        tasks[ident]['recovery_from_job']=job
    for name in ['coordinator_launch.json','status.json','coordinator_failure.json']:
        p=ROOT/name
        if p.exists():p.rename(history/name)
    jhistory=history/'judge';jhistory.mkdir()
    for name in ['launch.json','watch_status.json','watch_failure.json','watch.stdout.log','watch.stderr.log']:
        p=judge/name
        if p.exists():p.rename(jhistory/name)
    dump(ROOT/'effective_plan.json',plan)
    dump(history/'recovery.json',dict(at=now(),reason='Shared storage Errno121, including failed Triton output file',
        failed=FAILED,predictions=observations,judge_records_preserved=len(votes),
        new_api_retry_grants=0,science_unchanged=True,training_unchanged=True,
        missing_parent_exits='106545/106546 original receipt write failed; Slurm FAILED1:0 retained, no fabricated wait receipt'))
    for f in locks:f.close()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'coordinator.stdout.log').open('ab') as stdout,(ROOT/'coordinator.stderr.log').open('ab') as stderr:
        child=subprocess.Popen([PY,'-B','-u',str(ROOT/'control/coordinator.py')],cwd=str(ROOT),env=env,
            stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    record=dict(pid=child.pid,at=now(),script=str(ROOT/'control/coordinator.py'),remote_root=str(ROOT),maximum_gpu_requests=4,
                recovery_evidence=str(history/'recovery.json'))
    dump(ROOT/'coordinator_launch.json',record)
    print(json.dumps(dict(launch=record,recovered=observations,judge_preserved=len(votes))))
if __name__=='__main__':main()
