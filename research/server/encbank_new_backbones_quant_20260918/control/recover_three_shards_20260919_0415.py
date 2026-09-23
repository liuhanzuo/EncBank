"""Recover the three inspected failures, retaining the two active allocations."""
import datetime,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
FAILED={'large-final-m1-s0':'106607','large-final-m1-s1':'106608','large-final-m1-s2':'106609'}
ACTIVE={'large-final-m0-s3':'106606','large-final-m1-s3':'106640'}
def now():return datetime.datetime.utcnow().isoformat()+'Z'
def dump(p,v):
    t=p.with_suffix('.recovery.tmp')
    with t.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=35).stdout
def alive(pid,needle):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and needle.encode() in p.read_bytes()
def main():
    assert str(ROOT)=='/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918'
    history=ROOT/'maintenance_history/storage-recovery-20260919-0415'
    assert not history.exists(),'One-shot recovery already attempted: inspect receipts before any further action'
    proof=json.loads((ROOT/'control/storage_stability_0415.json').read_text())
    assert proof['eligible_for_inspected_recovery'] and proof['consecutive_successes']>=2 and time.time()-proof['checked_unix']<900
    assert (ROOT/'PAUSE_SUBMISSIONS.json').exists() and not (ROOT/'STOP_COORDINATOR').exists()
    owner=json.loads((ROOT/'coordinator_launch.json').read_text());assert owner['pid']==1861243
    assert alive(owner['pid'],str(ROOT/'control/coordinator.py'))
    assert not alive(1926367,'judge_watch.py')
    acct=run(['sacct','-j',','.join(list(FAILED.values())+list(ACTIVE.values())),'-n','-P','-o','JobIDRaw,State,ExitCode,End'])
    for job in FAILED.values():assert any(x.startswith(job+'|FAILED|1:0|') for x in acct.splitlines())
    queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'])
    own=[x for x in queue.splitlines() if 'qcm-q18-' in x]
    assert all(x.split('|')[0] in ACTIVE.values() for x in own)
    # Check the same shared filesystem from a CPU-only step in an existing owned allocation.
    active_probe=next(x.split('|')[0] for x in own if '|RUNNING|' in x)
    code="from pathlib import Path; import os,socket; p=Path('"+str(ROOT)+"/control/node-health-0415.tmp'); f=p.open('wb'); f.write(b'node-storage-ok'); f.flush(); os.fsync(f.fileno()); f.close(); assert p.read_bytes()==b'node-storage-ok'; q=p.with_suffix('.ok'); p.replace(q); q.unlink(); print(socket.gethostname()+': write-read-fsync-rename OK')"
    node=run(['srun','--jobid='+active_probe,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--time=00:01:00','/usr/bin/python3','-c',code])
    plan=json.loads((ROOT/'effective_plan.json').read_text());tasks={t['id']:t for t in plan['tasks']}
    observed={};locks=[]
    for ident,job in FAILED.items():
        out=ROOT/'runs'/ident;dest=(ROOT/tasks[ident]['complete']).parent
        assert json.loads((out/'submission.json').read_text())['job']==job
        parent=json.loads((out/'parent_exit.json').read_text())
        assert parent['actual_wait'] and parent['returncode']==1 and not parent['completion_exists']
        assert '[Errno 121]' in (out/'child.stderr.log').read_text()
        assert not (ROOT/tasks[ident]['complete']).exists()
        for d in [out,dest]:
            if d.exists():
                f=(d/'worker.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        p=dest/'predictions.jsonl';data=p.read_bytes() if p.exists() else b''
        rows=[json.loads(l) for l in data.splitlines()]
        assert len({(r['id'],r['arm']) for r in rows})==len(rows) and all(r['status']=='ok' for r in rows)
        observed[ident]=dict(saved_predictions=len(rows),sha256=hashlib.sha256(data).hexdigest())
    active_receipts={ident:(ROOT/'runs'/ident/'submission.json').read_bytes() for ident in ACTIVE}
    for ident,job in ACTIVE.items():assert json.loads(active_receipts[ident])['job']==job
    judge=ROOT/'judge_gpt6_astra';f=(judge/'watch.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert '[Errno 121]' in (judge/'watch.stderr.log').read_text()
    status=json.loads((judge/'status.json').read_text());assert status['available_complete'] and not status['errors']
    votes=[json.loads(l) for l in (judge/'judge_decisions.jsonl').read_text().splitlines()]
    assert len(votes)==9433 and len({(r['cohort'],r['arm'],r['id']) for r in votes})==9433
    history.mkdir();dump(history/'intent.json',dict(at=now(),failed=FAILED,active=ACTIVE,predictions=observed,stability=proof,node_check=node))
    (history/'sacct_original.txt').write_text(acct)
    (history/'effective_plan_before.json').write_bytes((ROOT/'effective_plan.json').read_bytes())
    # Stop only this CPU queue owner; worker processes and allocations are independent.
    assert alive(owner['pid'],str(ROOT/'control/coordinator.py'));os.kill(owner['pid'],signal.SIGTERM)
    for _ in range(40):
        if not alive(owner['pid'],str(ROOT/'control/coordinator.py')):break
        time.sleep(.25)
    assert not alive(owner['pid'],str(ROOT/'control/coordinator.py'))
    f=(ROOT/'coordinator.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    for ident,job in FAILED.items():
        out=ROOT/'runs'/ident;archive=out/'attempt_history'/('failed-'+job);archive.mkdir(parents=True)
        for p in list(out.iterdir()):
            if p.is_file() and p.name!='worker.lock':p.rename(archive/p.name)
        dest=(ROOT/tasks[ident]['complete']).parent
        for name in ['failure.json','progress.json']:
            p=dest/name
            if p.exists():p.rename(archive/('evaluation_'+name))
        tasks[ident]['recovery_from_job']=job
    for ident,data in active_receipts.items():assert (ROOT/'runs'/ident/'submission.json').read_bytes()==data
    for name in ['coordinator_launch.json','status.json','coordinator_failure.json']:
        p=ROOT/name
        if p.exists():p.rename(history/name)
    jhistory=history/'judge';jhistory.mkdir()
    for name in ['launch.json','watch_status.json','watch_failure.json','watch.stdout.log','watch.stderr.log']:
        p=judge/name
        if p.exists():p.rename(jhistory/name)
    dump(ROOT/'effective_plan.json',plan)
    dump(history/'recovery.json',dict(at=now(),failed=FAILED,predictions=observed,
        active_allocations_preserved=ACTIVE,judge_records_preserved=len(votes),new_api_retry_grants=0,
        science_unchanged=True,training_unchanged=True,stability=proof,node_check=node))
    (ROOT/'PAUSE_SUBMISSIONS.json').rename(history/'PAUSE_SUBMISSIONS.json')
    for f in locks:f.close()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'coordinator.stdout.log').open('ab') as stdout,(ROOT/'coordinator.stderr.log').open('ab') as stderr:
        child=subprocess.Popen([PY,'-B','-u',str(ROOT/'control/coordinator.py')],cwd=str(ROOT),env=env,
              stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    record=dict(pid=child.pid,at=now(),script=str(ROOT/'control/coordinator.py'),remote_root=str(ROOT),maximum_gpu_requests=4,
                recovery_evidence=str(history/'recovery.json'))
    dump(ROOT/'coordinator_launch.json',record)
    print(json.dumps(dict(launch=record,recovered=observed,preserved=ACTIVE,judge_preserved=len(votes),node_check=node)))
if __name__=='__main__':main()
