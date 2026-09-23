"""Retry saved-output finalization first; release missing 27B shards only after success."""
import datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
FAILED={'large-final-m0-s3':'106606','large-final-m1-s0':'106701','large-final-m1-s1':'106702'}
GATE='large-final-m0-s3'
def now():return datetime.datetime.utcnow().isoformat()+'Z'
def dump(p,v):
    t=p.with_suffix('.recovery.tmp')
    with t.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30).stdout
def main():
    assert str(ROOT)=='/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918'
    history=ROOT/'maintenance_history/storage-canary-20260919-0600'
    assert not history.exists(),'Recovery already attempted; inspect before any repeat'
    for pid,needle in [(3759947,'control/coordinator.py'),(3817318,'judge_watch.py')]:
        p=Path('/proc')/str(pid)/'cmdline';assert not (p.exists() and needle.encode() in p.read_bytes())
    proofs=json.loads((ROOT/'control/original_path_proofs_0600.json').read_text())
    assert proofs['login']['all_passed'] and proofs['gpu8']['actual_returncode']==0 and proofs['gpu8']['gpus_requested']==0
    assert proofs['gpu8']['result']['all_passed'] and proofs['gpu8']['result']['job']=='106868'
    assert time.time()-proofs['collected_unix']<900
    locks=[]
    for path in [ROOT/'coordinator.lock',ROOT/'judge_gpt6_astra/watch.lock',ROOT.parent/'qcomem_gpu_admission.lock']:
        f=path.open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b']);assert 'qcm-q18-' not in queue
    acct=run(['sacct','-j',','.join(list(FAILED.values())+['106640','106868']),'-n','-P','-o','JobIDRaw,State,ExitCode,End'])
    for job in FAILED.values():assert any(l.startswith(job+'|FAILED|1:0|') for l in acct.splitlines())
    for job in ['106640','106868']:assert any(l.startswith(job+'|COMPLETED|0:0|') for l in acct.splitlines())
    plan=json.loads((ROOT/'effective_plan.json').read_text());tasks={t['id']:t for t in plan['tasks']}
    observed={}
    for ident,job in FAILED.items():
        out=ROOT/'runs'/ident;dest=(ROOT/tasks[ident]['complete']).parent
        assert json.loads((out/'submission.json').read_text())['job']==job
        assert not (dest/'complete.json').exists()
        assert '[Errno 121]' in (out/'child.stderr.log').read_text()
        for d in [out,dest]:
            if d.exists():
                f=(d/'worker.lock').open('a+b');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        p=dest/'predictions.jsonl';data=p.read_bytes() if p.exists() else b''
        rows=[json.loads(l) for l in data.splitlines()]
        assert len({(x['id'],x['arm']) for x in rows})==len(rows)
        assert all(x['status']=='ok' and x['rank']==128 and x['step']==8000 for x in rows)
        if ident==GATE:assert len(rows)==1809
        observed[ident]=dict(records=len(rows),sha256=hashlib.sha256(data).hexdigest(),old_parent_exit_exists=(out/'parent_exit.json').exists())
    pending=ROOT/'runs/large-final-m1-s2'
    assert not (pending/'submission.json').exists() and not (pending/'parent_exit.json').exists()
    assert json.loads((pending/'attempt_history/failed-106609/submission.json').read_text())['job']=='106609'
    completed=ROOT/'runs/large-final-m1-s3'
    wait=json.loads((completed/'parent_exit.json').read_text());assert wait['actual_wait'] and wait['returncode']==0 and wait['job']=='106640'
    preserved=(completed/'submission.json').read_bytes()
    judge=ROOT/'judge_gpt6_astra';assert '[Errno 121]' in (judge/'watch.stderr.log').read_text()
    votes=[json.loads(l) for l in (judge/'judge_decisions.jsonl').read_text().splitlines()]
    assert len(votes)==9433 and len({(x['cohort'],x['arm'],x['id']) for x in votes})==9433
    status=json.loads((judge/'status.json').read_text());assert status['available_complete'] and not status['errors']
    history.mkdir();dump(history/'intent.json',dict(at=now(),failed=FAILED,observed=observed,proofs=proofs,gate=GATE))
    (history/'sacct_original.txt').write_text(acct)
    (history/'effective_plan_before.json').write_bytes((ROOT/'effective_plan.json').read_bytes())
    for ident,job in FAILED.items():
        out=ROOT/'runs'/ident;archive=out/'attempt_history'/('failed-'+job);archive.mkdir(parents=True)
        for p in list(out.iterdir()):
            if p.is_file() and p.name!='worker.lock':p.rename(archive/p.name)
        dest=(ROOT/tasks[ident]['complete']).parent
        for name in ['failure.json','progress.json']:
            p=dest/name
            if p.exists():p.rename(archive/('evaluation_'+name))
        tasks[ident]['recovery_from_job']=job
    for i in range(3):
        t=tasks['large-final-m1-s%d'%i]
        assert GATE not in t['depends'];t['depends'].append(GATE)
        t['operational_recovery_gate']='Original unchanged 9B worker must validate all 1809 saved outputs and exit0; no new answers needed'
    assert (completed/'submission.json').read_bytes()==preserved
    for name in ['coordinator_launch.json','status.json','coordinator_failure.json','PAUSE_SUBMISSIONS.json']:
        p=ROOT/name
        if p.exists():p.rename(history/name)
    jhistory=history/'judge';jhistory.mkdir()
    for name in ['launch.json','watch_status.json','watch_failure.json','watch.stdout.log','watch.stderr.log']:
        p=judge/name
        if p.exists():p.rename(jhistory/name)
    dump(ROOT/'effective_plan.json',plan)
    dump(history/'recovery.json',dict(at=now(),failed=FAILED,observed=observed,gate=GATE,
        scientific_code_and_data_unchanged=True,no_9b_answer_regeneration=True,
        requirement='Original evaluator skips saved ids; actual parent_wait0 and Slurm0:0 before missing 27B shards admitted',
        judge_records_preserved=9433,new_api_retry_grants=0,completed_27b_shard3_preserved=True,proofs=proofs))
    for f in locks:f.close()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    with (ROOT/'coordinator.stdout.log').open('ab') as stdout,(ROOT/'coordinator.stderr.log').open('ab') as stderr:
        child=subprocess.Popen([PY,'-B','-u',str(ROOT/'control/coordinator.py')],cwd=str(ROOT),env=env,
             stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    launch=dict(pid=child.pid,at=now(),script=str(ROOT/'control/coordinator.py'),remote_root=str(ROOT),maximum_gpu_requests=4,
                recovery_evidence=str(history/'recovery.json'))
    dump(ROOT/'coordinator_launch.json',launch)
    print(json.dumps(dict(launch=launch,gate=GATE,observed=observed)))
if __name__=='__main__':main()
