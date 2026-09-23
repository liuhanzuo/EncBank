"""User narrowed scope: withdraw only the new ctrl-* experiments, preserve prior tasks."""
import datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
NEW=Path(__file__).resolve().parent
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
    hist=ROOT/'maintenance_history/withdraw-serving-depth-20260918';assert not hist.exists()
    admission=(ROOT.parent/'qencbank_gpu_admission.lock').open('a+b');fcntl.flock(admission,fcntl.LOCK_EX)
    launch=json.loads((ROOT/'coordinator_launch.json').read_text());pid=launch['pid']
    proc=Path('/proc')/str(pid)/'cmdline';assert b'control/coordinator.py' in proc.read_bytes()
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or b'control/coordinator.py' not in proc.read_bytes():break
        time.sleep(.1)
    assert not proc.exists() or b'control/coordinator.py' not in proc.read_bytes()
    lock=(ROOT/'coordinator.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    current=json.loads((ROOT/'effective_plan.json').read_text())
    removed=[t for t in current['tasks'] if t['id'].startswith('ctrl-')];assert len(removed)==44
    ids={t['id'] for t in removed};queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T'],universal_newlines=True,timeout=30)
    cancelled=[]
    for line in queue.splitlines():
        job,name,state=line.split('|')
        if name.startswith('qcm-q18-ctrl-'):
            task=name[len('qcm-q18-'):];assert task in ids
            receipt=json.loads((ROOT/'runs'/task/'submission.json').read_text());assert receipt['job']==job
            subprocess.check_call(['scancel',job],timeout=30);cancelled.append(dict(job=job,task=task))
    original=json.loads((ROOT/'maintenance_history/add-serving-depth-20260918/effective_plan.json').read_text())
    assert len(original['tasks'])==28 and not any(t['id'] in ids for t in original['tasks'])
    hist.mkdir();(hist/'expanded_plan.json').write_bytes((ROOT/'effective_plan.json').read_bytes())
    dump(hist/'withdrawal.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Explicit new user scope: no new depth training or production concurrency; BGE matched TTFT plus existing logs only.',
        removed_tasks=list(sorted(ids)),cancelled_jobs=cancelled,queue_before=queue,previous_owner=launch))
    dump(ROOT/'effective_plan.json',original);dump(NEW/'WITHDRAWN_BY_USER.json',dict(withdrawn=True,tasks=44,
        cancelled_jobs=cancelled,evidence=str(hist),reason='New user request supersedes previous two additions.'))
    lock.close();admission.close()
    with (ROOT/'coordinator.stdout.log').open('ab') as out,(ROOT/'coordinator.stderr.log').open('ab') as err:
        child=subprocess.Popen([PY,'-B','-u',str(ROOT/'control/coordinator.py')],cwd=str(ROOT),
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    fresh=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_root=str(ROOT),
        maximum_gpu_requests=4,script=str(ROOT/'control/coordinator.py'),task_count=28,scope_change=str(hist))
    dump(ROOT/'coordinator_launch.json',fresh);dump(hist/'new_launch.json',fresh);print(json.dumps(fresh))
if __name__=='__main__':main()
