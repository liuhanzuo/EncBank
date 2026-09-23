"""Small CPU coordinator extension; no change to any scientific GPU process."""
import datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
    history=ROOT/'maintenance_history/add-local-reservation-20260919';assert not history.exists()
    admission=(ROOT.parent/'qcomem_gpu_admission.lock').open('a+b');fcntl.flock(admission,fcntl.LOCK_EX)
    launch=json.loads((ROOT/'coordinator_launch.json').read_text());pid=launch['pid'];proc=Path('/proc')/str(pid)/'cmdline'
    assert b'control/coordinator.py' in proc.read_bytes()
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists() or b'control/coordinator.py' not in proc.read_bytes():break
        time.sleep(.1)
    assert not proc.exists() or b'control/coordinator.py' not in proc.read_bytes()
    lock=(ROOT/'coordinator.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=ROOT/'control/coordinator.py';code=source.read_text();assert 'def local_gpu_reservation' not in code
    history.mkdir();(history/'coordinator_before.py').write_text(code);dump(history/'old_launch.json',launch)
    code=code.replace('def owned_jobs(plan):',"def local_gpu_reservation():\n    p=ROOT/'local_gpu_reservation.json'\n    if not p.exists():return 0\n    value=json.loads(p.read_text());assert value['gpus']==1 and value['token']\n    return 1\n\ndef owned_jobs(plan):")
    assert code.count('if used>=4:break')==1
    code=code.replace('if used>=4:break','if used+local_gpu_reservation()>=4:break')
    code=code.replace("task_count=len(plan['tasks']),tasks=states,submissions_paused=paused,", "task_count=len(plan['tasks']),tasks=states,submissions_paused=paused,local_gpu_reserved=local_gpu_reservation(),")
    compile(code,str(source),'exec');source.write_text(code)
    lock.close();admission.close()
    with (ROOT/'coordinator.stdout.log').open('ab') as out,(ROOT/'coordinator.stderr.log').open('ab') as err:
        child=subprocess.Popen([PY,'-B','-u',str(source)],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    fresh=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),script=str(source),
        remote_root=str(ROOT),maximum_gpu_requests=4,task_count=28,maintenance=str(history))
    dump(ROOT/'coordinator_launch.json',fresh);dump(history/'new_launch.json',fresh);print(json.dumps(fresh))
if __name__=='__main__':main()
