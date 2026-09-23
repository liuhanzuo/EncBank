"""One-shot recovery of FAILED107546; preserve actual failure and use local compile caches."""
import datetime,fcntl,json,os,subprocess
from pathlib import Path
ROOT=Path('/srv/encbank/comem_c3_c4_20260919')
def main():
    history=ROOT/'maintenance_history'/'eval-cache-20260919-1225'
    assert not history.exists(),'Recovery already attempted: inspect before doing anything else'
    old=json.loads((ROOT/'eval_owner_launch.json').read_text())
    proc=Path('/proc')/str(old['pid'])/'cmdline'
    assert not proc.exists() or b'eval_owner.py' not in proc.read_bytes(),'Old owner still active'
    lock=(ROOT/'eval_owner.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=ROOT/'runs/eval-j6.json';rec=json.loads(receipt.read_text());assert rec['job']=='107546'
    accounting=subprocess.check_output(['sacct','-j','107546','-n','-P','-o','JobIDRaw,State,ExitCode'],universal_newlines=True)
    assert '107546|FAILED|1:0' in accounting
    out=ROOT/'quality/j6_s42';wait=json.loads((out/'parent_exit.json').read_text());assert wait['actual_wait'] and wait['returncode']==1
    assert (out/'predictions.jsonl').stat().st_size==0 and not (out/'complete.json').exists()
    assert not any((ROOT/'runs'/('eval-j%d.json'%j)).exists() for j in [12,18])
    history.mkdir(parents=True)
    for f in [receipt,ROOT/'eval_owner_launch.json',ROOT/'eval_owner_failure.json']:
        f.rename(history/f.name)
    out.rename(history/'failed_quality_j6_s42')
    (history/'accounting.txt').write_text(accounting)
    (history/'recovery.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        failed_job='107546',failure='Triton cache cleanup os.removedirs EBUSY on BeeGFS',
        retained_predictions=0,change='only compilation/tmp caches -> per-job node-local /tmp',
        scientific_configuration_unchanged=True,other_gpu_jobs_preserved=True),indent=2))
    lock.close()
    r=subprocess.run(['/usr/bin/python3','-I','-B',str(ROOT/'start_eval_owner_remote.py')],universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert r.returncode==0,r.stderr
    (history/'launch_result.json').write_text(r.stdout);print(r.stdout)
if __name__=='__main__':main()
