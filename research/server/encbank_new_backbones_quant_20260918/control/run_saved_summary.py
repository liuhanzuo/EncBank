"""CPU-only saved-output summary with a detached parent that retains actual wait status."""
import argparse,datetime,fcntl,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
PY='/srv/encbank/Paper_Evolve/.venv/bin/python'
def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','run']);p.add_argument('--tag',required=True);a=p.parse_args()
    assert a.tag.replace('-','').isalnum()
    out=ROOT/'delivery'/'cpu_summary_runs'/a.tag;out.mkdir(parents=True,exist_ok=True)
    if a.action=='launch':
        with (out/'launch_intent.json').open('x') as f:json.dump(dict(at=datetime.datetime.utcnow().isoformat()+'Z'),f)
        with (out/'parent.stdout.log').open('ab') as stdout,(out/'parent.stderr.log').open('ab') as stderr:
            child=subprocess.Popen(['/usr/bin/python3',str(Path(__file__).resolve()),'run','--tag',a.tag],cwd=str(ROOT),
                stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        v=dict(pid=child.pid,tag=a.tag,path=str(out),cpu_only=True);dump(out/'launch.json',v);print(json.dumps(v));return
    lock=(ROOT/'delivery/cpu_summary.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'parent_exit.json').exists()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONPATH=str(ROOT)+':/srv/encbank/encbank_infra_recheck_20260912/deps:/srv/encbank/encbank_followups_20260913/Encbank:/srv/encbank/encbank_frozen_j12_20260912',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    cmd=[PY,'-B',str(ROOT/'summarize.py')]
    with (out/'stdout.log').open('ab') as stdout,(out/'stderr.log').open('ab') as stderr:
        child=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=stdout,stderr=stderr)
        dump(out/'child_start.json',dict(pid=child.pid,command=cmd));code=child.wait()
    if code==0:
        for name in ['summary.json','RESULTS_zh.md']:(out/name).write_bytes((ROOT/'delivery'/name).read_bytes())
    dump(out/'parent_exit.json',dict(actual_wait=True,returncode=code,child_pid=child.pid,cpu_only=True,at=datetime.datetime.utcnow().isoformat()+'Z'))
    assert code==0,'Inspect saved CPU summary error; no GPU rerun.'
if __name__=='__main__':main()
