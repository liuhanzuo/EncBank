"""Server-only admission of fixed, disjoint shards; never replay a submission or trial."""
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

U = Path(__file__).resolve().parent
D = U / 'scale4_20260921'
S = U.parent

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(v,indent=2)+'\n'); tmp.replace(p)

def main():
    lock=(D/'dispatcher.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry=read(D/'registry.json'); transfer=read(Path(registry['transfer_path']))
    runs={r['id']:r for r in registry['runs']}
    assert sha(Path(registry['transfer_path']))==registry['transfer_sha256']
    while True:
        pending=[rid for rid in registry['submission_order'] if not (Path(runs[rid]['root'])/'submission.json').exists()]
        for rid in registry['submission_order']:
            f=Path(runs[rid]['root'])/'submission.json'
            if f.exists():
                rec=read(f)
                assert rec['status']=='submitted' and rec.get('job_id'), 'Uncertain/failed submission requires review; no automatic retry'
        if not pending:
            save(D/'dispatcher_complete.json',dict(epoch=time.time(),all_shards_submitted=True,automatic_scientific_retries=0))
            return
        with (S.parent.parent/'terminal_bench_20260918/admission.lock').open('a') as admission:
            fcntl.flock(admission,fcntl.LOCK_EX)
            queue=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
            owned=[v for v in queue.splitlines() if any(n in v for n in ['qencbank-tb-','qencbank-agentmem-']) and 'gpu' in v.split('|')[-1].lower()]
            save(D/'dispatcher_status.json',dict(epoch=time.time(),pending_shards=pending,benchmark_gpu_queue=owned,gpu_limit=4))
            if len(owned)<4:
                rid=pending[0]; spec=runs[rid]; h=Path(spec['root']); p=read(h/'plan.json')
                assert not (h/'submission.json').exists() and not (h/'run_encbank').exists()
                assert read(h/'server_cpu_preflight.json')['status']=='PASS'
                assert sha(h/'server_source_manifest.json')==spec['source_manifest_sha256']
                for n,digest in read(h/'server_source_manifest.json').items(): assert sha(h/n)==digest,n
                old=Path(transfer['arms'][spec['arm']]['root'])
                assert (old/'execution/drain_requested.json').exists()
                assert not any((old/'execution/launches'/(t+'.json')).exists() for t in spec['tasks'])
                assert not any(p['job_name']==v.split('|')[1] for v in queue.splitlines())
                record=dict(status='intent',epoch=time.time(),arm=spec['arm'],shard=rid,root=str(h),
                    tasks=len(spec['tasks']),task_concurrency=8,gpu_requests=1,requested_nodes=None,
                    requested_time_limit='UNLIMITED',server_only=True,selection_sha256=registry['selection_sha256'],
                    source_manifest_sha256=sha(h/'server_source_manifest.json'),registry_sha256=sha(D/'registry.json'),
                    prior_owned_gpu_queue=owned,automatic_scientific_retries=0)
                save(h/'submission.json',record)
                child=subprocess.run(['sbatch','--parsable',str(h/'server.slurm')],capture_output=True,text=True)
                record.update(status='submitted' if child.returncode==0 else 'submission_failed',exit_code=child.returncode,
                              stdout=child.stdout,stderr=child.stderr,actual_parent_wait=True)
                if child.returncode==0:record['job_id']=child.stdout.strip().split(';')[0]
                save(h/'submission.json',record)
                assert child.returncode==0,record
                print(json.dumps(record),flush=True)
        time.sleep(20)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        save(D/'dispatcher_failure.json',dict(epoch=time.time(),error=repr(exc),automatic_retry=False))
        raise
