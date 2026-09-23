"""Single extra pilot GPU, protected intent and no automatic submission retries."""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent
with (H.parent.parent/'pilot_admission.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert not (H/'submission.json').exists(),'Existing or uncertain intent; do not duplicate'
    assert json.loads((H/'cpu_preflight.json').read_text())['status']=='PASS'
    manifest=json.loads((H/'source_manifest.json').read_text())
    for n,d in manifest.items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==d,n
    q=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%N'],text=True)
    rows=q.splitlines()
    benchmark=[l for l in rows if ('qcomem-tb-' in l or 'qcomem-agentmem-' in l) and 'gpu' in l.split('|')[3]]
    pilots=[l for l in rows if 'qcomem-kernel-' in l]
    assert len(benchmark)<=4 and not pilots,(benchmark,pilots)
    record=dict(status='intent',epoch=time.time(),authorization=json.loads((H/'protocol.json').read_text())['authorization'],
        benchmark_queue=benchmark,extra_gpu_limit=1,production_gpu_limit=4,requested_gpu=1,requested_nodes=None,
        requested_excluded_nodes=['gpu-node1','gpu-node3','gpu-node6'],requested_time_limit='UNLIMITED',automatic_retries=0,
        source_manifest_sha256=hashlib.sha256((H/'source_manifest.json').read_bytes()).hexdigest())
    with (H/'submission.json').open('x') as f:json.dump(record,f,indent=2)
    p=subprocess.run(['sbatch','--parsable',str(H/'probe.slurm')],text=True,capture_output=True)
    record.update(status='submitted' if p.returncode==0 else 'submission_failed',stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,actual_parent_wait=True)
    if p.returncode==0:record['job_id']=p.stdout.strip().split(';')[0]
    (H/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record));assert p.returncode==0
