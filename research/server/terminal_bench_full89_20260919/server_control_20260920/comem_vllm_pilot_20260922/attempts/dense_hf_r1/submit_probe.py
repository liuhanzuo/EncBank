"""One authorized extra GPU; separate from the four production benchmark GPUs."""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path

H=Path(__file__).resolve().parent
def main():
    with (H/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (H/'submission.json').exists(),'Do not resubmit an existing or uncertain intent'
        q=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%N'],text=True)
        lines=q.splitlines()
        benchmark=[l for l in lines if ('qcomem-tb-' in l or 'qcomem-agentmem-' in l) and 'gpu' in l.split('|')[3]]
        probes=[l for l in lines if 'qcomem-kernel-' in l]
        assert len(benchmark)<=4 and not probes,(benchmark,probes)
        record=dict(status='intent',epoch=time.time(),authorization='User 2026-09-22 explicitly requests Dense Transformers TPS versus CoMem; existing authorization permits one separate pilot GPU.',
            benchmark_queue=benchmark,extra_gpu_limit=1,production_gpu_limit=4,requested_gpu=1,requested_nodes=None,
            requested_excluded_nodes=['gpu-node3'],requested_time_limit='UNLIMITED',scientific_retries=0,source_manifest=json.loads((H/'source_manifest.json').read_text()))
        for n,d in record['source_manifest'].items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==d,n
        with (H/'submission.json').open('x') as f:json.dump(record,f,indent=2)
        p=subprocess.run(['sbatch','--parsable',str(H/'probe.slurm')],text=True,capture_output=True)
        record.update(status='submitted' if p.returncode==0 else 'submission_failed',stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,actual_parent_wait=True)
        if p.returncode==0:record['job_id']=p.stdout.strip().split(';')[0]
        (H/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record));assert p.returncode==0

if __name__=='__main__':main()
