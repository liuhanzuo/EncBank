"""Single admission owner, using the working Slurm route; no automatic resubmit."""
import datetime,fcntl,json,subprocess
from pathlib import Path
ROOT=Path('/srv/encbank/comem_c3_c4_20260919')
def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
    for d in ['logs','runs','cache','training']:(ROOT/d).mkdir(exist_ok=True)
    with (ROOT/'admission.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads(Path('/srv/encbank/comem_sparse_slurm_20260912/models/Qwen3-8B/config.json').read_text())
        assert cfg['num_hidden_layers']==36 and cfg['model_type']=='qwen3'
        for j in [6,12,18]:
            p=ROOT/'runs'/('train-j%d.json'%j)
            if p.exists():continue
            queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],universal_newlines=True,timeout=20)
            own=[l for l in queue.splitlines() if '|qcm-c34-' in l or '|qcm-q18-' in l]
            used=sum(int(l.split('|')[3].split(':')[-1]) for l in own)
            assert used+1<4,('Three remote training GPUs plus one reserved local serving GPU',own)
            name='qcm-c34-train-j%d'%j
            assert not any('|'+name+'|' in l for l in own)
            cmd=['sbatch','--parsable','--job-name='+name,'--dependency=singleton',
                 '--partition=gpu','--gres=gpu:nvidia_l20d:1','--cpus-per-task=4','--mem=96G','--time=48:00:00',
                 '--output='+str(ROOT/'logs'/('train-j%d-%%j.out'%j)),
                 '--error='+str(ROOT/'logs'/('train-j%d-%%j.err'%j)),
                 '--chdir='+str(ROOT),str(ROOT/'train_job.sh'),str(j)]
            rec=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,queue_before=own,local_gpu_reserved=1)
            dump(p,rec)
            r=subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
            rec.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
            if r.returncode==0:
                rec['job']=r.stdout.strip().split(';')[0];assert rec['job'].isdigit()
            dump(p,rec);assert r.returncode==0,rec
            print(json.dumps(dict(j=j,job=rec['job'])),flush=True)
if __name__=='__main__':main()
