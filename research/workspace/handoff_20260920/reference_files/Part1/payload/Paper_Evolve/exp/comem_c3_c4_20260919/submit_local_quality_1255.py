"""One-shot recovery of three verified failed quality jobs, <=3 remote GPUs."""
import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'delivery/node_local_quality_20260919_1255'
REMOTE=r'''
import datetime,fcntl,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);r=Path('/srv/encbank/comem_c3_c4_20260919')
control=Path('/tmp/qcm-c34-control-20021-20260919-1255')
assert os.getuid()==20021 and not control.exists()
f=Path('/proc/3454603/cmdline');assert not(f.exists() and b'eval_owner.py' in f.read_bytes())
locks=[]
for name in ['admission.lock','eval_owner.lock']:
 h=(r/name).open('rb');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=40).stdout
acct=run(['sacct','-j',','.join(t['previous_job'] for t in p['tasks']),'-n','-P','-o','JobIDRaw,State,ExitCode,End'])
for t in p['tasks']:
 assert any(l.startswith(t['previous_job']+'|FAILED|1:0|') for l in acct.splitlines())
 folder=r/'quality'/('j%d_s42'%t['j']);assert not(folder/'complete.json').exists()
 assert hashlib.sha256((folder/'predictions.jsonl').read_bytes()).hexdigest()==t['sha256']
control.mkdir(mode=0o700)
def dump(name,v):
 with (control/name).open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
dump('intent.json',dict(tasks=p['tasks'],sacct=acct))
(control/'bootstrap.py').write_text(p['script'])
receipts=[]
for t in p['tasks']:
 queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'])
 jobs=[l.split('|') for l in queue.splitlines() if l.split('|')[1].startswith(('qcm-c34-','qcm-q18-'))]
 assert sum(int(j[3].split(':')[-1]) for j in jobs)<3, jobs
 name='qcm-c34-local-j%d'%t['j'];assert all(j[1]!=name for j in jobs)
 args=' '.join(map(shlex.quote,['--j',str(t['j']),'--previous-job',t['previous_job'],'--saved-sha256',t['sha256']]))
 script='#!/bin/bash\nset -euo pipefail\nexec /usr/bin/python3 -I -B -u - '+args+" <<'QCM_PY'\n"+p['script']+'\nQCM_PY\n'
 command=['sbatch','--parsable','--job-name='+name,'--dependency=singleton','--partition=gpu','--nodelist='+t['node'],'--gres=gpu:nvidia_l20d:1','--cpus-per-task=4','--mem=96G','--time=12:00:00','--chdir=/tmp','--output=/tmp/qcm-c34-local-20021-%j.out','--error=/tmp/qcm-c34-local-20021-%j.err']
 dump('intent-j%d.json'%t['j'],dict(command=command,queue=jobs))
 res=subprocess.run(command,input=script,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40)
 rec=dict(t,command=command,returncode=res.returncode,stdout=res.stdout,stderr=res.stderr,at=datetime.datetime.utcnow().isoformat()+'Z')
 if res.returncode==0:
  rec['job']=res.stdout.strip().split(';')[0];assert rec['job'].isdigit();rec['stage']='/tmp/qcm-c34-quality-liuhanzuo-'+rec['job']
 dump('submission-j%d.json'%t['j'],rec);receipts.append(rec);assert res.returncode==0, rec
report=dict(control=str(control),tasks=receipts,canonical_owner_stopped=True,local_gpu_reserved=1,total_limit=4)
dump('recovery.json',report);print(json.dumps(report))
'''
def main():
    assert not OUT.exists(),'One-shot already attempted; inspect actual receipts before retry'
    snapshot=json.loads((ROOT/'delivery/storage_failure_20260919_1255/snapshot.json').read_text())
    payload=dict(tasks=snapshot['tasks'],script=(ROOT/'node_local_quality.py').read_text())
    OUT.mkdir();(OUT/'intent.json').write_text(json.dumps(payload['tasks'],indent=2))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(REMOTE)],input=json.dumps(payload),capture_output=True,encoding='utf-8',timeout=110)
    (OUT/'stdout.log').write_text(r.stdout,encoding='utf-8');(OUT/'stderr.log').write_text(r.stderr,encoding='utf-8')
    assert r.returncode==0,r.stderr
    (OUT/'recovery.json').write_text(json.dumps(json.loads(r.stdout),indent=2));print(r.stdout)
if __name__=='__main__':main()
