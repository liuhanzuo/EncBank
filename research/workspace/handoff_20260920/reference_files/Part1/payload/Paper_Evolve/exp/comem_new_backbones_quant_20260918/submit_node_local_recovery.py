"""One-shot admission of two inspected failures; node-local outputs avoid BeeGFS writes."""
import hashlib, json, shlex, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'delivery/node_local_recovery_20260919_0625'

REMOTE = r'''
import datetime,fcntl,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin)
r=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')
control=Path('/tmp/qcm-q18-control-20021-20260919-0625')
assert os.getuid()==20021 and not control.exists(), 'One-shot already attempted: inspect local controller receipts before any retry'
for pid,needle in [(1599472,'control/coordinator.py'),(2005194,'judge_watch.py')]:
 f=Path('/proc')/str(pid)/'cmdline';assert not (f.exists() and needle.encode() in f.read_bytes())
assert p['proof']['actual_returncode']==0 and p['proof']['node']=='gpu-host'
locks=[]
for f in [r/'coordinator.lock',r.parent/'qcomem_gpu_admission.lock']:
 handle=f.open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
def run(cmd):return subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=45).stdout
acct=run(['sacct','-j','106878,106879','-n','-P','-o','JobIDRaw,State,ExitCode,End'])
for job in ['106878','106879']:assert any(l.startswith(job+'|FAILED|1:0|') for l in acct.splitlines())
for item in p['tasks']:
 out=r/'runs'/item['task'];assert json.loads((out/'submission.json').read_text())['job']==item['previous_job']
 assert '[Errno 121]' in (out/'child.stderr.log').read_text()
 source=r/'results/large-final/Qwen3.8-27B'/('shard%d'%item['shard'])
 assert not (source/'complete.json').exists()
 assert hashlib.sha256((source/'predictions.jsonl').read_bytes()).hexdigest()==item['sha256']
control.mkdir(mode=0o700)
def dump(name,v):
 with (control/name).open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
dump('intent.json',dict(at=datetime.datetime.utcnow().isoformat()+'Z',tasks=p['tasks'],scientific_changes=False,proof=p['proof']))
(control/'bootstrap.py').write_text(p['script'])
receipts=[]
for item in p['tasks']:
 queue=run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'])
 jobs=[l.split('|') for l in queue.splitlines() if l.split('|')[1].startswith('qcm-q18-')]
 assert sum(int(j[3].split(':')[-1]) for j in jobs)<4
 name='qcm-q18-'+item['task'];assert all(j[1]!=name for j in jobs)
 args=' '.join(map(shlex.quote,['--shard',str(item['shard']),'--previous-job',item['previous_job'],'--saved-sha256',item['sha256']]))
 script='#!/bin/bash\nset -euo pipefail\nexec /usr/bin/python3 -u - '+args+" <<'QCM_PY'\n"+p['script']+'\nQCM_PY\n'
 command=['sbatch','--parsable','--job-name='+name,'--partition=gpu','--nodelist=gpu-node4','--gres=gpu:nvidia_l20d:1',
 '--cpus-per-task=4','--mem=128G','--time=72:00:00','--chdir=/tmp',
 '--output=/tmp/qcm-q18-local-20021-%j.out','--error=/tmp/qcm-q18-local-20021-%j.err']
 dump('submit-intent-s%d.json'%item['shard'],dict(command=command,queue=jobs,task=item))
 result=subprocess.run(command,input=script,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 rec=dict(item,command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
          at=datetime.datetime.utcnow().isoformat()+'Z',node='gpu-node4')
 if result.returncode==0:
  rec['job']=result.stdout.strip().split(';')[0];assert rec['job'].isdigit()
  rec['stage']='/tmp/qcm-q18-liuhanzuo-'+rec['job']
 dump('submission-s%d.json'%item['shard'],rec);receipts.append(rec)
 assert result.returncode==0, 'Submission not accepted; inspect controller receipt before retry'
report=dict(control=str(control),tasks=receipts,canonical_owner_stopped=True,canonical_receipts_preserved=True,
 total_limit=4,shared_outputs_pending=True)
dump('recovery.json',report)
print(json.dumps(report))
'''

def main():
    assert not OUT.exists(), 'Already attempted: inspect records before retry'
    source=ROOT/'delivery/storage_failure_20260919_0617'
    tasks=[]
    for shard,job in [(1,'106878'),(2,'106879')]:
        data=(source/('qwen38_shard%d_predictions.jsonl'%shard)).read_bytes()
        tasks.append(dict(task='large-final-m1-s%d'%shard,shard=shard,previous_job=job,
                          sha256=hashlib.sha256(data).hexdigest(),saved_records=len(data.splitlines())))
    payload=dict(tasks=tasks,script=(ROOT/'control/node_local_evaluation.py').read_text(encoding='utf-8'),
                 proof=json.loads((source/'node_local_stage_probe.json').read_text(encoding='utf-8')))
    OUT.mkdir()
    (OUT/'local_intent.json').write_text(json.dumps(tasks,indent=2),encoding='utf-8')
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1',
                           '/usr/bin/python3 -c '+shlex.quote(REMOTE)],input=json.dumps(payload),
                           capture_output=True,text=True,timeout=110)
    (OUT/'submission_stdout.txt').write_text(result.stdout,encoding='utf-8')
    (OUT/'submission_stderr.txt').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    (OUT/'recovery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
