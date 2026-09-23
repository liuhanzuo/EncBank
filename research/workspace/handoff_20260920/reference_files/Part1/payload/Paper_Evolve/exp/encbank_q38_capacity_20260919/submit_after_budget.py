"""Prioritize 27B capacity after current deployment, then resume pending 8B costs."""
import json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;CONTROL='/tmp/qcm-q38-capacity-control-liuhanzuo-20260919'
def ssh(code,data=None):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(code)],input=json.dumps(data) if data else None,capture_output=True,encoding='utf8',timeout=45)
    if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout)
    return r.stdout
def main():
    assert not (ROOT/'submission.json').exists()
    with tarfile.open(ROOT/'payload.tar','w') as t:
        for p in ROOT.glob('*.py'):
            if p.name not in ['bootstrap.py','submit.py']:t.add(p,arcname=p.name)
        for n in ['plan.json','inputs.json','PROTOCOL_zh.md']:t.add(ROOT/n,arcname=n)
    ssh("import os;from pathlib import Path;p=Path("+repr(CONTROL)+");assert os.getuid()==20021 and p.exists() and not(p/'submission.json').exists()")
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(ROOT/'payload.tar'),'gpu-node1:'+CONTROL+'/payload.tar'],check=True,timeout=60)
    code=r'''
import json,subprocess,sys,datetime
from pathlib import Path
p=json.load(sys.stdin);root=Path('/tmp/qcm-q38-capacity-control-liuhanzuo-20260919')
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T'],universal_newlines=True)
assert not any('|qcm-q38-capacity|' in x for x in queue.splitlines())
old=subprocess.check_output(['scontrol','show','job','108619'],universal_newlines=True)
assert 'JobName=qcm-adaptive-b300' in old and 'JobState=PENDING' in old,old
subprocess.check_call(['scontrol','hold','108619'])
cmd=['sbatch','--parsable','--job-name=qcm-q38-capacity','--dependency=afterok:108559,singleton','--partition=gpu','--nodelist=gpu-node1','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1','--cpus-per-task=8','--mem=96G','--time=06:00:00','--chdir=/tmp','--output=/tmp/qcm-q38-capacity-%j.out','--error=/tmp/qcm-q38-capacity-%j.err']
script="#!/bin/bash\nset -euo pipefail\nexec /usr/bin/python3 -I -B -u - <<'CAPACITY_PY'\n"+p['bootstrap']+'\nCAPACITY_PY\n'
r=subprocess.run(cmd,input=script,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
record=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,queue_before=queue,budget_before=old,at=datetime.datetime.utcnow().isoformat()+'Z')
if r.returncode==0:
 job=r.stdout.strip().split(';')[0];assert job.isdigit();record.update(job=job,stage='/tmp/qcm-q38-capacity-liuhanzuo-'+job,node='gpu-node1')
 (root/'submission.json').write_text(json.dumps(record))
 subprocess.check_call(['scontrol','update','JobId=108619','Dependency=afterok:'+job+',singleton'])
 subprocess.check_call(['scontrol','release','108619'])
 record['budget_new_dependency']='afterok:'+job+',singleton'
else:subprocess.check_call(['scontrol','release','108619'])
(root/'submission.json').write_text(json.dumps(record));print(json.dumps(record));assert r.returncode==0
'''
    out=ssh(code,dict(bootstrap=(ROOT/'bootstrap.py').read_text()));record=json.loads(out)
    (ROOT/'submission.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['queue_before','budget_before','command']}))
if __name__=='__main__':main()
