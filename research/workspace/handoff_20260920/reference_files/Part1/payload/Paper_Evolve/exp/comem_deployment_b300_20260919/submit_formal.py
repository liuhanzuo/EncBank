"""One-shot single-card admission; inspect receipts rather than blindly retry."""
import json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONTROL='/tmp/qcm-deploy-control-liuhanzuo-20260919-formal'
def ssh(code,payload=None):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','/usr/bin/python3 -I -B -c '+shlex.quote(code)],input=json.dumps(payload) if payload is not None else None,capture_output=True,encoding='utf-8',timeout=45)
    assert r.returncode==0,r.stderr;return r.stdout
def main():
    assert json.loads((ROOT/'submission.json').read_text())['job']=='108460'
    with tarfile.open(ROOT/'payload.tar','w') as tar:
        for p in ROOT.glob('*.py'):
            if p.name not in ['submit.py','bootstrap.py']:tar.add(p,arcname=p.name)
        for name in ['workloads.json','PROTOCOL_zh.md']:tar.add(ROOT/name,arcname=name)
        package=Path('F:/Paper_Evolve/exp/comem_followups_20260913/COMem/comem')
        for p in package.rglob('*.py'):
            if '__pycache__' not in p.parts:tar.add(p,arcname=str(Path('comem')/p.relative_to(package)))
    print(ssh("import os;from pathlib import Path;p=Path('"+CONTROL+"');assert os.getuid()==20021 and not p.exists();p.mkdir(mode=0o700);print(str(p))"),flush=True)
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(ROOT/'payload.tar'),'gpu-node1:'+CONTROL+'/payload.tar'],check=True,timeout=90)
    code=r'''
import datetime,json,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);control=Path('/tmp/qcm-deploy-control-liuhanzuo-20260919-formal')
assert not(control/'submission.json').exists()
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'],universal_newlines=True)
assert not any('|qcm-deploy-b300|' in l for l in queue.splitlines())
command=['sbatch','--parsable','--job-name=qcm-deploy-b300','--dependency=singleton','--partition=gpu','--nodelist=gpu-node1','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1','--cpus-per-task=8','--mem=96G','--time=12:00:00','--chdir=/tmp','--output=/tmp/qcm-deploy-%j.out','--error=/tmp/qcm-deploy-%j.err']
script="#!/bin/bash\nset -euo pipefail\nexec /usr/bin/python3 -I -B -u - <<'QCM_PY'\n"+p['bootstrap']+'\nQCM_PY\n'
(control/'intent.json').write_text(json.dumps(dict(command=command,queue=queue,at=datetime.datetime.utcnow().isoformat()+'Z'),indent=2))
r=subprocess.run(command,input=script,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
record=dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,at=datetime.datetime.utcnow().isoformat()+'Z')
if r.returncode==0:
 record['job']=r.stdout.strip().split(';')[0];assert record['job'].isdigit();record['stage']='/tmp/qcm-deploy-liuhanzuo-'+record['job'];record['node']='gpu-node1'
(control/'submission.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
assert r.returncode==0
'''
    text=ssh(code,dict(bootstrap=(ROOT/'bootstrap_formal.py').read_text()));record=json.loads(text)
    (ROOT/'submission.json').write_text(json.dumps(record,indent=2));print(text)
if __name__=='__main__':main()
