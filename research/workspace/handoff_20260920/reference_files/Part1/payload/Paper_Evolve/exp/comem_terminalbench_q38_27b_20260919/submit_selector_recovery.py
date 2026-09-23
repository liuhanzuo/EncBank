"""Resume only unfinished selector arms after a verified failed single-GPU job."""
import json,shlex,subprocess,tarfile
from pathlib import Path
from submit import ssh
from io_utils import save
ROOT=Path(__file__).resolve().parent
CONTROL='/tmp/qcm-q38-tb-control-liuhanzuo-20260920-recovery'

def main():
    assert json.loads((ROOT/'selector_boundary_validation.json').read_text())['passed']
    assert json.loads((ROOT/'cpu_check.json').read_text())['passed']
    assert not (ROOT/'owner_registration.json').exists()
    for arm in ['iter_k12','iter_k48']:
        receipt=json.loads((ROOT/'receipts'/(arm+'.json')).read_text())
        assert receipt['actual_child_wait'] and receipt['exit_code']==0
        release=json.loads((ROOT/'receipts'/(arm+'.released.json')).read_text())
        assert len(release)==6 and any(not d['remaining_sessions'] for d in release.values())
    archive=ROOT/'infrastructure_attempts'/'20260920_bm25_empty_history'
    code="import json;from pathlib import Path;r=Path('/tmp/qcm-q38-tb-liuhanzuo-109054');print(json.dumps(dict(files={n:json.loads((r/n).read_text()) for n in ['parent_exit.json','failure.json','environment.json','probe_correctness.json'] if (r/n).exists()},events=(r/'events.jsonl').read_text(),selection_source=(r/'selection.py').read_text(),mailbox={p.name:json.loads(p.read_text()) for p in (r/'mailbox').glob('*.json')})))"
    evidence=json.loads(ssh(code));assert evidence['files']['parent_exit.json']['actual_wait'] and evidence['files']['parent_exit.json']['returncode']==1
    save(archive/'remote_job_evidence.json',evidence)
    with tarfile.open(ROOT/'payload_selector_recovery.tar','w') as tar:
        for path in ROOT.glob('*.py'):
            if path.name not in ['submit.py','submit_selector_recovery.py','bootstrap.py']:tar.add(path,arcname=path.name)
        for name in ['plan.json','README_zh.md']:tar.add(ROOT/name,arcname=name)
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',str(ROOT/'payload_selector_recovery.tar'),'gpu-node1:'+CONTROL+'/payload.tar'],check=True,timeout=90)
    bootstrap=(ROOT/'bootstrap.py').read_text().replace('/tmp/qcm-q38-tb-control-liuhanzuo-20260919',CONTROL)
    code=r'''
import datetime,json,os,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);control=Path('/tmp/qcm-q38-tb-control-liuhanzuo-20260920-recovery')
assert os.getuid()==20021
receipt=control/'submission.json'
if receipt.exists():
 print(receipt.read_text());raise SystemExit(0)
assert not(control/'intent.json').exists(),'Ambiguous prior sbatch intent: inspect instead of submitting twice'
old=subprocess.check_output(['sacct','-j','109054','-n','-P','-o','JobIDRaw,State,ExitCode'],universal_newlines=True)
assert '109054|FAILED|1:0' in old
queue=subprocess.check_output(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'],universal_newlines=True)
assert not any('|qcm-q38-tb-' in line for line in queue.splitlines()),'Own GPU campaign already active or queued'
command=['sbatch','--parsable','--job-name=qcm-q38-tb-recovery','--dependency=singleton','--partition=gpu','--nodelist=gpu-node1','--nodes=1','--ntasks=1','--gres=gpu:nvidia_l20d:1','--cpus-per-task=8','--mem=96G','--time=24:00:00','--chdir=/tmp','--output=/tmp/qcm-q38-tb-%j.out','--error=/tmp/qcm-q38-tb-%j.err']
script="#!/bin/bash\nset -euo pipefail\nexec /usr/bin/python3 -I -B -u - <<'QCM_PY'\n"+p['bootstrap']+'\nQCM_PY\n'
with (control/'intent.json').open('x') as stream:json.dump(dict(command=command,queue=queue,prior_job=old,at=datetime.datetime.utcnow().isoformat()+'Z'),stream,indent=2)
r=subprocess.run(command,input=script,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
record=dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,at=datetime.datetime.utcnow().isoformat()+'Z',recovery_from='109054',remaining_arms=['bm25_p90','iter48_qk_p90'])
if r.returncode==0:
 record['job']=r.stdout.strip().split(';')[0];assert record['job'].isdigit();record['stage']='/tmp/qcm-q38-tb-liuhanzuo-'+record['job'];record['node']='gpu-node1'
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))
assert r.returncode==0
'''
    record=json.loads(ssh(code,dict(bootstrap=bootstrap)))
    assert record['returncode']==0
    save(ROOT/'submission_selector_recovery.json',record)
    save(ROOT/'submission.json',record)
    print(json.dumps(record))

if __name__=='__main__':main()
