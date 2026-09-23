"""Retain live work, pause admissions, and capture the failed 04:18 recovery."""
import json,shlex,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
CODE=r'''
import datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path
r=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
record=dict(at=datetime.datetime.utcnow().isoformat()+'Z',reason='Recovery after two healthy checks failed again; inspect BeeGFS service before further retries',active_jobs_preserved=True,
    pause_file_saved=False,pause_file_error='Errno121 writing PAUSE_SUBMISSIONS.persistent-storage.tmp; no repeated write attempted')
owner=json.loads((r/'coordinator_launch.json').read_text());assert owner['pid']==3759947
p=Path('/proc')/str(owner['pid'])/'cmdline'
if p.exists() and str(r/'control/coordinator.py').encode() in p.read_bytes():
    os.kill(owner['pid'],signal.SIGTERM)
    for _ in range(40):
        if not p.exists() or not p.read_bytes():break
        time.sleep(.25)
assert not p.exists() or str(r/'control/coordinator.py').encode() not in p.read_bytes()
record['cpu_owner_stopped']=True
result=dict(pause=record,receipt={},errors={})
for task in ['large-final-m0-s3']+['large-final-m1-s%d'%i for i in range(4)]:
    d=r/'runs'/task;p=d/'submission.json'
    if not p.exists():continue
    sub=json.loads(p.read_text());parent=d/'parent_exit.json'
    result['receipt'][task]=dict(submission=sub,parent_exit=json.loads(parent.read_text()) if parent.exists() else None)
    p=d/'child.stderr.log'
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));result['errors'][task]=f.read().decode('utf-8','replace')
cmd=['sacct','-j',','.join(v['submission']['job'] for v in result['receipt'].values()),'-n','-P','-o','JobIDRaw,State,ExitCode,End']
result['accounting']=subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,check=True,timeout=20).stdout
cmd=['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%R']
result['own_queue']=[x for x in subprocess.run(cmd,universal_newlines=True,stdout=subprocess.PIPE,check=True,timeout=20).stdout.splitlines() if 'qcm-q18-' in x]
p=r/'judge_gpt6_astra/watch.stderr.log';result['judge_error']=p.read_text()[-4000:]
print(json.dumps(result))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','gpu-node1','timeout -k 5s 40s /usr/bin/python3 -c '+shlex.quote(CODE)],text=True,capture_output=True,timeout=50)
assert p.returncode==0,p.stderr
x=json.loads(p.stdout);out=r/'delivery/storage_failure_20260919_recurrence'
(out/'failed_recovery_0422.json').write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
state=out/'stability.json';old=json.loads(state.read_text())
(out/'stability_before_failed_recovery_0422.json').write_text(json.dumps(old,indent=2)+'\n')
old.update(healthy=False,consecutive_successes=0,eligible_for_inspected_recovery=False,checked_unix=time.time(),
    invalidated_reason='Actual jobs and Judge failed with Errno121 after generic health checks; no repeated automated retries based only on these probes')
state.write_text(json.dumps(old,indent=2)+'\n')
print(json.dumps(dict(pause=x['pause'],accounting=x['accounting'],queue=x['own_queue'],errors=x['errors'],judge_error=x['judge_error'])))
