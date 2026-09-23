"""Read current scheduler, owner and progress without importing model dependencies."""
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = r'''
import datetime,json,os,subprocess
from pathlib import Path
r=Path('/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918')
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
plan=read(r/'effective_plan.json')
status=read(r/'status.json') or {}
owner=read(r/'coordinator_launch.json') or {}
cmd=Path('/proc')/str(owner.get('pid',0))/'cmdline'
owner['process_alive']=cmd.exists() and b'control/coordinator.py' in cmd.read_bytes()
raw=subprocess.run(['squeue','-u','liuhanzuo','-h','-o','%i|%j|%T|%b|%R'],universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=20).stdout
jobs=[]
for line in raw.splitlines():
    ident,name,state,gres,reason=line.split('|',4)
    if name.startswith('qcm-q18-'):
        jobs.append(dict(job=ident,task=name[8:],state=state,gpus=int(gres.split(':')[-1]),reason=reason))
active=[]
for job in jobs:
    task=next(t for t in plan['tasks'] if t['id']==job['task'])
    out=r/'runs'/task['id']
    progress=read((r/task['complete']).parent/'progress.json')
    err=out/'child.stderr.log'
    tail=''
    if err.exists():
        with err.open('rb') as f:
            f.seek(max(0,err.stat().st_size-2500));tail=f.read().decode('utf-8','replace')
    active.append(dict(**job,progress=progress,start=read(out/'start.json'),parent_exit=read(out/'parent_exit.json'),stderr_tail=tail))
evidence=read(r/'maintenance_history/storage-recovery-20260918-1722/recovery.json')
judge={}
for name in ['launch.json','watch_status.json','watch_failure.json','progress.json','latest_service_check.json']:
    value=read(r/'judge_gpt6_astra'/name)
    if value is not None:judge[name]=value
launch=judge.get('launch.json')
if launch:
    process=Path('/proc')/str(launch['pid'])/'cmdline'
    launch['process_alive']=process.exists() and b'judge_watch.py' in process.read_bytes()
controller_disk=os.statvfs('/var/spool')
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    controller_available_bytes=controller_disk.f_bavail*controller_disk.f_frsize,
    submissions_paused=(r/'PAUSE_SUBMISSIONS.json').exists(),
    local_gpu_reservation=read(r/'local_gpu_reservation.json'),
    owner=owner,queue=jobs,own_gpu_requests=sum(j['gpus'] for j in jobs),
    phase=status.get('phase'),completed=status.get('completed'),failed=status.get('failed'),
    status_at=status.get('at'),active=active,recovery=evidence,judge=judge,plan=plan)))
'''


def main():
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1',
      'timeout -k 5s 30s /usr/bin/python3 -c ' + shlex.quote(CODE)],
      text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    local_recovery = ROOT / 'delivery/node_local_recovery_20260919_0630'
    if (local_recovery / 'recovery.json').exists():
        recovery = json.loads((local_recovery / 'recovery.json').read_text(encoding='utf-8'))
        mirror_path = local_recovery / 'monitor_state.json'
        mirror = json.loads(mirror_path.read_text(encoding='utf-8')) if mirror_path.exists() else {}
        by_job = {r['job']: r for r in mirror.get('records', [])}
        jobs = {r['job'] for r in recovery['tasks']}
        data['canonical_status_stale'] = not data['owner']['process_alive']
        data['node_local_recovery'] = dict(path=str(local_recovery), jobs=sorted(jobs),
            mirror_at=mirror.get('at'), instruction='Run monitor_node_local_recovery.py --tag 20260919_0630 first; shared canonical receipts remain failed historical attempts until delivery reconciles them.')
        for entry in data['active']:
            if entry['job'] not in jobs:
                continue
            entry['canonical_progress_ignored'] = entry['progress']
            observed = by_job.get(entry['job'], {})
            files = observed.get('files', {})
            entry['progress'] = files.get('evaluation_progress.json')
            entry['start'] = files.get('start.json')
            entry['parent_exit'] = files.get('parent_exit.json')
            entry['stderr_tail'] = observed.get('child.stderr.log', '')
            entry['progress_source'] = dict(kind='node-local mirror', at=mirror.get('at'),
                                            mirrored_records=observed.get('mirrored_records'))
    (ROOT / 'effective_plan.json').write_text(json.dumps(data.pop('plan'), indent=2) + '\n', encoding='utf-8')
    (ROOT / 'monitor_state.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'delivery/storage_recovery_20260918_1900.json').write_text(json.dumps(data['recovery'], indent=2) + '\n', encoding='utf-8')
    print(json.dumps(data, ensure_ascii=True))


if __name__ == '__main__':
    main()
