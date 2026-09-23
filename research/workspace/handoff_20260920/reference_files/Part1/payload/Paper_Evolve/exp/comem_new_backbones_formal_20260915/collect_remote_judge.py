"""Read remote Judge progress; never launch a worker or call a model."""
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import collections,datetime,json,pathlib
root=pathlib.Path('/srv/encbank/comem_new_backbones_formal_20260915')
out=root/'judge_gpt6_astra'/'formal_new_models'
files={}
for name in ('worker_status.json','progress.json','status.json','protocol.json',
             'transport_calibration.json','remote_worker_launch.json'):
    p=out/name
    if p.exists(): files[name]=json.loads(p.read_text())
worker=files.get('worker_status.json',{})
pid=worker.get('pid')
cmd=pathlib.Path('/proc')/str(pid)/'cmdline'
running=cmd.exists() and b'locomo_judge_remote_worker.py' in cmd.read_bytes()
counts=collections.Counter()
records=0
decisions=out/'judge_decisions.jsonl'
if decisions.exists():
    with decisions.open() as f:
        for line in f:
            if not line.endswith('\n'): continue
            row=json.loads(line)
            counts[row['judge_raw']]+=1
            records+=1
print(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      worker_running=running,files=files,decision_records=records,label_counts=dict(counts))))
'''


def main():
    command = '/srv/encbank/Paper_Evolve/.venv/bin/python -c ' + shlex.quote(REMOTE)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'gpu-node1', command], capture_output=True, text=True,
                            encoding='utf-8', check=True, timeout=40)
    snapshot = json.loads(result.stdout)
    out = ROOT / 'judge_gpt6_astra' / 'formal_new_models' / 'remote_status'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in snapshot['files'].items():
        (out / name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    (out / 'snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n', encoding='utf-8')
    path = ROOT / 'judge_status.json'
    state = json.loads(path.read_text())
    backup = ROOT / 'judge_gpt6_astra' / 'judge_status_before_remote_resume.json'
    if not backup.exists(): backup.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    worker = snapshot['files'].get('worker_status.json', {})
    progress = snapshot['files'].get('progress.json', {})
    calibration = snapshot['files'].get('transport_calibration.json', {})
    phase = worker.get('phase', 'UNKNOWN')
    if 'failure' in state: state['previous_failure'] = state.pop('failure')
    state.update(status=phase, execution_host='gpu-node1', execution_user='liuhanzuo',
                 worker_pid=worker.get('pid'), worker_running=snapshot['worker_running'],
                 available_records=27804, generation_complete=True,
                 formal_decisions=snapshot['decision_records'],
                 semantic_decisions=sum(snapshot['label_counts'].get(k, 0) for k in ('CORRECT', 'WRONG')),
                 local_abstention_decisions=snapshot['label_counts'].get('LOCAL_ABSTENTION_RULE', 0),
                 failed_requests=progress.get('errors', 0), last_observed_at=snapshot['at'],
                 current_connection_ok=bool(calibration.get('passed')) and not progress.get('errors'),
                 calibration_passed=calibration.get('passed', False),
                 worker_status='judge_gpt6_astra/formal_new_models/remote_status/worker_status.json',
                 next_action='Monitor the existing server-local worker; do not start the old Windows Judge. Collect and verify all 27804 decisions before reporting full LoCoMo scores.')
    path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    if calibration.get('passed') and 'protocol.json' in snapshot['files']:
        (ROOT / 'judge_protocol.json').write_text(json.dumps(snapshot['files']['protocol.json'], indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(at=snapshot['at'], phase=phase, worker_running=snapshot['worker_running'],
                         decisions=snapshot['decision_records'], available_records=27804,
                         semantic_decisions=state['semantic_decisions'],
                         local_abstention_decisions=state['local_abstention_decisions'],
                         errors=state['failed_requests'], calibration_passed=state['calibration_passed'])))


if __name__ == '__main__':
    main()
