"""Bounded, read-only stdlib snapshot; suitable for the cluster's Python 3.6."""
import datetime,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OWNER=Path('/srv/encbank/qcomem_align_codex_20260911/comem_new_backbones_quant_20260918')

def get(p):return json.loads(p.read_text()) if p.exists() else None

def main():
    state=get(OWNER/'status.json') or {};plan=get(ROOT/'submitted_plan.json') or {'tasks':[]}
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_pid=state.get('pid'),
        all_gpu_requests=state.get('total_owned_gpu_requests'),tasks={},owned_queue=state.get('owned_queue'))
    for t in plan['tasks']:
        status=(state.get('tasks') or {}).get(t['id'],{})
        complete=Path(t['complete']);entry=dict(status=status,completion=get(complete))
        progress=complete.parent/'progress.json'
        entry['progress']=get(progress)
        entry['parent_exit']=get(OWNER/'runs'/t['id']/'parent_exit.json')
        err=OWNER/'runs'/t['id']/'child.stderr.log'
        if err.exists() and status.get('state') in ('FAILED','OUT_OF_MEMORY','INVALID_COMPLETION'):
            with err.open('rb') as f:f.seek(max(0,err.stat().st_size-5000));entry['stderr_tail']=f.read().decode(errors='replace')
        result['tasks'][t['id']]=entry
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
