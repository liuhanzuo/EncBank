"""Finalize only after verified capacity; no automatic duplicate model/benchmark runs."""
import json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text());CAP=Path(P['capacity_dir'])
def main():
    assert not (ROOT/'submission.json').exists()
    assert json.loads((CAP/'owner_status.json').read_text())['phase']=='COMPLETE'
    conditions=json.loads((CAP/'summary.json').read_text())['conditions']
    bound=min(x['largest_tested_successful_batch'] for x in conditions)
    assert bound>=8,'Review actual memory results before choosing a smaller service/session layout'
    P.update(decode_batch_size=8,concurrent_tasks=6,max_live_sessions=6,status='CAPACITY_CHECKED_CPU_FINALIZATION',
        capacity_evidence=str(CAP/'summary.json'),capacity_minimum_tested_bound=bound)
    (ROOT/'plan.json').write_text(json.dumps(P,indent=2))
    for arm in P['arms']:
        cfg=dict(job_name='qcm27b_'+arm,jobs_dir=P['local_wsl_root']+'/results',n_attempts=1,n_concurrent_trials=6,retry=dict(max_retries=0),
            environment=dict(type='docker'),agents=[dict(import_path='tb_agent:MemoryTerminus',model_name='Qwen3.8-27B/'+arm,
            kwargs=dict(enable_summarize=False,proactive_summarization_threshold=0,interleaved_thinking=True,collect_rollout_details=True,
                store_all_messages=True,record_terminal_session=False,reasoning_effort='xhigh',temperature=1.0))],
            tasks=[dict(path=P['tasks_root']+'/'+n) for n in P['tasks']])
        (ROOT/'harbor_configs'/(arm+'.json')).write_text(json.dumps(cfg,indent=2))
    command='cd '+shlex.quote(P['local_wsl_root'])+' && LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(P['host_runtime']+'/bin/python')+' -B cpu_check.py'
    check=subprocess.run(['wsl','-d','Ubuntu','--exec','/bin/bash','--noprofile','--norc','-c',command],capture_output=True,timeout=150)
    (ROOT/'cpu_check.stdout.log').write_bytes(check.stdout);(ROOT/'cpu_check.stderr.log').write_bytes(check.stderr)
    assert check.returncode==0,check.stderr.decode('utf8','replace')[-2000:]
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'submit.py')],cwd=ROOT)
    assert result.returncode==0
    print('Submitted 27B service. Start exactly one Windows owner in a hidden window.')
if __name__=='__main__':main()
