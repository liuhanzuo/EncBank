"""Submit from gpu-node1. No client-side process is started or required."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from server_preflight import check, check_source
from server_transport import save

H = Path(__file__).resolve().parent


def predecessor_closed(plan):
    if plan.get('handoff_barrier_jobs'):
        for job in plan['handoff_barrier_jobs']:
            barrier=subprocess.check_output(['sacct','-X','-j',job,'--noheader','-P','--format=JobID,State'],text=True)
            state=[line.split('|')[1] for line in barrier.splitlines() if line.startswith(job+'|')]
            assert len(state)==1 and state[0].split()[0] in {'FAILED','COMPLETED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'}, 'A newer controller is still active: '+barrier
        reconciliation=json.loads((H/'successor_reconciliation.json').read_text())
        assert reconciliation['status']=='PASS' and reconciliation['all_old_controllers_closed']
        assert set(reconciliation['jobs'])==set(plan['handoff_barrier_jobs'])
        assert reconciliation['plan_sha256']==hashlib.sha256((H/'plan.json').read_bytes()).hexdigest()
        assert set(reconciliation['remaining_tasks'])==set(plan['tasks'])
    output = subprocess.check_output(['sacct', '-X', '-j', plan['predecessor_job_id'], '--noheader', '-P',
                                      '--format=JobID,State,ExitCode'], text=True)
    rows = [line.split('|') for line in output.splitlines() if line.startswith(plan['predecessor_job_id'] + '|')]
    assert len(rows) == 1 and rows[0][1].split()[0] in {'FAILED', 'COMPLETED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'}, output
    receipt = Path(plan['predecessor_root']) / ('run_' + plan['arm']) / 'process_receipt.json'
    assert receipt.exists() and json.loads(receipt.read_text())['actual_parent_wait']
    if plan['arm'] == 'dense':
        # R5 was active at handoff. Its old 49-task allowlist must not be replayed wholesale.
        record = json.loads((H / 'predecessor_reconciliation.json').read_text())
        assert record['status'] == 'PASS' and record['predecessor_job_id'] == plan['predecessor_job_id']
        assert record['all_old_controllers_closed'] and record['all_task_parents_accounted']
        assert record['plan_sha256'] == hashlib.sha256((H / 'plan.json').read_bytes()).hexdigest()
        assert set(record['remaining_tasks']) == set(plan['tasks'])
    else:
        failure = Path(plan['predecessor_root']) / 'run_encbank/worker_failure.json'
        assert 'AssertionError' in json.loads(failure.read_text())['error']
        assert not (Path(plan['predecessor_root']) / 'run_encbank/worker_ready.json').exists()
    return rows[0]


def submit():
    assert sys.platform == 'linux'
    plan = json.loads((H / 'plan.json').read_text())
    preflight = check(check_container=True, checkpoint=(plan['arm'] == 'encbank'))
    save(H / 'server_preflight.json', preflight)
    root = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_20260918')
    with (root / 'admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_source()
        prior = predecessor_closed(plan)
        assert not (H / 'submission.json').exists() and not (H / ('run_' + plan['arm'])).exists()
        queue = subprocess.check_output(['squeue', '-r', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%b'], text=True)
        own = [line for line in queue.splitlines() if 'qencbank-agentmem-' in line or 'qencbank-tb-' in line]
        gpu_own=[line for line in own if 'gpu' in line.split('|')[-1].lower()]
        family='dense' if plan['arm']=='dense' else 'k'+str(plan['top_k_chunks'])
        assert not any(family in line.split('|')[1] for line in gpu_own), 'Another GPU controller owns this experiment arm: '+repr(gpu_own)
        assert len(gpu_own) < 4 and not any(plan['job_name'] in line for line in own), own
        record = {'status': 'intent', 'epoch': time.time(), 'prior_queue': own, 'predecessor': prior,
                  'source_manifest_sha256': preflight['source_manifest_sha256'], 'server_only': True}
        save(H / 'submission.json', record)
        result = subprocess.run(['sbatch', '--parsable', str(H / 'server.slurm')], capture_output=True, text=True, timeout=30)
        record.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr, actual_parent_wait=True)
        if result.returncode == 0:
            record.update(status='submitted', job_id=result.stdout.strip().split(';')[0])
        save(H / 'submission.json', record)
        assert result.returncode == 0, result.stderr
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'submit'])
    args = parser.parse_args()
    if args.command == 'submit':
        submit()
    else:
        print(json.dumps(check(), indent=2))
