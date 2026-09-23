"""Server-only disjoint supplement: one allocation, three sequential model arms.

The existing 49/79/79 task controllers retain ownership. This entry point only
admits the two explicitly omitted tasks, with a frozen live-plan audit.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

B = Path('/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919')
S = B / 'server_control_20260920'
R = Path('/srv/encbank/qencbank_runtime_20260911/server_control_20260920')
OUT = S / 'supplement_20260921'
TASKS = ['regex-chess', 'vulnerable-secret']
ARMS = [('dense', 'dense_server_r6_20260920', '111876', 'dense_no_task_deadline_r5_20260920'),
        ('k12', 'encbank_k12_server_r6_20260920', '112400', 'encbank_k12_no_task_deadline_r6_20260920'),
        ('k48', 'encbank_k48_server_r6_20260920', '112403', 'encbank_k48_no_task_deadline_r6_20260920')]


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_ownership():
    rows = []
    for family, source, job, old in ARMS:
        p = B / old / 'plan.json'
        d = json.loads(p.read_text())
        assert not set(TASKS) & set(d['tasks']), (family, 'task already owned')
        partition = json.loads((S / 'runs' / source / 'predecessor_partition.json').read_text())
        completed = {x['task'] for x in partition.get('retained_normal', [])}
        assert not completed & set(TASKS)
        rows.append(dict(family=family, job_id=job, path=str(p), sha256=sha(p),
                         tasks=d['tasks'], retained_normal=sorted(completed)))
    return rows


def build(probe_job, secret_probe_job=None):
    assert not OUT.exists(), 'Never rebuild a prepared or used supplement'
    ownership = audit_ownership()
    proofs = []
    for task, suffix in zip(TASKS, ['regex', 'secret']):
        selected_job = secret_probe_job if suffix == 'secret' and secret_probe_job else probe_job
        p = R / ('managed-' + selected_job + '-' + suffix) / 'report.json'
        d = json.loads(p.read_text())
        assert d['task'] == task and d['status'] == 'BASIC_PASS' and d['cleanup_verified']
        assert d['managed_backend'] and d['exec_in_memory_cgroup']
        assert d['hostname'] == 'gpu-host'
        proofs.append(dict(task=task, report=str(p), sha256=sha(p)))
    OUT.mkdir()
    runs = []
    for family, source, job, old in ARMS:
        original = S / 'runs' / source
        name = family + '_supplement_20260921'
        target = OUT / name
        target.mkdir()
        manifest = json.loads((original / 'server_source_manifest.json').read_text())
        for file, digest in manifest.items():
            assert Path(file).name == file and sha(original / file) == digest
            if Path(file).suffix in {'.py', '.json', '.slurm'}:
                shutil.copy2(original / file, target / file)
        plan = json.loads((target / 'plan.json').read_text())
        old_root, old_ipc = plan['remote_root'], plan['ipc_root']
        plan.update(remote_root=str(target), tasks=TASKS, job_name='qencbank-tb-supplement-20260921',
                    selection='Two tasks explicitly omitted from every active allowlist; no outcome selection',
                    supplemental_only=True, predecessor_partition_reconciled=False,
                    evidence_id='E-TB21-SERVER-APPTAINER-SUPPLEMENT-' + family.upper() + '-20260921',
                    ipc_root=str(R.parent / ('t89sp21' + family)), concurrent_tasks=2,
                    host_memory_budget_mb=20480,
                    protocol_boundary='Server-only managed Apptainer series, recorded separately from legacy Docker',
                    resource_inventory=[dict(task=t, memory_mb=tomllib.loads((R/'tasks'/t/'task.toml').read_text())['environment']['memory_mb'],
                                             cpus=1) for t in TASKS])
        for key, leaf in [('rpc_root','rpc'),('results_root','results'),('controller_tmp','tmp'),('controller_cache','cache')]:
            plan[key] = str(R / name / leaf)
        plan.pop('handoff_barrier_jobs', None)
        save(target / 'plan.json', plan)
        template_path = target / (plan['arm'] + '_harbor_template.json')
        template = json.loads(template_path.read_text())
        template.update(job_name=name, jobs_dir=plan['results_root'], tasks=[{'path':str(R/'tasks'/t)} for t in TASKS])
        save(template_path, template)
        slurm = (target/'server.slurm').read_text().replace(old_root,str(target)).replace(old_ipc,plan['ipc_root'])
        slurm = '\n'.join(line for line in slurm.splitlines() if not line.startswith('#SBATCH --exclude=')) + '\n'
        (target/'server.slurm').write_text(slurm)
        save(target/'container_qualification.json',dict(status='PASS', scope='all_plan_task_environments',
             plan_sha256=sha(target/'plan.json'), tasks=TASKS, evidence=proofs,
             runtime_integration_report=str(R/'managed-integration-112454/report.json'),
             qualification='Per-image terminal, tmux, file transfer, network isolation, HTTPS, memory cgroup and cleanup',
             limitation='Environment compatibility only; no benchmark answers/verifiers executed; CPU affinity without CPU quota'))
        save(target/'server_source_manifest.json',{p.name:sha(p) for p in target.iterdir()
                                                  if p.is_file() and p.name!='server_source_manifest.json'})
        runs.append(str(target))
    save(OUT/'batch.json',dict(status='prepared', epoch=time.time(), tasks=TASKS, runs=runs,
                             ownership=ownership, server_only=True, automatic_scientific_retries=0))
    script = f'''#!/bin/bash
#SBATCH --job-name=qencbank-tb-supplement-20260921
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --no-requeue
#SBATCH --output={OUT}/slurm-%j.log
set -euo pipefail
umask 077
export LITELLM_LOCAL_MODEL_COST_MAP=True
{R}/harbor_env/bin/python {S}/supplement_20260921.py run
'''
    (OUT/'batch.slurm').write_text(script)
    print(json.dumps(dict(status='prepared',runs=runs)))


def submit():
    batch=json.loads((OUT/'batch.json').read_text())
    # CPU identity/import/container checks must succeed for every arm before GPU allocation.
    for name in batch['runs']:
        p=Path(name);plan=json.loads((p/'plan.json').read_text())
        command=[plan['harbor_python'],str(p/'server_preflight.py')]
        if plan['arm']=='encbank':command.append('--checkpoint')
        result=subprocess.run(command,cwd=p,capture_output=True,text=True,timeout=600)
        assert result.returncode==0,(name,result.stdout,result.stderr)
    lockroot=B.parent/'terminal_bench_20260918'
    with (lockroot/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (OUT/'submission.json').exists()
        assert audit_ownership()==batch['ownership'], 'Ownership changed; review before dispatch'
        queue=subprocess.check_output(['squeue','-r','-u','liuhanzuo','-h','-o','%i|%j|%T|%b'],text=True)
        own=[line for line in queue.splitlines() if any(x in line for x in ['qencbank-tb-','qencbank-agentmem-']) and 'gpu' in line.split('|')[-1].lower()]
        assert len(own)<4,own
        known={x[2] for x in ARMS}
        assert all(line.split('|')[0] in known for line in own), 'Unknown controller; refuse duplicate tasks'
        for run in batch['runs']:
            assert not (Path(run)/'execution').exists()
        record=dict(status='intent',epoch=time.time(),prior_queue=own,batch_sha256=sha(OUT/'batch.json'),server_only=True)
        save(OUT/'submission.json',record)
        result=subprocess.run(['sbatch','--parsable',str(OUT/'batch.slurm')],capture_output=True,text=True,timeout=30)
        record.update(status='submitted' if result.returncode==0 else 'failed',stdout=result.stdout,stderr=result.stderr,
                      exit_code=result.returncode,actual_parent_wait=True)
        if result.returncode==0:record['job_id']=result.stdout.strip().split(';')[0]
        save(OUT/'submission.json',record)
        assert result.returncode==0,result.stderr
        print(json.dumps(record,indent=2))


def run():
    assert os.environ.get('SLURM_JOB_ID')
    batch=json.loads((OUT/'batch.json').read_text())
    assert audit_ownership()==batch['ownership']
    receipt=OUT/'allocation_receipt.json'
    assert not receipt.exists()
    completed=[]
    for root in batch['runs']:
        p=Path(root)
        save(OUT/'status.json',dict(state='running',active_run=root,completed=completed,job_id=os.environ['SLURM_JOB_ID']))
        with (p/'allocation.stdout.log').open('xb') as out,(p/'allocation.stderr.log').open('xb') as err:
            result=subprocess.run(['bash',str(p/'server.slurm')],cwd=p,stdout=out,stderr=err)
        row=dict(run=root,exit_code=result.returncode,actual_parent_wait=True,epoch=time.time())
        completed.append(row)
        save(p/'batch_arm_receipt.json',row)
        # A failed arm is retained, never automatically replayed. Ensure GPU holder
        # closure before moving to a different arm within the same allocation.
        plan=json.loads((p/'plan.json').read_text())
        holder=p/('run_'+plan['arm'])
        if holder.exists():
            closure=json.loads((holder/'process_receipt.json').read_text())
            assert closure['actual_parent_wait']
        save(receipt,dict(state='running',arms=completed,job_id=os.environ['SLURM_JOB_ID']))
    save(receipt,dict(state='completed',arms=completed,job_id=os.environ['SLURM_JOB_ID'],server_only=True))
    save(OUT/'status.json',dict(state='completed',completed=completed,job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    assert sys.platform=='linux', 'All execution is server-only'
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['build','submit','run']);parser.add_argument('--probe-job');parser.add_argument('--secret-probe-job')
    args=parser.parse_args()
    if args.command=='build':build(args.probe_job,args.secret_probe_job)
    elif args.command=='submit':submit()
    else:run()
