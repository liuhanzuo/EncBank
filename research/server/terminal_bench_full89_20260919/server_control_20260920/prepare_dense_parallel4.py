"""Prepare four-way Dense continuation for five never-started tasks."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import tarfile
import time
import tomllib

B = pathlib.Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919')
S = B / 'server_control_20260920'
R = pathlib.Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')
H = S / 'dense_parallel4_20260921'
OLD = S / 'supplement_20260921/dense_supplement_20260921'
TASKS = ['mcmc-sampling-stan', 'mteb-leaderboard', 'rstan-to-pystan',
         'torch-pipeline-parallelism', 'torch-tensor-parallelism']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')


def main():
    assert not H.exists(), 'Never overwrite a prepared or used run'
    state = subprocess.check_output(['sacct', '-X', '-j', '113514', '--format=State', '-n', '-P'], text=True).strip()
    assert state.startswith('CANCELLED'), state
    old_events = B/'dense_no_task_deadline_r6_20260921/run_dense/events.jsonl'
    started = {e['task'] for line in old_events.read_text().splitlines()
               if (e := json.loads(line))['event'] == 'request_start'}
    assert not set(TASKS) & started
    archive = pathlib.Path('/srv/encbank/COMem_Migration_20260920/QCOMEM_PROGRESS_EVIDENCE_20260921_1130.tar.gz')
    with tarfile.open(archive) as tar:
        name = next(n for n in tar.getnames() if n.endswith('/closed_verification.json'))
        verification = json.load(tar.extractfile(name))
    normals = verification['arms']['dense']['retained_normal']
    # Schema-independent containment check on the retained task metadata.
    assert all(task not in json.dumps(normals) for task in TASKS)
    H.mkdir()
    manifest = json.loads((OLD/'server_source_manifest.json').read_text())
    for name, digest in manifest.items():
        f = OLD/name
        assert sha(f) == digest, name
        shutil.copy2(f, H/name)
    for name in ['apptainer_environment.py', 'apptainer_service.py', 'apptainer_executor.py']:
        shutil.copy2(S/name, H/name)
    for name in ['qualify_dense_parallel4.py', 'submit_dense_parallel4.py']:
        shutil.copy2(S/name, H/name)
    p = json.loads((H/'plan.json').read_text())
    oldroot, oldipc = p['remote_root'], p['ipc_root']
    runtime = R/H.name
    p.update(evidence_id='E-TB21-DENSE-PARALLEL4-20260921', tasks=TASKS,
             remote_root=str(H), concurrent_tasks=4, host_memory_budget_mb=40960,
             task_walltime_seconds=None, gpu_allocation_walltime_seconds=None,
             predecessor_job_id='113514', predecessor_root=str(B/'dense_no_task_deadline_r6_20260921'),
             predecessor_partition_reconciled=True, supplemental_only=False,
             selection='Five never-started tasks; exclude all normal outcomes, context boundaries and active supplement tasks',
             job_name='qcomem-tb-dense-parallel4', ipc_root=str(R.parent/'t89dp4'),
             host_admission='Four tasks maximum; original8GiB each,40GiB server reservations plus2GiB fresh headroom',
             timeout_protocol='No cumulative task or Slurm allocation deadline; original12000s request fault guard retained',
             authorization='User explicitly permits up to4 GPUs and Dense concurrency4 or8; begin at4',
             excluded_import_stall_node=None,
             prior_import_stall_note='Historical gpu-node1 import stall; require fresh no-CUDA engine import qualification before launch')
    p['resource_inventory'] = []
    for task in TASKS:
        cfg = tomllib.loads((R/'tasks'/task/'task.toml').read_text())
        assert cfg.get('agent', {}).get('timeout_sec') is None
        p['resource_inventory'].append(dict(task=task, memory_mb=cfg['environment']['memory_mb'], cpus=cfg['environment']['cpus']))
    for key, child in [('rpc_root','rpc'),('results_root','results'),('controller_tmp','tmp'),('controller_cache','cache')]:
        p[key] = str(runtime/child)
    save(H/'plan.json',p)
    template = json.loads((H/'dense_harbor_template.json').read_text())
    template.update(job_name=H.name, jobs_dir=p['results_root'],tasks=[{'path':str(R/'tasks'/t)} for t in TASKS])
    save(H/'dense_harbor_template.json',template)
    env = (H/'apptainer_environment.py').read_text()
    assert env.count("'-p','RuntimeMaxSec=36h',") == 1
    (H/'apptainer_environment.py').write_text(env.replace("'-p','RuntimeMaxSec=36h',",''))
    # Keep task validity separate from the Harbor parent process return code.
    owner = (H/'server_owner.py').read_text()
    marker = '                    host_admission.release(H.name, name)'
    assert owner.count(marker) == 1
    inserted = '''                    summaries=[]
                    for result_path in results:
                        result=json.loads(result_path.read_text())
                        exception=result.get('exception_info') or {}
                        normal=bool(requests) and result.get('verifier_result') is not None and not exception
                        summaries.append(dict(path=str(result_path),normal_completed=normal,
                            exception_type=exception.get('exception_type'),exception_message=exception.get('exception_message'),
                            verifier_result=result.get('verifier_result')))
                    save(O/'task_outcomes'/(name+'.json'),dict(task=name,model_requests=len(requests),results=summaries,
                        valid_result=(len(summaries)==1 and summaries[0]['normal_completed']),actual_parent_wait=True))
'''
    (H/'server_owner.py').write_text(owner.replace(marker, inserted+marker))
    slurm = (H/'server.slurm').read_text().replace(oldroot,str(H)).replace(oldipc,p['ipc_root'])
    slurm = slurm.replace('#SBATCH --time=36:00:00','#SBATCH --time=0').replace('#SBATCH --job-name=qcomem-tb-dense-server-r6','#SBATCH --job-name='+p['job_name'])
    (H/'server.slurm').write_text(slurm)
    # These old reports are copied metadata, never used as qualification for this plan.
    for name in ['container_qualification.json','server_preflight.json','server_cpu_preflight.json']:
        if (H/name).exists(): (H/name).rename(H/('historical_'+name))
    save(H/'selection_proof.json',dict(epoch=time.time(), predecessor_job='113514', predecessor_state=state,
         cancellation_reason='CANCELLED by0; actual operator/cause unknown; no normal worker exit receipt',
         no_inference_started_for_selected_tasks=True, selected_tasks=TASKS,
         old_events_sha256=sha(old_events), old_tasks_with_requests=sorted(started),
         latest_verified_archive=str(archive), archive_sha256=sha(archive),
         active_supplement_excluded=['regex-chess','vulnerable-secret'],
         all_prior_results_preserved=True, checkpoint_downloaded=False))
    # Validate model/engine imports and all five real environment setups without inference.
    qualify=f'''#!/bin/bash
#SBATCH --job-name=qcomem-dense-p4-qualification
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --output={H}/qualification-%j.log
set -euo pipefail
umask 077
cd {H}
export PYTHONPATH={H} LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
export TMPDIR={runtime}/tmp APPTAINER_CACHEDIR={R}/apptainer_cache APPTAINER_TMPDIR={runtime}/tmp
mkdir -p "$TMPDIR"
{p['harbor_python']} {H}/qualify_dense_parallel4.py
'''
    (H/'qualification.slurm').write_text(qualify)
    source_files=[f for f in H.iterdir() if f.is_file() and f.name!='server_source_manifest.json']
    save(H/'server_source_manifest.json',{f.name:sha(f) for f in source_files})
    result=subprocess.run(['sbatch','--parsable',str(H/'qualification.slurm')],capture_output=True,text=True)
    save(H/'qualification_submission.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time(),model_calls=0))
    assert result.returncode==0,result.stderr
    print(json.dumps(dict(prepared=str(H),qualification_job=result.stdout.strip(),task_concurrency=4,model_calls=0)))


if __name__=='__main__':main()
