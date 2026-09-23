"""Build a new six-attempt recovery, retaining the failed Apptainer attempts."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

B=Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919')
S=B/'server_control_20260920'
OUT=S/'docker_supplement_r2_20260921'
CODE=S/'docker_hybrid_code_20260921'
R=Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')
LOCAL=Path('/srv/encbank/client/comem_local_20260921')
TASKS=['regex-chess','vulnerable-secret']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    assert not OUT.exists(),'Do not rebuild a used recovery'
    prior=S/'supplement_20260921'
    receipt=json.loads((prior/'allocation_receipt.json').read_text())
    assert receipt['state']=='completed' and all(x['actual_parent_wait'] for x in receipt['arms'])
    proofs=[]
    for family in ['dense','k12','k48']:
        old=prior/(family+'_supplement_20260921')
        for task in TASKS:
            p=old/'execution/receipts'/(task+'.json');d=json.loads(p.read_text())
            assert d['actual_parent_wait'] and d['requests']==0
            assert len(d['result_paths'])==1
            f=Path(d['result_paths'][0]);result=json.loads(f.read_text())
            assert result.get('verifier_result') is None and result.get('exception_info')
            proofs.append(dict(family=family,task=task,receipt=str(p),receipt_sha256=sha(p),result=str(f),result_sha256=sha(f),classification='PRE_AGENT_ENVIRONMENT_FAILURE',model_requests=0))
    OUT.mkdir()
    for family in ['dense','k12','k48']:
        old=prior/(family+'_supplement_20260921');new=OUT/family;new.mkdir()
        manifest=json.loads((old/'server_source_manifest.json').read_text())
        for name,digest in manifest.items():
            assert sha(old/name)==digest,name
            if name.endswith(('.py','.json','.slurm')):shutil.copy2(old/name,new/name)
        p=json.loads((new/'plan.json').read_text());oldroot=p['remote_root'];oldipc=p['ipc_root']
        p.update(remote_root=str(new),local_root=str(LOCAL/'runs'/family),server_only=False,
                 hybrid_execution='Local Ubuntu-24.04 Docker and controller; GPU model and checkpoint stay on server',
                 container_backend='docker',container_protocol_change=False,
                 container_result_series='local_docker_20260921',container_cpu_enforcement='docker_cpu_quota',
                 container_memory_enforcement='docker_cgroup_memory_max',container_runtime='Docker Engine 29.8.1',
                 host_memory_budget_mb=12288,host_admission_root=str(LOCAL/'host_admission'),
                 task_root=str(LOCAL/'tasks'),harbor_python='/srv/encbank/client/comem_harbor_023/bin/python',
                 rpc_root=str(LOCAL/'rpc'/family),results_root=str(LOCAL/'results'/family),
                 controller_tmp=str(LOCAL/'tmp'/family),controller_cache=str(LOCAL/'cache'/family),
                 container_cache=str(LOCAL/'cache'),sif_cache=str(LOCAL/'unused_sif'),
                 ipc_root=str(R.parent/('t89dh21'+family)),job_name='qcomem-tb-docker-supplement-r2',
                 recovery_from_job='112585',recovery_classification='Six pre-agent environment failures; zero model requests',
                 transport='Verified immutable reply via existing OpenSSH identity; no inference retry',
                 protocol_boundary='Frozen model/sampling/context/task inputs; local Docker replaces failed Apptainer task environment')
        save(new/'plan.json',p)
        templatefile=new/(p['arm']+'_harbor_template.json');template=json.loads(templatefile.read_text())
        proxy='http://192.0.2.1:7897'
        template.update(jobs_dir=p['results_root'],job_name='docker-supplement-'+family,
                        tasks=[{'path':str(LOCAL/'tasks'/t)} for t in TASKS],
                        environment=dict(type='docker',env={'http_proxy':proxy,'https_proxy':proxy,'HTTP_PROXY':proxy,'HTTPS_PROXY':proxy,
                                                           'NO_PROXY':'localhost,127.0.0.1,::1'},cpu_enforcement_policy='strict',memory_enforcement_policy='strict'))
        save(templatefile,template)
        for name in ['file_bridge.py','hybrid_transport.py']:shutil.copy2(CODE/name,new/name)
        owner=(old/'server_owner.py').read_text()
        owner=owner.replace('from server_transport import Transport, save','from server_transport import save\nfrom hybrid_transport import Transport')
        owner=owner.replace("assert sys.platform == 'linux' and os.environ.get('SLURM_JOB_ID')","assert sys.platform == 'linux'")
        owner=owner.replace("assert H == Path(P['remote_root']).resolve()","assert H == Path(P['local_root']).resolve()")
        owner=owner.replace("os.environ['SLURM_JOB_ID']","json.loads((H.parents[1]/'submission.json').read_text())['job_id']")
        owner=owner.replace("'server_only': True","'server_only': False")
        owner=owner.replace("'state': 'draining' if DRAIN else 'server_tasks_running'","'state': 'draining' if DRAIN else 'local_docker_tasks_running'")
        # Preserve process closure separately from scientific validity. Harbor can
        # exit zero even when its trial contains an environment exception.
        marker="                    host_admission.release(H.name, name)"
        insertion="""                    summaries=[]
                    for result_path in results:
                        result=json.loads(result_path.read_text())
                        exception=result.get('exception_info') or {}
                        normal=bool(requests) and result.get('verifier_result') is not None and not exception
                        summaries.append(dict(path=str(result_path),normal_completed=normal,
                            exception_type=exception.get('exception_type'),exception_message=exception.get('exception_message'),
                            verifier_result=result.get('verifier_result')))
                    save(O/'task_outcomes'/(name+'.json'),dict(task=name,model_requests=len(requests),results=summaries,
                        valid_result=(len(summaries)==1 and summaries[0]['normal_completed']),actual_parent_wait=True))
"""
        assert owner.count(marker)==1
        owner=owner.replace(marker,insertion+marker)
        owner=owner.replace("    save(O / 'owner_complete.json',", "    save(O / 'status.json', {'state':'closed', 'completed':completed, 'pending':[r['task'] for r in pending], 'epoch':time.time()})\n    save(O / 'owner_complete.json',")
        (new/'hybrid_owner.py').write_text(owner)
        slurm=(old/'server.slurm').read_text().replace(oldroot,str(new)).replace(oldipc,p['ipc_root'])
        slurm=slurm.replace(' -u server_job.py',' -u holder.py')
        slurm='\n'.join(x for x in slurm.splitlines() if not x.startswith('#SBATCH '))+'\n'
        # Verify the frozen source on the allocated host before creating a worker.
        slurm=slurm.replace('umask 077','umask 077\n/srv/encbank/qcomem_runtime_20260911/python312/bin/python '+str(OUT/'remote_batch.py')+' verify '+family)
        (new/'gpu.sh').write_text(slurm)
        # The old Apptainer qualification is historical, never an admission input here.
        save(new/'source_manifest.json',{f.name:sha(f) for f in new.iterdir() if f.is_file() and f.name not in {'source_manifest.json','server_source_manifest.json'}})
    for name in ['remote_batch.py','local_batch.py']:shutil.copy2(CODE/name,OUT/name)
    save(OUT/'recovery_evidence.json',dict(prior_job='112585',tasks=TASKS,rows=proofs))
    slurm=f'''#!/bin/bash
#SBATCH --job-name=qcomem-tb-docker-supplement-r2
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-node3
#SBATCH --gres=gpu:nvidia_l20d:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --no-requeue
#SBATCH --output={OUT}/slurm-%j.log
set -euo pipefail
umask 077
/srv/encbank/qcomem_runtime_20260911/python312/bin/python {OUT}/remote_batch.py run
'''
    (OUT/'batch.slurm').write_text(slurm)
    with tarfile.open(OUT/'local_bundle.tar.gz','w:gz') as tar:
        tar.add(OUT/'local_batch.py',arcname='local_batch.py')
        for family in ['dense','k12','k48']:
            for f in (OUT/family).iterdir():
                if f.is_file():tar.add(f,arcname='runs/'+family+'/'+f.name)
        for task in TASKS:tar.add(R/'tasks'/task,arcname='tasks/'+task)
        tar.add(R/'task_copy_manifest.json',arcname='task_copy_manifest.json')
    save(OUT/'bundle_sha256.json',dict(sha256=sha(OUT/'local_bundle.tar.gz'),bytes=(OUT/'local_bundle.tar.gz').stat().st_size))
    print(json.dumps(dict(status='prepared',path=str(OUT),bundle_sha256=sha(OUT/'local_bundle.tar.gz'))))


if __name__=='__main__':main()
