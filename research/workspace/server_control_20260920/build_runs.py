"""Assemble three new server-side runs without changing the preserved R5 sources."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

H = Path(__file__).resolve().parent
REMOTE = '/srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920'
RUNTIME = '/srv/encbank/qencbank_runtime_20260911/server_control_20260920'
SOURCES = H.parent / 'handoff_20260920/source_increment/qencbank/paper_autonomous_multifork_iteration/evidence/terminal_bench_full89_20260919'
COMMON = ['server_owner.py', 'server_transport.py', 'host_admission.py', 'runtime_identity.py',
          'server_job.py', 'server_preflight.py', 'deploy.py', 'observe.py']
WORKER = ['agent_worker.py', 'holder.py', 'live_mailbox.py', 'persistence.py', 'storage_preflight.py',
          'model_path_preflight.py', 'model_recovery_manifest.json', 'cpu_preflight_remote.py',
          'startup_wait.py', 'control_plane_recovery.py', 'task_manifest.json',
          'toolchain_guard.py', 'ipc_path_guard.py', 'capacity_admission.py',
          'common.py', 'hybrid_reader.py', 'session.py', 'memory_selectors.py',
          'batch_cache.py', 'batch_cache_refill.py', 'service_loop.py']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def replace(path, old, new):
    text = path.read_text(encoding='utf-8')
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new), encoding='utf-8', newline='\n')


def build(backend='managed'):
    for arm, previous, name, job in [
        ('dense', 'dense_no_task_deadline_r5_20260920', 'dense_server_r6_20260920', '111876'),
        ('encbank', 'encbank_k12_no_task_deadline_r5_20260920', 'encbank_k12_server_r6_20260920', '111884'),
        ('encbank', 'encbank_k48_no_task_deadline_r5_20260920', 'encbank_k48_server_r6_20260920', '111886'),
    ]:
        source = SOURCES / previous
        output = H / 'runs' / name
        if (output / 'submission.json').exists() or (output / 'execution').exists():
            raise RuntimeError('Refuse to rebuild a used run: ' + str(output))
        output.mkdir(parents=True, exist_ok=True)
        for file in COMMON:
            shutil.copy2(H / file, output / file)
        if backend == 'managed':
            for file in ['apptainer_environment.py', 'apptainer_service.py', 'apptainer_executor.py']:
                shutil.copy2(H / file, output / file)
        for file in WORKER:
            if (source / file).exists():
                shutil.copy2(source / file, output / file)
        shutil.copy2(source / 'partition_final.json', output / 'predecessor_partition.json')
        plan = json.loads((source / 'plan.json').read_text())
        runtime = RUNTIME + '/' + name
        plan.update(remote_root=REMOTE + '/runs/' + name,
                    task_root=RUNTIME + '/tasks', rpc_root=runtime + '/rpc',
                    results_root=runtime + '/results', controller_tmp=runtime + '/tmp',
                    controller_cache=runtime + '/cache', container_cache=RUNTIME + '/apptainer_cache',
                    sif_cache=RUNTIME + '/sif', harbor_python=RUNTIME + '/harbor_env/bin/python',
                    host_admission_root=RUNTIME + '/host_admission', host_memory_budget_mb=20480,
                    container_backend=backend, docker_host='unix:///var/run/docker.sock',
                    server_only=True, predecessor_root=str(PurePosixPath(REMOTE).parent / previous),
                    predecessor_job_id=job, job_name='qencbank-tb-' + name.replace('_20260920', '').replace('_', '-'),
                    predecessor_partition_reconciled=(arm == 'encbank'), model_layers=64,
                    transport='Authenticated cluster-local HTTP; durable server-side request/response records',
                    deployment_boundary='Linux controller and task environments; no client-computer dependency',
                    container_protocol_change=(backend != 'docker'),
                    ipc_root='/srv/encbank/qencbank_runtime_20260911/' +
                             {'dense': 't89sr6d', 'encbank_k12': 't89sr6k12', 'encbank_k48': 't89sr6k48'}[
                                 'dense' if arm == 'dense' else ('encbank_k12' if plan['top_k_chunks'] == 12 else 'encbank_k48')])
        if backend == 'managed':
            plan.update(container_runtime='Managed Apptainer instance + per-task user service + slirp4netns',
                        container_cpu_enforcement='affinity_only_no_cgroup_cpu_quota',
                        container_memory_enforcement='per_task_service_cgroup_memory_max',
                        container_result_series='server_apptainer_not_pooled_with_legacy_docker',
                        container_qualified_nodes=['gpu-node1'],container_qualified_hostnames=['gpu-host'])
            if arm == 'encbank':
                plan['handoff_barrier_jobs']=['112400' if plan['top_k_chunks']==12 else '112403']
                plan['predecessor_partition_reconciled']=False
        if arm == 'encbank':
            plan['training_model_identity'] = dict(name='Qwen3.8-27B', j=21, L=64,
                revision=plan['model_revision'],
                path='/srv/encbank/encbank_new_backbones_20260915/models/Qwen3.8-27B')
            replace(output / 'agent_worker.py',
                    "cfg=MODELS[1];assert cfg['j']==P['j'] and cfg['path']==P['model']",
                    "from runtime_identity import resolve_configs\n    training_cfg,cfg=resolve_configs(P,MODELS[1])")
            replace(output / 'agent_worker.py', 'load_state(reader,ckpt,cfg);del ckpt;model.requires_grad_(False)',
                    'resolve_configs(P,training_cfg,ckpt)\n    load_state(reader,ckpt,training_cfg);del ckpt;model.requires_grad_(False)')
        save(output / 'plan.json', plan)
        rpc = (source / 'tb_agent_rpc.py').read_text(encoding='utf-8')
        rpc = '\n'.join("RPC=Path(P['rpc_root'])" if line.startswith('RPC=Path(') else line for line in rpc.splitlines()) + '\n'
        start, end = rpc.index('def save(p,d):'), rpc.index('class MailboxLLM')
        rpc = rpc[:start] + 'from server_transport import save\n' + rpc[end:]
        # A controller death must close the task as infrastructure failure, not leave an unbounded mailbox wait.
        rpc = rpc.replace("if err.exists():raise RuntimeError(json.loads(err.read_text())['error'])",
                          "if err.exists():raise RuntimeError(json.loads(err.read_text())['error'])\n                if time.monotonic()-began>P['per_call_transport_watchdog_seconds']+240:raise TimeoutError('Server controller response watchdog; no request replay')")
        (output / 'tb_agent_rpc.py').write_text(rpc, encoding='utf-8')
        template = json.loads((source / (arm + '_harbor_template.json')).read_text())
        template.update(job_name=name, jobs_dir=plan['results_root'])
        template['tasks'] = [{'path': str(Path(plan['task_root']) / task).replace('\\', '/')} for task in plan['tasks']]
        template['environment'] = {'type': 'singularity' if backend == 'managed' else backend}
        if backend in {'singularity', 'managed'}:
            template['environment']['kwargs'] = {'singularity_image_cache_dir': plan['sif_cache'],
                                                'singularity_no_mount': 'home,tmp,bind-paths'}
        if backend == 'managed':
            template['environment']['import_path'] = 'apptainer_environment:ManagedApptainerEnvironment'
        save(output / (arm + '_harbor_template.json'), template)
        if (output / 'ipc_path_guard.py').exists():
            replace(output / 'ipc_path_guard.py',
                    "expected=Path('/srv/encbank/qencbank_runtime_20260911/t89dnlim5').resolve()",
                    "expected=Path(json.loads(Path('plan.json').read_text())['ipc_root']).resolve()")
        slurm = (source / (arm + '.slurm')).read_text()
        slurm = slurm.replace(str(PurePosixPath(REMOTE).parent / previous), plan['remote_root'])
        slurm = slurm.replace('qencbank-tb-dense-nolimit-r5-codex', plan['job_name'])
        slurm = '\n'.join('#SBATCH --job-name=' + plan['job_name'] if line.startswith('#SBATCH --job-name=') else line for line in slurm.splitlines()) + '\n'
        if backend == 'managed':
            # A pinned node must not also remain in a predecessor's exclusion list.
            slurm = '\n'.join(line for line in slurm.splitlines() if not line.startswith('#SBATCH --exclude=')) + '\n'
            slurm = slurm.replace('#SBATCH --job-name='+plan['job_name'],
                                  '#SBATCH --job-name='+plan['job_name']+'\n#SBATCH --nodelist=gpu-node1')
        # All task containers and model process share one managed allocation.
        slurm = '\n'.join('#SBATCH --cpus-per-task=16' if line.startswith('#SBATCH --cpus-per-task=') else line for line in slurm.splitlines()) + '\n'
        slurm = slurm.replace('/srv/encbank/qencbank_runtime_20260911/t89dnlim5', plan['ipc_root'])
        slurm = slurm.replace(' -u holder.py', ' -u server_job.py')
        slurm = slurm.replace('set -euo pipefail', 'set -euo pipefail\numask 077')
        if 'VLLM_RPC_BASE_PATH=' not in slurm:
            # Encbank has no vLLM sockets, but its temporary files still need the registered server location.
            slurm = slurm.replace('set -euo pipefail\numask 077',
                                  'set -euo pipefail\numask 077\nexport TMPDIR=' + plan['ipc_root'] + '\nmkdir -p "$TMPDIR"')
        (output / 'server.slurm').write_text(slurm, encoding='utf-8', newline='\n')
        mailbox = output / 'live_mailbox.py'
        mailbox.write_text(mailbox.read_text(encoding='utf-8').replace('ssh-node-memory-v1', 'cluster-http-memory-v1').replace('local Harbor parents closed', 'server Harbor parents closed'), encoding='utf-8', newline='\n')
        manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()
                    if path.is_file() and path.name != 'server_source_manifest.json'}
        save(output / 'server_source_manifest.json', manifest)
        print(name, len(plan['tasks']), 'tasks;', len(manifest), 'files;', backend)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['docker', 'singularity', 'managed'], default='managed')
    build(parser.parse_args().backend)
