"""Server-only, model-free container capability probe; never reads task solutions/tests."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import socket
import subprocess
import time
import tomllib
from pathlib import Path

ROOT = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')


def audit():
    rows = []
    for directory in sorted((ROOT / 'tasks').iterdir()):
        cfg = tomllib.loads((directory / 'task.toml').read_text())
        dockerfile = directory / 'environment/Dockerfile'
        lines = dockerfile.read_text().splitlines() if dockerfile.exists() else []
        rows.append(dict(task=directory.name, environment=cfg.get('environment', {}),
                         docker_directives=[line for line in lines if line.strip().split(' ', 1)[0].upper()
                                            in {'FROM', 'USER', 'WORKDIR', 'ENTRYPOINT', 'CMD', 'EXPOSE', 'VOLUME'}],
                         compose=any((directory / 'environment').glob('*compose*'))))
    return rows


async def probe(task, output, candidate=False, managed=False):
    from harbor.environments.singularity.singularity import SingularityEnvironment
    if candidate:
        import importlib.util
        spec = importlib.util.spec_from_file_location('comem_candidate', ROOT / 'container_backend_candidate/singularity.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SingularityEnvironment = module.SingularityEnvironment
    if managed:
        from apptainer_environment import ManagedApptainerEnvironment
        SingularityEnvironment = ManagedApptainerEnvironment
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=output / 'container.log', level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    for key, value in {'TMPDIR': ROOT / 'tmp', 'APPTAINER_CACHEDIR': ROOT / 'apptainer_cache',
                       'APPTAINER_TMPDIR': ROOT / 'tmp', 'SINGULARITY_CACHEDIR': ROOT / 'apptainer_cache'}.items():
        os.environ[key] = str(value)
    directory = ROOT / 'tasks' / task
    cfg = tomllib.loads((directory / 'task.toml').read_text())
    report = dict(status='RUNNING', task=task, hostname=socket.gethostname(),
                  slurm_job=os.getenv('SLURM_JOB_ID'), model_calls=0, benchmark_attempts=0,
                  image=cfg['environment']['docker_image'], candidate_backend=candidate, managed_backend=managed,
                  start=time.time(), checks={})
    environment = SingularityEnvironment(environment_dir=directory / 'environment', environment_name=task,
        session_id=output.name, trial_paths=TrialPaths(trial_dir=output / 'trial'),
        task_env_config=EnvironmentConfig.model_validate(cfg['environment']),
        singularity_image_cache_dir=ROOT / 'sif', singularity_no_mount='home,tmp,bind-paths',
        logger=logging.getLogger('container_probe'))
    async def command(name, cmd):
        result = await environment.exec(cmd, timeout_sec=30)
        report['checks'][name] = result.model_dump()
        assert result.return_code == 0, (name, result)
    try:
        await asyncio.wait_for(environment.start(force_build=False), timeout=600)
        await command('terminal', 'id; pwd; command -v bash; command -v tmux; command -v asciinema')
        await command('write', 'mkdir -p /opt/comem-probe; printf server-only > /opt/comem-probe/value')
        source = output / 'upload.txt'
        source.write_text('container-transfer-check\n')
        await environment.upload_file(source, '/opt/comem-probe/upload.txt')
        await environment.download_file('/opt/comem-probe/upload.txt', output / 'download.txt')
        assert source.read_bytes() == (output / 'download.txt').read_bytes()
        report['checks']['file_roundtrip'] = 'PASS'
        await command('tmux', "tmux -L comem-probe new-session -d -s check 'sleep 10'; tmux -L comem-probe has-session -t check; tmux -L comem-probe kill-server")
        await command('home_isolation', 'test ! -e /srv/encbank/qcomem_runtime_20260911/models')
        # Generic capability checks, not task solutions or verifier execution.
        for name, cmd in {
            'switch_uid': "su nobody -s /bin/sh -c 'id -u'",
            'network_namespace': 'readlink /proc/self/ns/net',
            'cgroup': 'cat /proc/self/cgroup',
        }.items():
            result = await environment.exec(cmd, timeout_sec=30)
            report['checks'][name] = result.model_dump()
        report['host_network_namespace'] = os.readlink('/proc/self/ns/net')
        if managed:
            await command('internet', 'python3 -c \'import urllib.request; print(urllib.request.urlopen("https://example.com",timeout=20).status)\'')
            assert report['checks']['network_namespace']['stdout'].strip() != report['host_network_namespace']
            report['service'] = json.loads((environment._root/'service.json').read_text())
            report['exec_in_memory_cgroup'] = report['service']['cgroup'] in report['checks']['cgroup']['stdout']
            assert report['exec_in_memory_cgroup']
        report['status'] = 'BASIC_PASS'
        report['full_benchmark_qualified'] = False
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = repr(error)
    finally:
        await environment.stop(delete=True)
        report['launcher_stopped'] = environment._server_process is None or environment._server_process.returncode is not None
        report['cleanup_verified'] = False  # Apptainer can daemonize children outside the launcher tree.
        if managed and hasattr(environment,'_root'):
            report['closure'] = json.loads((environment._root/'closure.json').read_text())
            service_path = environment._root/'service.json'
            service = json.loads(service_path.read_text()) if service_path.exists() else {}
            pid = service.get('instance_pid')
            report['instance_pid_gone'] = pid is None or not Path('/proc',str(pid)).exists()
            report['cleanup_verified'] = report['instance_pid_gone'] and report['closure']['service_state'] in {'inactive','unknown','failed'}
            if not report['cleanup_verified']:
                report['status']='FAIL'
        report['end'] = time.time()
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--managed', action='store_true')
    args = parser.parse_args()
    assert os.name == 'posix', 'Run on the server only'
    if args.task:
        assert args.task in {row['task'] for row in audit()}
        args.output.resolve().relative_to(ROOT)
        asyncio.run(probe(args.task, args.output, args.candidate, args.managed))
    else:
        print(json.dumps(audit(), indent=2))
