"""Validate server paths, frozen tasks, container access, and checkpoint identity without a GPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

H = Path(__file__).resolve().parent
BOUND = Path('/srv/encbank').resolve()


def check_source():
    manifest = json.loads((H / 'server_source_manifest.json').read_text())
    for name, digest in manifest.items():
        assert Path(name).name == name
        assert hashlib.sha256((H / name).read_bytes()).hexdigest() == digest, name
    return hashlib.sha256((H / 'server_source_manifest.json').read_bytes()).hexdigest()


def docker_server_version(env):
    # docker info --format can return exit 0 plus an empty version on permission denial.
    result = subprocess.run(['docker', 'version', '--format', '{{json .Server}}'],
                            capture_output=True, text=True, timeout=30, env=env)
    server = json.loads(result.stdout) if result.stdout.strip() else None
    assert result.returncode == 0 and isinstance(server, dict) and server.get('Version'), \
        'Server Docker is unavailable: ' + result.stderr
    return server['Version']


def check(check_container=True, checkpoint=False):
    assert sys.platform == 'linux'
    source_sha = check_source()
    plan = json.loads((H / 'plan.json').read_text())
    assert plan['server_only'] and str(H) == plan['remote_root']
    for key in ['remote_root', 'task_root', 'rpc_root', 'results_root', 'controller_tmp',
                'controller_cache', 'container_cache', 'sif_cache', 'harbor_python',
                'engine_python', 'model', 'host_admission_root', 'ipc_root']:
        path = Path(plan[key])
        assert path.is_absolute() and '\\' not in plan[key] and '/mnt/' not in plan[key], key
        # A venv interpreter can legitimately link to the system Python binary.
        if key in {'harbor_python', 'engine_python'}:
            path.parent.resolve().relative_to(BOUND)
            assert path.is_file(), key
        else:
            path.resolve().relative_to(BOUND)
    assert plan['attempts_per_task'] == 1 and plan['automatic_scientific_retries'] == 0
    assert not plan['official_task_timeouts'] and plan['task_walltime_seconds'] is None
    assert len(plan['tasks']) == len(set(plan['tasks']))
    assert set(plan['tasks']) == {row['task'] for row in plan['resource_inventory']}
    assert max(row['memory_mb'] for row in plan['resource_inventory']) <= plan['host_memory_budget_mb']
    task_record = json.loads((Path(plan['task_root']).parent / 'task_copy_manifest.json').read_text())
    assert task_record['revision'] == plan['revision']
    expected = {row['path']: row for row in task_record['files']}
    checked_tasks = 0
    for name in plan['tasks']:
        path = Path(plan['task_root']) / name / 'task.toml'
        task = tomllib.loads(path.read_text())
        assert task.get('agent', {}).get('timeout_sec') is None, name
        for relative, row in expected.items():
            if relative.startswith(name + '/'):
                data = (Path(plan['task_root']) / relative).read_bytes()
                assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], relative
        assert name + '/task.toml' in expected
        checked_tasks += 1
    command = [plan['engine_python'], str(H / 'model_path_preflight.py')]
    model = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert model.returncode == 0, model.stderr
    checkpoint_result = None
    if checkpoint and plan['arm'] == 'encbank':
        script = """import json,torch
from pathlib import Path
from common import MODELS
from runtime_identity import resolve_configs
p=json.loads(Path('plan.json').read_text())
saved=torch.load(p['adapter_path'],map_location='cpu',weights_only=True)
identity,runtime=resolve_configs(p,MODELS[1],saved)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',training_model=identity,runtime_model=runtime,step=saved['step'],cuda_context=False)))
"""
        result = subprocess.run([plan['engine_python'], '-c', script], cwd=H, capture_output=True,
                                text=True, timeout=180, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert result.returncode == 0, result.stderr
        checkpoint_result = json.loads(result.stdout)
    harbor = subprocess.run([plan['harbor_python'], '-c',
        "import importlib.metadata;from harbor.agents.terminus_2.terminus_2 import Terminus2;import tb_agent_rpc;print(importlib.metadata.version('harbor'))"],
        cwd=H, env=dict(os.environ, PYTHONPATH=str(H), LITELLM_LOCAL_MODEL_COST_MAP='True'),
        capture_output=True, text=True, timeout=60)
    assert harbor.returncode == 0 and harbor.stdout.strip() == '0.23.0', harbor.stderr
    container_result = {'checked': False}
    if check_container:
        if plan['container_backend'] == 'docker':
            env = dict(os.environ, DOCKER_HOST=plan['docker_host'])
            version = docker_server_version(env)
            compose = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True, timeout=30, env=env)
            assert compose.returncode == 0, compose.stderr
            container_result = {'checked': True, 'backend': 'docker', 'version': version,
                                'compose': compose.stdout.strip()}
        else:
            if plan['container_backend'] == 'managed':
                assert os.uname().nodename in plan['container_qualified_hostnames'], 'This node has not been qualified for managed Apptainer'
                state = subprocess.run(['systemctl','--user','is-system-running'],capture_output=True,text=True,timeout=15)
                assert state.stdout.strip() in {'running','degraded'}, 'User service manager unavailable on this node'
                linger=subprocess.run(['loginctl','show-user','liuhanzuo','-p','Linger','--value'],capture_output=True,text=True,timeout=15)
                assert linger.returncode==0 and linger.stdout.strip()=='yes', 'User services must persist after client disconnect'
                from apptainer_environment import ManagedApptainerEnvironment
                assert ManagedApptainerEnvironment.resource_capabilities().memory_limit
            # A different backend requires an explicit compatibility record for these frozen tasks.
            qualification = H / 'container_qualification.json'
            assert qualification.exists(), 'Singularity needs task/environment compatibility qualification; a basic image probe is insufficient'
            record = json.loads(qualification.read_text())
            assert record['status'] == 'PASS' and record['plan_sha256'] == hashlib.sha256((H / 'plan.json').read_bytes()).hexdigest()
            if plan['container_backend'] == 'managed':
                assert record.get('scope') == 'all_plan_task_environments', 'Runtime smoke checks alone do not qualify every frozen task'
                assert set(record['tasks']) == set(plan['tasks'])
            container_result = record
    return {'status': 'PASS', 'epoch': time.time(), 'server_only': True, 'model_calls': 0,
            'source_manifest_sha256': source_sha, 'tasks_checked': checked_tasks,
            'model_path': json.loads(model.stdout), 'checkpoint_identity': checkpoint_result,
            'harbor_version': harbor.stdout.strip(), 'container': container_result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--without-container', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    try:
        report = check(not args.without_container, args.checkpoint)
    except Exception as exc:
        report = {'status': 'FAIL', 'epoch': time.time(), 'error': repr(exc), 'model_calls': 0}
    name = 'server_cpu_preflight.json' if args.without_container else 'server_preflight.json'
    (H / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['status'] == 'PASS' else 1)
