"""Check the allocated host with one real install-only environment, before model load."""
import asyncio
import json
import os
import socket
import time
import traceback
from pathlib import Path


def check_runtime_node(plan):
    import harbor_unbounded
    from harbor.trial.trial import Trial
    from harbor.models.trial.config import TrialConfig

    root = Path(__file__).resolve().parent
    out = root / 'node_runtime_qualification'
    out.mkdir(exist_ok=False)
    qualification = json.loads((root / 'container_qualification.json').read_text())
    candidates = [r for r in qualification['checks'] if r['task'] in plan['tasks']
                  and r['status'] == 'PASS' and r['network_policy']['network_mode'] == 'public']
    task = min(candidates, key=lambda r: r['elapsed_seconds'])['task']
    template = json.loads((root / 'comem_harbor_template.json').read_text())
    started = time.time()
    row = dict(status='FAIL', hostname=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'],
               task=task, scope='allocated_host_runtime_smoke_not_all_task_qualification',
               model_calls=0, benchmark_attempts=0, started_epoch=started)

    async def probe():
        cfg = TrialConfig.model_validate(dict(task={'path': str(Path(plan['task_root']) / task)},
            trial_name='node-runtime-' + task, trials_dir=str(out), install_only=True,
            agent=template['agents'][0], environment=template['environment'], verifier={}))
        trial = await Trial.create(cfg)
        limits = {n: getattr(trial, n) for n in ['_agent_timeout_sec', '_verifier_timeout_sec',
                    '_agent_setup_timeout_sec', '_environment_build_timeout_sec']}
        assert all(v is None for v in limits.values()), limits
        result = await trial.run()
        assert result.exception_info is None, result.exception_info
        assert result.agent_execution is None and result.verifier_result is None
        env = trial.agent_environment
        spec = json.loads((env._root / 'spec.json').read_text())
        service = json.loads((env._root / 'service.json').read_text())
        closure = json.loads((env._root / 'closure.json').read_text())
        network = json.loads((env._root / 'network_diagnostic.json').read_text())
        expected = next(r for r in plan['resource_inventory'] if r['task'] == task)
        assert int(service['memory_max']) == expected['memory_mb'] * 2**20
        assert closure['cgroup_empty'] and len(spec['affinity']) == expected['cpus']
        assert spec['internet'] and 'tap0' in network['stdout']
        row.update(status='PASS', service_root=str(env._root), closure=closure,
                   memory_max=service['memory_max'], affinity=spec['affinity'],
                   network_policy=spec['network_policy'], resolved_time_limits=limits)

    try:
        asyncio.run(probe())
    except Exception:
        row['error'] = traceback.format_exc()
    row.update(ended_epoch=time.time(), elapsed_seconds=time.time() - started)
    (root / 'node_runtime_preflight.json').write_text(json.dumps(row, indent=2) + '\n')
    assert row['status'] == 'PASS', row
    return row
