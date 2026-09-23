"""Verify frozen inputs plus real Harbor Docker startup and Terminus setup; no model calls."""
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT=Path('/srv/encbank/client/comem_local_20260921')
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'


async def probe(task,run):
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.config import ResourceMode
    sys.path.insert(0,str(run))
    from tb_agent_rpc import ConcurrentDenseTerminus
    config=json.loads((run/'dense_harbor_template.json').read_text())
    directory=ROOT/'tasks'/task
    cfg=tomllib.loads((directory/'task.toml').read_text())
    output=ROOT/'qualification'/task
    output.mkdir(parents=True,exist_ok=False)
    env=DockerEnvironment(environment_dir=directory/'environment',environment_name=task,
        session_id='comem-qualification-'+task,trial_paths=TrialPaths(trial_dir=output),
        task_env_config=EnvironmentConfig.model_validate(cfg['environment']),
        persistent_env=config['environment']['env'],cpu_enforcement_policy=ResourceMode.LIMIT,
        memory_enforcement_policy=ResourceMode.LIMIT,logger=logging.getLogger(task))
    record=dict(task=task,status='RUNNING',model_calls=0)
    try:
        await asyncio.wait_for(env.start(force_build=False),timeout=600)
        agent=ConcurrentDenseTerminus(logs_dir=output/'agent',model_name=config['agents'][0]['model_name'],**config['agents'][0]['kwargs'])
        await asyncio.wait_for(agent.setup(env),timeout=360)
        result=await env.exec('command -v tmux; command -v asciinema; cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/cpu.max',timeout_sec=30)
        assert result.return_code==0,result.stderr
        record.update(status='PASS',environment_start=True,terminus_setup=True,resource_check=result.stdout)
        assert not list((ROOT/'rpc').rglob('*.request.json')),'Preflight must never call model'
    except Exception as error:record.update(status='FAIL',error=repr(error))
    finally:
        await env.stop(delete=True)
        record['environment_stopped']=True
        (output/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    report=dict(status='RUNNING',model_calls=0,epoch=time.time(),harbor=importlib.metadata.version('harbor'))
    assert report['harbor']=='0.23.0'
    frozen=json.loads((ROOT/'task_copy_manifest.json').read_text())
    checked=0
    for row in frozen['files']:
        if row['path'].split('/')[0] not in {'regex-chess','vulnerable-secret'}:continue
        p=ROOT/'tasks'/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        checked+=1
    report['task_files_hash_checked']=checked
    for family in ['dense','k12','k48']:
        r=ROOT/'runs'/family
        for name,digest in json.loads((r/'source_manifest.json').read_text()).items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
        code="import tb_agent_rpc, hybrid_owner; from harbor.models.job.config import JobConfig; import json; from pathlib import Path; p=json.loads(Path('plan.json').read_text()); JobConfig.model_validate_json(Path(p['arm']+'_harbor_template.json').read_text()); print('PASS')"
        result=subprocess.run([sys.executable,'-c',code],cwd=r,env=dict(os.environ,PYTHONPATH=str(r)),capture_output=True,text=True,timeout=90)
        assert result.returncode==0,(family,result.stderr)
    async def checks():return await asyncio.gather(*(probe(t,ROOT/'runs/dense') for t in ['regex-chess','vulnerable-secret']))
    report['environment_checks']=asyncio.run(checks())
    report['status']='PASS' if all(x['status']=='PASS' for x in report['environment_checks']) else 'FAIL'
    (ROOT/'local_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
