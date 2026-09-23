"""Full Harbor environment and agent setup, four concurrent worlds, zero inference."""
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import tomllib
import traceback

H=Path(__file__).resolve().parent
P=json.loads((H/'plan.json').read_text())
OUT=H/'qualification'


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


async def check(task, semaphore):
    from apptainer_environment import ManagedApptainerEnvironment
    from tb_agent_rpc import ConcurrentDenseTerminus
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    async with semaphore:
        out=OUT/task;out.mkdir(parents=True,exist_ok=False)
        cfg=tomllib.loads((Path(P['task_root'])/task/'task.toml').read_text())
        template=json.loads((H/'dense_harbor_template.json').read_text())
        env=ManagedApptainerEnvironment(environment_dir=Path(P['task_root'])/task/'environment',
             environment_name=task,session_id='p4qualification-'+task,
             trial_paths=TrialPaths(trial_dir=out),task_env_config=EnvironmentConfig.model_validate(cfg['environment']),
             singularity_image_cache_dir=Path(P['sif_cache']),singularity_no_mount='home,tmp,bind-paths',logger=logging.getLogger(task))
        row=dict(task=task,status='RUNNING',started_epoch=time.time(),model_calls=0)
        save(out/'report.json',row)
        try:
            await asyncio.wait_for(env.start(force_build=False),timeout=600)
            agent=ConcurrentDenseTerminus(logs_dir=out/'agent',model_name=template['agents'][0]['model_name'],**template['agents'][0]['kwargs'])
            await asyncio.wait_for(agent.setup(env),timeout=600)
            result=await env.exec('command -v tmux; command -v asciinema; cat /proc/self/cgroup; test ! -e /srv/encbank/qencbank_runtime_20260911/models',timeout_sec=30)
            assert result.return_code==0,result.stderr
            record=json.loads((env._root/'service.json').read_text())
            assert int(record['memory_max'])==cfg['environment']['memory_mb']*2**20
            assert record['cgroup'] in result.stdout
            row.update(status='PASS',environment_started=True,terminus_setup=True,memory_max=record['memory_max'],service_root=str(env._root))
        except Exception:
            row.update(status='FAIL',error=traceback.format_exc())
        finally:
            try:
                await env.stop(delete=True)
                row['closure']=json.loads((env._root/'closure.json').read_text())
                assert row['closure']['cgroup_empty']
            except Exception:
                row.update(status='FAIL',cleanup_error=traceback.format_exc())
            row['ended_epoch']=time.time();row['elapsed_seconds']=row['ended_epoch']-row['started_epoch']
            save(out/'report.json',row)
        assert not list(Path(P['rpc_root']).glob('**/*.request.json')),'Qualification must not invoke model'
        print(json.dumps(row),flush=True)
        return row


def main():
    OUT.mkdir(exist_ok=False)
    from server_preflight import check as preflight
    cpu=preflight(check_container=False)
    save(H/'qualification_cpu.json',cpu)
    began=time.time()
    imports=subprocess.run([P['engine_python'],'-c','import torch,vllm; assert not torch.cuda.is_initialized(); print(torch.__version__,vllm.__version__)'],
            capture_output=True,text=True,timeout=240,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    save(H/'engine_import_qualification.json',dict(exit_code=imports.returncode,stdout=imports.stdout,stderr=imports.stderr[-3000:],elapsed_seconds=time.time()-began,cuda_visible_devices='',model_calls=0))
    assert imports.returncode==0,imports.stderr
    async def all_checks():
        semaphore=asyncio.Semaphore(4)
        return await asyncio.gather(*(check(t,semaphore) for t in P['tasks']))
    rows=asyncio.run(all_checks())
    report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
         scope='all_plan_task_environments',plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
         tasks=P['tasks'],checks=rows,task_concurrency=4,model_calls=0,benchmark_attempts=0,
         qualification='Actual Harbor0.23 environment start and Terminus setup; original RAM; no model/answer/verifier calls',
         limitation='Managed Apptainer uses CPU affinity, recorded separately from legacy Docker')
    save(H/'container_qualification.json',report)
    print(json.dumps({'status':report['status'],'tasks':len(rows),'model_calls':0}),flush=True)
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
