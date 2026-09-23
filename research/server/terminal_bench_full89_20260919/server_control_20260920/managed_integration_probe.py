"""Real Docker-free integration checks; no model requests or benchmark scoring."""
import asyncio, json, os, time, tomllib
from pathlib import Path
from apptainer_environment import ManagedApptainerEnvironment, ROOT
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


async def main():
    job=os.environ['SLURM_JOB_ID']
    output=ROOT/('managed-integration-'+job)
    output.mkdir(mode=0o700)
    task=ROOT/'tasks/cancel-async-tasks'
    cfg=tomllib.loads((task/'task.toml').read_text())
    environments=[ManagedApptainerEnvironment(environment_dir=task/'environment',environment_name='container-check',
        session_id=job+'-'+str(i),trial_paths=TrialPaths(trial_dir=output/str(i)),
        task_env_config=EnvironmentConfig.model_validate(cfg['environment']),singularity_image_cache_dir=ROOT/'sif') for i in range(2)]
    report=dict(status='RUNNING',model_calls=0,benchmark_attempts=0,job_id=job,checks={})
    async def command(environment,name,text,seconds=30):
        result=await environment.exec(text,timeout_sec=seconds)
        report['checks'][name]=result.model_dump()
        assert result.return_code==0,(name,result)
        return result
    try:
        results=await asyncio.gather(*(e.start(False) for e in environments),return_exceptions=True)
        assert not any(isinstance(r,BaseException) for r in results),repr(results)
        states=[json.loads((e._root/'service.json').read_text()) for e in environments]
        assert states[0]['network']!=states[1]['network']
        report['independent_networks']=[s['network'] for s in states]
        for i,e in enumerate(environments):
            await command(e,'server_'+str(i),f"mkdir -p /app/probe-web; printf instance-{i} > /app/probe-web/index.html; cd /app/probe-web; nohup python3 -m http.server 8337 --bind 127.0.0.1 >/tmp/probe-web.log 2>&1 </dev/null &")
        await asyncio.sleep(1)
        for i,e in enumerate(environments):
            result=await command(e,'same_port_'+str(i),'python3 -c \'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8337",timeout=5).read().decode())\'')
            assert result.stdout.strip()==f'instance-{i}'
        await command(environments[0],'disk_over_64mb','dd if=/dev/zero of=/app/probe.disk bs=1M count=128 status=none; test "$(stat -c %s /app/probe.disk)" = 134217728; rm /app/probe.disk')
        group=Path('/sys/fs/cgroup')/states[0]['cgroup'].lstrip('/')
        def events():
            return dict(line.split() for line in (group/'memory.events').read_text().splitlines())
        before=events()
        result=await environments[0].exec('python3 -c "x=bytearray(2300*1024*1024); print(len(x))"',timeout_sec=30)
        after=events()
        report['oom_probe']=dict(result=result.model_dump(),before=before,after=after)
        assert result.return_code!=0 and int(after['oom_kill'])>int(before['oom_kill'])
        report['status']='PASS'
    except Exception as error:
        report.update(status='FAIL',error=repr(error))
    finally:
        report['cleanup']=[]
        for e in environments:
            try:
                await e.stop(True)
                report['cleanup'].append(json.loads((e._root/'closure.json').read_text()))
            except Exception as error:
                report['cleanup'].append({'error':repr(error)})
                report['status']='FAIL'
        report['epoch']=time.time()
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    assert report['status']=='PASS'


asyncio.run(main())
