"""Actual Harbor install-only paths for all official environments, zero model calls."""
import asyncio,hashlib,json,time,traceback
from pathlib import Path
import harbor_unbounded
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

H=Path(__file__).resolve().parent;P=json.loads((H/'plan.json').read_text())
T=json.loads((H/'dense_harbor_template.json').read_text());OUT=H/'qualification'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

async def check(task,sem):
    async with sem:
        started=time.time();row=dict(task=task,status='FAIL',started_epoch=started,model_calls=0)
        try:
            cfg=TrialConfig.model_validate(dict(task={'path':str(Path(P['task_root'])/task)},
                trial_name='qualification-'+task,trials_dir=str(OUT),install_only=True,
                agent=T['agents'][0],environment=T['environment'],verifier={}))
            trial=await Trial.create(cfg)
            limits={n:getattr(trial,n) for n in ['_agent_timeout_sec','_verifier_timeout_sec',
                '_agent_setup_timeout_sec','_environment_build_timeout_sec']}
            assert all(v is None for v in limits.values()),limits
            result=await trial.run();env=trial.agent_environment
            row.update(exception_type=result.exception_info.exception_type if result.exception_info else None,
                exception_message=result.exception_info.exception_message if result.exception_info else None,
                resolved_time_limits=limits)
            assert result.exception_info is None,row['exception_message']
            assert result.agent_execution is None and result.verifier_result is None
            spec=json.loads((env._root/'spec.json').read_text());service=json.loads((env._root/'service.json').read_text())
            closure=json.loads((env._root/'closure.json').read_text());network=json.loads((env._root/'network_diagnostic.json').read_text())
            expected=next(r for r in P['resource_inventory'] if r['task']==task)
            assert int(service['memory_max'])==expected['memory_mb']*2**20 and closure['cgroup_empty']
            assert len(spec['affinity'])==expected['cpus']
            if spec['internet']:assert 'tap0' in network['stdout']
            row.update(status='PASS',service_root=str(env._root),closure=closure,
                network_policy=spec['network_policy'],memory_max=service['memory_max'],affinity=spec['affinity'])
        except Exception:row['audit_error']=traceback.format_exc()
        row.update(ended_epoch=time.time(),elapsed_seconds=time.time()-started)
        save(OUT/(task+'.json'),row)
        print(json.dumps({k:v for k,v in row.items() if k in ['task','status','elapsed_seconds','exception_type','exception_message','audit_error']}),flush=True)
        return row

async def main():
    OUT.mkdir(exist_ok=False)
    from server_preflight import check as preflight
    save(H/'qualification_cpu.json',preflight(check_container=False))
    sem=asyncio.Semaphore(8)
    rows=await asyncio.gather(*(check(t,sem) for t in P['tasks']))
    assert not list(Path(P['rpc_root']).glob('**/*.request.json'))
    report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
        scope='all_plan_task_environments',plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
        backend_hashes={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['apptainer_environment.py','apptainer_service.py','apptainer_executor.py','cpu_slots.py','harbor_unbounded.py']},
        tasks=P['tasks'],checks=rows,task_concurrency=8,model_calls=0,benchmark_attempts=0)
    save(H/'container_qualification.json',report)
    print(json.dumps(dict(status=report['status'],passed=sum(r['status']=='PASS' for r in rows),total=len(rows))),flush=True)

if __name__=='__main__':asyncio.run(main())
