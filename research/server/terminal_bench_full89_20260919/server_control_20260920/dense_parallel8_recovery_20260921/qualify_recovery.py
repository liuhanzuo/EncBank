import asyncio,hashlib,json,os,time,traceback
from pathlib import Path
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig
H=Path(__file__).resolve().parent
P=json.loads((H/'plan.json').read_text())
T=json.loads((H/'dense_harbor_template.json').read_text())
OUT=H/'qualification'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
async def check(task,sem):
 async with sem:
  started=time.time()
  cfg=TrialConfig.model_validate(dict(task={'path':str(Path(P['task_root'])/task)},trial_name='qualification-'+task,
    trials_dir=str(OUT),install_only=True,agent=T['agents'][0],environment=T['environment'],
    verifier={},artifacts=[],extra_instruction_paths=[],extra_instructions=[]))
  trial=await Trial.create(cfg)
  policy=trial.agent_environment._network_policy.model_dump(mode='json')
  result=await trial.run()
  env=trial.agent_environment
  row=dict(task=task,status='FAIL',started_epoch=started,model_calls=0,
    network_policy=policy,legacy_allow_internet=env.task_env_config.allow_internet,
    exception_type=result.exception_info.exception_type if result.exception_info else None,
    exception_message=result.exception_info.exception_message if result.exception_info else None)
  try:
   assert result.exception_info is None,row['exception_message']
   assert result.agent_execution is None and result.verifier_result is None
   spec=json.loads((env._root/'spec.json').read_text())
   service=json.loads((env._root/'service.json').read_text())
   closure=json.loads((env._root/'closure.json').read_text())
   network=json.loads((env._root/'network_diagnostic.json').read_text())
   assert spec['internet'] is True and policy['network_mode']=='public'
   assert 'tap0' in network['stdout']
   assert int(service['memory_max'])==8192*2**20 and closure['cgroup_empty']
   row.update(status='PASS',service_root=str(env._root),closure=closure,
     resolved_internet=spec['internet'],memory_max=service['memory_max'],affinity=service['affinity'])
  except Exception:
   row['audit_error']=traceback.format_exc()
  row.update(ended_epoch=time.time(),elapsed_seconds=time.time()-started)
  save(OUT/(task+'.json'),row)
  print(json.dumps(row),flush=True)
  return row
async def main():
 OUT.mkdir(exist_ok=False)
 from server_preflight import check as preflight
 save(H/'qualification_cpu.json',preflight(check_container=False))
 sem=asyncio.Semaphore(P['concurrent_tasks'])
 rows=await asyncio.gather(*(check(t,sem) for t in P['tasks']))
 assert not list(Path(P['rpc_root']).glob('**/*.request.json'))
 report=dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
  scope='all_plan_task_environments',plan_sha256=hashlib.sha256((H/'plan.json').read_bytes()).hexdigest(),
  tasks=P['tasks'],checks=rows,task_concurrency=P['concurrent_tasks'],qualified_simultaneous_tasks=len(rows),
  model_calls=0,benchmark_attempts=0,qualification='Actual Harbor Trial.create and install_only Trial.run path; no model or verifier calls')
 save(H/'container_qualification.json',report)
 print(json.dumps({'status':report['status'],'configured_ceiling':P['concurrent_tasks'],'qualified_tasks':len(rows)}),flush=True)
 if report['status']!='PASS':raise SystemExit(1)
asyncio.run(main())
