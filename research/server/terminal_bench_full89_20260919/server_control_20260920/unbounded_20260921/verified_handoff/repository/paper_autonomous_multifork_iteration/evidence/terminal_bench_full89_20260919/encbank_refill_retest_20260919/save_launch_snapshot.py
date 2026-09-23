"""Archive immutable current observations and verify saved heartbeat settings."""
from pathlib import Path
import datetime,hashlib,json,tomllib
R=Path('/srv/encbank/legacy_workspace');P=R/'paper_autonomous_multifork_iteration';H=Path(__file__).resolve().parent;D=H.parent/'dense_restore37_20260919'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def ref(p):return dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
payload=load(H/'automation_payload.json');actual=tomllib.loads(Path('/srv/encbank/client/.codex/automations/q-encbank/automation.toml').read_text(encoding='utf8'))
assert actual['prompt'].strip()==payload['prompt'].strip() and actual['rrule']==payload['rrule'] and actual['status']==payload['status']
save(H/'automation_verification.json',dict(status='PASS',automation_id='q-encbank',rrule=actual['rrule'],prompt_sha256=hashlib.sha256(actual['prompt'].encode()).hexdigest()))
record=load(H/'restore_registration.json');record['updated_at']=datetime.datetime.now().astimezone().isoformat()
for h in [H,D]:
    dst=h/'launch_snapshot.json';assert not dst.exists();dst.write_bytes((h/'latest_observation.json').read_bytes())
    obs=load(dst);remote=obs['remote'][h.name];local=obs['local'][h.name];e=record['new_jobs'][h.name]
    e.update(observation=ref(dst),accounting=remote['accounting'],owner_alive=local['current_controller_pid_running'],actual_owner_pid=local['current_controller_pid'],worker_ready='worker_ready.json' in remote)
    e['state']='GPU_WORKER_READY' if e['worker_ready'] else ('GPU_STARTUP' if '|RUNNING|' in e['accounting'] else 'WAITING_SLURM')
    e['responses']=remote['responses'];e['transport_errors']=len(local['transport_errors']);e['active_tasks']=list(local.get('execution/status.json',{}).get('active',{}))
    assert e['owner_alive'] and 'worker_failure.json' not in remote and 'process_receipt.json' not in remote
    reg=load(h/'registration.json');reg['latest_observation']=e['observation'];reg['state']=e['state'];save(h/'registration.json',reg)
save(H/'restore_registration.json',record)
for n in ['codex_experiment_status.json','paper_state.json']:
    path=P/'state'/n;obj=load(path);x=obj if n.startswith('codex') else obj['active_edge_continuation'];x['updated_at']=record['updated_at']
    x['current_experiment']['latest_restore']=ref(H/'restore_registration.json');x['current_experiment']['active_restored_campaigns']=record['new_jobs'];x['latest_restore_registration']=ref(H/'restore_registration.json');save(path,obj)
path=P/'evidence/experiment_registry.json';obj=load(path)
for i,x in enumerate(obj['planned_campaigns']):
    if x.get('evidence_id')==record['evidence_id']:obj['planned_campaigns'][i]=record
obj['current_original_encbank_terminal_bench_state']=record;save(path,obj)
print(json.dumps({n:dict(job=v['job_id'],state=v['state'],responses=v['responses'],active=v['active_tasks']) for n,v in record['new_jobs'].items()}))
