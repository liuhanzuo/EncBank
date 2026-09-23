"""Refresh source references after a held, never-started job was corrected."""
from pathlib import Path
import datetime,hashlib,json,tomllib
R=Path('/srv/encbank/legacy_workspace');P=R/'paper_autonomous_multifork_iteration';H=Path(__file__).resolve().parent;D=H.parent/'dense_restore37_20260919'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def ref(p):return dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert load(H/'padding_trim_release.json')['exit_code']==0
record=load(H/'restore_registration.json');record.update(updated_at=datetime.datetime.now().astimezone().isoformat(),padding_reclamation=ref(H/'padding_trim_registration.json'))
for h in [H,D]:
    e=record['new_jobs'][h.name];e['source_manifest']=ref(h/'remote_source_manifest.json');e['observation']=ref(h/'latest_observation.json')
    obs=load(h/'latest_observation.json');remote=obs['remote'][h.name];local=obs['local'][h.name]
    e['accounting']=remote['accounting'];e['owner_alive']=local['current_controller_pid_running'];e['actual_owner_pid']=local['current_controller_pid'];e['worker_ready']='worker_ready.json' in remote
    assert e['owner_alive'] and 'worker_failure.json' not in remote
    e['state']='WAITING_SLURM' if 'PENDING' in e['accounting'] else 'INSPECT_OBSERVATION_FOR_CURRENT_PHASE'
    reg=load(h/'registration.json');reg['source_manifest']=e['source_manifest'];reg['latest_observation']=e['observation'];save(h/'registration.json',reg)
save(H/'restore_registration.json',record)
note='Encbank108870在未启动时暂挂，回收所有活跃行共同屏蔽的旧KV填充列；CPU与独立串行前向8步一致至2.13e-7，缓存列31降18且H不变。新源码核远端SHA后已释放同一作业，无新GPU提交。证据padding_trim_registration.json与padding_trim_release.json。'
for n in ['codex_experiment_status.json','paper_state.json']:
    path=P/'state'/n;obj=load(path);x=obj if n.startswith('codex') else obj['active_edge_continuation'];x['updated_at']=record['updated_at'];x['next_priority']+=' '+note
    cur=x['current_experiment'];cur['latest_restore']=ref(H/'restore_registration.json');cur['active_restored_campaigns']=record['new_jobs'];x['latest_restore_registration']=ref(H/'restore_registration.json');save(path,obj)
path=P/'evidence/experiment_registry.json';obj=load(path)
for i,x in enumerate(obj['planned_campaigns']):
    if x.get('evidence_id')==record['evidence_id']:obj['planned_campaigns'][i]=record
obj['current_original_encbank_terminal_bench_state']=record;save(path,obj)
q=R/'EXPERIMENT_QUEUE.md';text=q.read_text(encoding='utf8');heading='## Terminal-Bench 动态补位重测与 Dense 续跑（当前）\n\n';text=text.replace(heading,heading+note+'\n\n',1);q.write_text(text,encoding='utf8',newline='\n')
with (P/'state/decision_log.md').open('a',encoding='utf8',newline='\n') as f:f.write('\n'+record['updated_at']+' '+note+'\n')
payload=load(H/'automation_payload.json');payload['prompt']=payload['prompt'].replace('\n\n共同约束：','\n\n'+note+'\n\n共同约束：',1);save(H/'automation_payload.json',payload)
save(H/'focused_verification.json',dict(status='PASS',at=record['updated_at'],source_hashes_match=all(hashlib.sha256((h/n).read_bytes()).hexdigest()==d for h in [H,D] for n,d in load(h/'remote_source_manifest.json').items()),new_gpu_jobs=2,no_existing_result_overwrite=True,actual_job_states={n:v['accounting'] for n,v in record['new_jobs'].items()},manuscript_changed=False))
print(json.dumps({n:dict(job=v['job_id'],state=v['state'],owner_alive=v['owner_alive']) for n,v in record['new_jobs'].items()}))
