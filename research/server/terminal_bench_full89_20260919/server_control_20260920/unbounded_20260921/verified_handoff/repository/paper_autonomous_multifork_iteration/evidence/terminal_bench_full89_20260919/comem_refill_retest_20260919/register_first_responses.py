"""Register actual replies/refill plus separate two pre-agent infrastructure failures."""
from pathlib import Path
import datetime,hashlib,json,statistics
R=Path('/srv/encbank/legacy_workspace');P=R/'paper_autonomous_multifork_iteration';H=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def ref(p):return dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
obs=load(H/'launch_snapshot.json');r=obs['remote'][H.name];l=obs['local'][H.name]
events=[json.loads(s) for s in r['events.jsonl']];assert any(x['event']=='batch_refill' for x in events)
box=R/'.runtime/terminal_bench_full89_20260919'/('local_rpc_'+H.name)/'comem';rows=[]
for p in sorted(box.glob('*.response.json')):
    q=load(p);b=load(p.with_name(p.name.replace('.response.','.broker.')))
    assert b['status']=='delivered' and hashlib.sha256(p.read_bytes()).hexdigest()==b['proof']['sha256']
    rows.append(dict(path=str(p),sha256=b['proof']['sha256'],task=q['task'],step=q['step'],status=q['status'],generated_tokens=q['generated_tokens'],queue_seconds=q['queue_seconds'],batch_number=q['batch_number'],admission_rows=q['batch_initial_size'],H_immutable=q['state_hashes_unchanged']))
assert rows and all(x['status']=='ok' and x['H_immutable'] for x in rows)
infra=load(H/'compose_cwd_guard_registration.json');now=datetime.datetime.now().astimezone().isoformat()
record=dict(at=now,status='RUNNING_VERIFIED_REPLIES_AND_REFILL',job_id='108870',snapshot=ref(H/'launch_snapshot.json'),hardware=r['worker_ready.json'],model_replies=len(rows),reply_rows=rows,
    queue_seconds_median=statistics.median(x['queue_seconds'] for x in rows),queue_seconds_max=max(x['queue_seconds'] for x in rows),
    actual_refill_event=[x for x in events if x['event']=='batch_refill'],pre_agent_failures=infra['failures'],pre_agent_recovery=ref(H/'compose_cwd_guard_registration.json'),
    benchmark_complete=False,task_scores_reported=False,old_static_results_preserved=True)
save(H/'first_response_progress.json',record)
note=f'截至{now}，CoMem108870已在lj-gpu8/实际hostname gpu-host、L20D/CC10.3完成载模；8个任务环境推进，已核{len(rows)}条真实ok回复与H不变，GPU事件实际batch_refill出现。Dense108879仍PENDING，owner正常。两条adaptive-rejection-sampler/build-cython-ext在Docker compose启动报getwd且model_requests=0、verifier=null，属于基础设施失败不能计零；已为两新campaign后续Harbor进程安装本地稳定cwd shim，保持全部Compose参数和显式project-directory，6次只读CPU CLI验证PASS，非底层WSL故障已修复声明。当前有效8任务不中断；首批结束后仅在新目录补这两条pre-agent失败，不重跑已有效评分题。证据first_response_progress.json、compose_cwd_guard_registration.json。'
for n in ['codex_experiment_status.json','paper_state.json']:
    path=P/'state'/n;obj=load(path);x=obj if n.startswith('codex') else obj['active_edge_continuation'];x['updated_at']=now;x['next_priority']+=' '+note
    x['current_experiment']['first_refill_progress']=ref(H/'first_response_progress.json');x['current_experiment']['pre_agent_infrastructure_pending']=[z['task'] for z in infra['failures']];save(path,obj)
path=P/'evidence/experiment_registry.json';obj=load(path);obj['completed_analyses'].append(dict(evidence_id='E-TB21-REFILL-FIRST-RESPONSES-20260919',record=ref(H/'first_response_progress.json'),status=record['status']));obj['current_original_comem_terminal_bench_state']['first_progress']=ref(H/'first_response_progress.json');save(path,obj)
q=R/'EXPERIMENT_QUEUE.md';text=q.read_text(encoding='utf8');heading='## Terminal-Bench 动态补位重测与 Dense 续跑（当前）\n\n';text=text.replace(heading,heading+note+'\n\n',1);q.write_text(text,encoding='utf8',newline='\n')
with (P/'state/decision_log.md').open('a',encoding='utf8',newline='\n') as f:f.write('\n'+note+'\n')
payload=load(H/'automation_payload.json');payload['prompt']=payload['prompt'].replace('\n\n共同约束：','\n\n'+note+'\n\n共同约束：',1);save(H/'automation_payload.json',payload)
with (H/'STATUS.md').open('a',encoding='utf8',newline='\n') as f:f.write('\n'+note+'\n')
print(json.dumps(dict(status=record['status'],model_replies=len(rows),queue_median_seconds=record['queue_seconds_median'],queue_max_seconds=record['queue_seconds_max'],pre_agent_missing=len(infra['failures']))))
