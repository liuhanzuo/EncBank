"""Register actual new submissions; supersede old no-retest instructions explicitly."""
from pathlib import Path
import datetime,hashlib,json,shutil
R=Path('/srv/encbank/legacy_workspace');P=R/'paper_autonomous_multifork_iteration';H=Path(__file__).resolve().parent;D=H.parent/'dense_restore37_20260919'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def ref(p):return dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(D/'observe.py').write_text((H/'observe.py').read_text().replace("(H.name,'comem',","(H.name,'dense',"),encoding='utf8',newline='\n')
(D/'.gitignore').write_text('__pycache__/\nexecution/\nowner*.log\n',encoding='utf8',newline='\n')
now=datetime.datetime.now().astimezone().isoformat();entries={}
for h in [H,D]:
    p=load(h/'plan.json');s=load(h/'submission.json');assert s['status']=='submitted'
    entries[h.name]=dict(job_id=s['job_id'],plan=ref(h/'plan.json'),source_manifest=ref(h/'remote_source_manifest.json'),submission=ref(h/'submission.json'),native_launch=ref(h/'native_launch.json'),remote_root=p['remote_root'],task_count=len(p['tasks']),state='SUBMITTED_WAITING_ACTUAL_WORKER',checks={n:load(h/n)['status'] for n in ['cpu_preflight_local.json','cpu_preflight_remote.json']})
    reg=load(h/'registration.json');reg.update(status='SUBMITTED',submission=ref(h/'submission.json'),updated_at=now);save(h/'registration.json',reg)
record=dict(at=now,evidence_id='E-TB21-RESTORE-REFILL8-DENSE37-20260919',state='REFILL_RETEST_AND_DENSE_CONTINUATION_SUBMITTED',new_jobs=entries,
    current_gpu_requests=2,max_gpu_requests=4,old_comem_deployment=dict(accepted=15,passed=1,failed=14,preserved=True,pooled_with_retest=False),dense_retained=dict(accepted=52,passed=32,failed=20),
    full_benchmark_complete=False,service_cpu_check=ref(H/'service_cpu_check.json'),service_cpu_actual_wait=ref(H/'service_cpu_execution.json'),
    authorization='User explicitly requested restore and retest after static-cohort queue starvation. Full89 new CoMem scheduler run, not score-selected retries. Dense37 missing-only continuation.',
    storage_boundary='Current preparation writes passed; allocated-node formal gate remains. No proof shared BeeGFS underlying fault repaired. Prior errno121 failures retained; no response-generation retry.')
save(H/'restore_registration.json',record)
priority=('2026-09-19用户明确要求立即恢复并重新测试，取代旧的CoMem15条不得重测限制，但旧15条(1通过/14未通过)仍完整保留为旧静态调度结果，不与新结果混合或择优。'
    '已提交CoMem动态补位全89题新任务108870，根comem_refill_retest_20260919；Dense仅接续剩37题108879，根dense_restore37_20260919，已验52题(32通过/20未通过)禁止重跑。'
    '新任务均单GPU、cap240/free250GiB稳定45秒；原始官方任务超时、模型、LoRA、采样和k12保持。最多8路CoMem decode，运行中空槽补位，独立Write/prefill仍串行。'
    '微型CPU缓存数值检查PASS，实际service循环CPU测试23.02秒真实wait0：长请求未结束即补位，取消/到期及活跃H释放保护PASS；不冒称27B生成完全等价或实际吞吐已验证。'
    '优先运行两个新根的observe.py核squeue/sacct、真实owner/worker、事件、响应和每题退出；owner活跃禁止旁路重复提交。GPU排队/载模不算失败。'
    '旧107787/107885真实wait1关闭且不重启。当前共享盘准备写入通过，allocated-node正式storage_preflight仍必需；这不证明BeeGFS底层修复。若再次errno121，保存原输出/真实退出，不盲重试、不把受损题记零分；不得读权限不足的认证文件或他人凭据。'
    '当前两新请求占本线程2/4额度，独立于qcm-*他项；实际GPU型号以worker记录为准。完整89题前不更新论文总分。')
heading='## Terminal-Bench 动态补位重测与 Dense 续跑（当前）\n\n'+now+'。'+priority+'\n\n'
q=R/'EXPERIMENT_QUEUE.md';text=q.read_text(encoding='utf8');assert heading.split('\n')[0] not in text
text=text.replace('# Q-CoMem 当前实验队列\n\n','# Q-CoMem 当前实验队列\n\n'+heading,1).replace('## Terminal-Bench 超时裁定与动态补位准备（当前）','## Terminal-Bench 超时裁定与动态补位准备（历史；重测授权已更新）',1)
q.write_text(text,encoding='utf8',newline='\n')
with (P/'state/decision_log.md').open('a',encoding='utf8',newline='\n') as f:f.write('\n\n'+heading)
for n in ['codex_experiment_status.json','paper_state.json']:
    path=P/'state'/n;obj=load(path);x=obj if n.startswith('codex') else obj['active_edge_continuation']
    x.update(status=record['state'],updated_at=now,next_priority=priority)
    cur=x['current_experiment'];cur.update(state=record['state'],latest_restore=ref(H/'restore_registration.json'),active_restored_campaigns=entries,current_gpu_jobs=2,comem_current_retest_total=89,comem_current_retest_completed=0,comem_old_deployment_retained=15,dense_frozen_tasks=52,dense_remaining_tasks=37)
    for name in ['dense_capacity_resume','comem_batch8_service']:
        if name in cur.get('current_plans',{}):cur['current_plans'][name]['state']='CLOSED_FAILED_STORAGE_HISTORICAL'
    x['latest_restore_registration']=ref(H/'restore_registration.json');save(path,obj)
path=P/'evidence/experiment_registry.json';obj=load(path);obj['planned_campaigns'].append(record);obj['current_original_comem_terminal_bench_state']=record;save(path,obj)
payload=load(H.parent/'retrieval_topk_followup/automation_payload.json');common=payload['prompt'].split('共同约束：',1)[1]
payload['prompt']='每3小时巡检/srv/encbank/legacy_workspace，保留每3小时23分quiet策略。先读AGENTS.md、最新队列/decision_log/codex_experiment_status/experiment_registry及paper_state.active_edge_continuation，再查真实进程输出。仅实质进展、完成、失败或必要用户行动通知。已有自主授权，不重复询问许可。\n\n当前：'+priority+'\n\n共同约束：'+common
save(H/'automation_payload.json',payload)
summary='# Terminal-Bench 恢复登记\n\n'+priority+'\n\n作业108870：CoMem完整89题重测；作业108879：Dense剩余37题。提交并不代表模型已载入。\n\n检查证据：service_cpu_check.json、service_cpu_execution.json、两根cpu_preflight_*.json与remote_source_manifest.json。旧负面结果保留在storage_incident_20260919_1709。\n'
(H/'STATUS.md').write_text(summary,encoding='utf8',newline='\n')
print(json.dumps(dict(state=record['state'],jobs={k:v['job_id'] for k,v in entries.items()})))
