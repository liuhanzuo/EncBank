"""Combine unique valid pairs, preserving infrastructure attempts separately."""
import datetime,hashlib,json,statistics,subprocess
from pathlib import Path
B=Path('/srv/encbank/qcomem_align_codex_20260911'); OUT=Path(__file__).resolve().parent
R2=B/'hidden_reader_terminal_r2_20260921'; R4=B/'hidden_reader_terminal_r4_20260921'; R5=B/'hidden_reader_terminal_r5_20260921'
def load(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def med(xs):return statistics.median(xs) if xs else None
tasks=load(R2/'plan.json')['tasks'];rows=[];paired=[];replays=[];audit=[]
roots=load(R5/'roots.json') or {}
for path,expected in load(R2/'task_manifest.json').items():assert sha(Path(path))==expected,path
for task in tasks:
    root=R2 if task=='cancel-async-tasks' else Path(roots[task])
    environment=load(root/'pairs'/task/'environment.json')
    parent=load(root/'pairs'/task/'parent_exit.json');taskrows={}
    for arm in ['native','n24']:
        directory=root/'results'/(task+'--'+arm);result=load(directory/'actual_result.json')
        receipt=load(directory/'execution_receipt.json');wait=load(root/'jobs'/('wait-'+task+'--'+arm+'.json'))
        request_paths=sorted((root/'mailbox'/task).glob(arm+'_*.request.json'))
        responses=[]
        for path in request_paths:
            response=load(path.with_name(path.name.replace('.request.','.response.')))
            if response:
                assert response['request_sha256']==sha(path),(task,arm,path)
                responses.append(response)
        valid=bool(result and receipt and wait and wait['exit_code']==0 and wait['actual_parent_wait']
            and receipt['closure'] and receipt['closure']['cgroup_empty'] and not result['exception_info']
            and result['verifier_result'] and responses and len(responses)==len(request_paths)
            and all(r['status']=='ok' and r['H12_versions_unchanged'] for r in responses))
        rewards=result['verifier_result']['rewards'] if valid else None
        state='有效完成' if valid else ('未完成/异常' if receipt or load(root/'pairs'/task/'integrated_failure.json') or load(root/'pairs'/task/'worker_failure.json') else ('运行中' if request_paths else '待运行'))
        metrics={k:sum(r.get(k,0) or 0 for r in responses) for k in ['generated_tokens','request_seconds','prefill_seconds','write_seconds','selection_seconds','decode_seconds']}
        row=dict(task=task,arm=arm,root=str(root),valid=valid,state=state,rewards=rewards,metrics=metrics,
            exception=(result.get('exception_info') or {}).get('exception_type') if result else None,
            wall_seconds=receipt['elapsed_seconds'] if receipt else None,calls=len(responses),
            pending_calls=len(request_paths)-len(responses),selected_chunk_counts=[r.get('selected_chunks',0) for r in responses],
            archived_chunk_counts=[r.get('archived_chunks',0) for r in responses],
            environment=environment,worker_parent=parent,
            history_calls=sum(r.get('selected_chunks',0)>0 for r in responses),
            full12_history_calls=sum(r.get('selected_chunks',0)==12 for r in responses))
        rows.append(row);taskrows[arm]=row
    if all(r['valid'] for r in taskrows.values()):
        a,b=taskrows['native'],taskrows['n24']
        paired.append(dict(task=task,wall_ratio=b['wall_seconds']/a['wall_seconds'],
            model_time_ratio=b['metrics']['request_seconds']/a['metrics']['request_seconds'],
            native_rewards=a['rewards'],n24_rewards=b['rewards'],
            native_tokens=a['metrics']['generated_tokens'],n24_tokens=b['metrics']['generated_tokens']))
    data=load(root/'pairs'/task/'replay.json') or []
    for rid in sorted({r['source_request'] for r in data}):
        group={arm:[r for r in data if r['source_request']==rid and r['arm']==arm] for arm in ['native','n36','n24']}
        if not all(len(v)==2 for v in group.values()):continue
        assert all(r['H12_versions_unchanged'] and r['status'] in ['ok','replay_ok'] for v in group.values() for r in v)
        assert len({(r['logical_prompt_sha256'],r['forced_tokens'],r['generated_tokens'],tuple(r['selected_indices'])) for r in data if r['source_request']==rid})==1
        times={arm:{key:med([r[key] for r in group[arm] if r.get(key) is not None]) for key in ['prefill_seconds','memory_build_seconds','decode_seconds','request_seconds']} for arm in group}
        replays.append(dict(task=task,request=rid,selected_chunks=group['native'][0]['selected_chunks'],query_tokens=group['native'][0]['query_tokens'],times=times,
            n24_to_native_prefill=times['n24']['prefill_seconds']/times['native']['prefill_seconds'],
            n24_to_n36_build=times['n24']['memory_build_seconds']/times['n36']['memory_build_seconds'],
            n24_to_native_request=times['n24']['request_seconds']/times['native']['request_seconds']))
for root in [B/'hidden_reader_terminal_20260921',R2,B/'hidden_reader_terminal_r3_20260921',R4]+[Path(r) for r in roots.values()]:
    source_manifest=load(root/'source_manifest.json') or {}
    changed=[name for name,h in source_manifest.items() if not (root/name).exists() or sha(root/name)!=h]
    assert not changed,(root,changed)
    jobs=([r for r in (load(R5/'submissions.json') or []) if r['root']==str(root)] if root.parent==R5 else load(root/'jobs/submissions_run.json')) or []
    accounting=subprocess.run(['sacct','-j',','.join(j['job'] for j in jobs),'-X','-n','-P','-o','JobID,JobName%55,State,ExitCode,NodeList,Start,End'],capture_output=True,text=True).stdout
    trials=[]
    for p in sorted((root/'results').glob('*/execution_receipt.json')):
        rec=load(p);valid_selected=any(r['valid'] and r['root']==str(root) and p.parent.name==r['task']+'--'+r['arm'] for r in rows)
        trials.append(dict(name=p.parent.name,selected_valid=valid_selected,model_calls=rec['model_calls'],
            exception=(rec['exception'] or {}).get('exception_type'),verifier=rec['verifier'],closure=rec['closure']))
    audit.append(dict(root=str(root),frozen_sources=len(source_manifest),changed=changed,accounting=accounting,trials=trials,
        controller_parent=load(root/'jobs/controller_parent_exit.json')))
summary=dict(at=datetime.datetime.now().astimezone().isoformat(),expected_trials=12,valid_trials=sum(r['valid'] for r in rows),
    rows=rows,paired=paired,replays=replays,audit=audit,
    workers_closed=all(r['worker_parent'] and r['worker_parent']['actual_parent_wait'] and r['worker_parent']['exit_code']==0 for r in rows))
save(OUT/'summary.json',summary)
lines=['# COMem n24：真实 Terminal-Bench 探索实验','',f"更新时间：{summary['at']}；有效完成 {summary['valid_trials']}/12 次任务运行。",'',
    '6 个预先选定的 Terminal-Bench 2.1 任务，每题 native/n24 各一个独立新环境。模型为 Qwen3-8B + 原 COMem LoRA，非 thinking、贪心生成。此结果独立于另一个 27B 全量 benchmark，不是完整 89 题成绩。','',
    'native 是原 COMem 上层联合 prefill；n24 对历史第 13～24 层做联合因果计算，25～36 层限制在 chunk 内。当前 query/生成 token 仍执行全部层，上层读取全部选中历史 KV。H12 独立写入，chunk=512、top12；每次调用检索一次，生成中不重检索。未加入 hot KV buffer 或投影头。','',
    '| 任务 | 配置 | 状态 | reward | 总耗时 s | 模型 s | prefill s | decode s | 调用 | 输出 token | 含历史/12块调用 |',
    '|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    m=r['metrics']; wall=f"{r['wall_seconds']:.2f}" if r['wall_seconds'] is not None else '—'
    lines.append(f"| {r['task']} | {r['arm']} | {r['state']} | {r['rewards']} | {wall} | {m['request_seconds']:.2f} | {m['prefill_seconds']:.3f} | {m['decode_seconds']:.2f} | {r['calls']} | {m['generated_tokens']} | {r['history_calls']}/{r['full12_history_calls']} |")
lines+=['','任务总耗时包含环境准备、模型交互、终端命令与原判分器，不包含模型加载、排队或回放。不同生成轨迹的总耗时比不能直接当作推理引擎加速倍数。错误或中断不会折算为正常 reward=0。','',
    '## 固定会话、固定输出回放','',
    '每题按预设规则选至多 3 条 native 中归档 chunk 最多的请求，使用同一输入和原输出前至多 128 个 token。native/n36/n24 各重复两次、顺序反转，下表为中位数。n36 分拆 memory/query，便于单独比较重建；native 联合 memory/query，更贴近现有实际路径。','',
    '| 任务/请求 | chunk | query token | native prefill ms | n24 prefill ms | n24/native | n36 重建 ms | n24 重建 ms | n24/n36 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in replays:
    t=r['times'];lines.append(f"| {r['task']}/{r['request']} | {r['selected_chunks']} | {r['query_tokens']} | {1000*t['native']['prefill_seconds']:.2f} | {1000*t['n24']['prefill_seconds']:.2f} | {r['n24_to_native_prefill']:.3f} | {1000*t['n36']['memory_build_seconds']:.2f} | {1000*t['n24']['memory_build_seconds']:.2f} | {r['n24_to_n36_build']:.3f} |")
lines+=['','## 当前可解释的结果','']
for r in rows:
    if r['calls'] and (r['valid'] or r['full12_history_calls']):
        m=r['metrics'];share=100*m['prefill_seconds']/m['request_seconds']
        lines.append(f"- {r['task']}/{r['arm']}：{r['calls']} 次已返回调用中，prefill 占模型时间 {share:.2f}%，decode 占 {100*m['decode_seconds']/m['request_seconds']:.2f}%。状态：{r['state']}。")
lines+=['','重建只占 prefill 的一部分。例如 prefill 占总模型时间 3% 时，即使把全部 prefill 缩短 17%，模型总时间理论上也仅减少约 0.51%；还没有计入终端命令和环境时间。此式用于解释耗时构成，不是已测得的总加速率。','',
    '完成但 reward=0 仍是有效失败样本；两组都失败不能证明质量保持。未结束任务没有最终得分。当前样本少，且带历史的已完成任务可能只有少量 chunk，必须结合上表真实规模解读。']
lines+=['','## 有效性与保留记录','',
    '初版缺少 PEFT 搜索路径，部分 R2 GPU 空闲显存不足，R2/R3 较长会话触发了新增完整性检查读取 inference tensor 版本号的错误。另有节点 user bus 环境验收失败。R4 模型多轮测试已通过，但独立 CPU 作业排队，后续控制器迁移遇到依赖路径和 user bus 启动时序问题，未产生真实任务模型请求。所有旧证据保留，未计为质量 0。R2 cancel-async-tasks 两组正常完成并保留，其余五题在 R5 使用未改动的 R4 模型代码、整合作业控制后从新环境运行。每个 GPU 先验证分层 mask、native 等价性，以及 native/n24 三轮、超过 12 个历史 chunk 的 H12 缓存不变性。修复只改变缓存张量的版本计数可用性，不改变权重或计算公式。','',
    '环境为另一实验已实现的 Harbor 0.23.0 / Terminus2 / Apptainer 后端；R5 每题在实际节点先完成零模型调用的 install-only 验收，再加载模型。每题一个 Slurm 作业、8 个逻辑 CPU：任务控制器与模型绑定不同物理核；同题两组的分配相同。实际位置与分配以 jobs/sacct/cpu_partition 记录为准。每个任务两组由同一个常驻模型、同一 GPU 顺序运行，先后顺序交替。R2 保留的短任务使用早期独立 CPU 控制器，不能直接与其他题横向比较绝对耗时。','',
    '没有人为任务或输出时间限制；原生 40960 位置和物理内存边界仍然存在，触及边界必须记为未完成。仅一次探索性运行，任务数少，不能据此证明质量非劣或统计显著加速。','',
    f"冻结源码复核：{sum(a['frozen_sources'] for a in audit)} 个文件记录全部一致。有效请求的输入 SHA、响应完整性、实际父进程 wait 和容器清理已核对。有效模型 worker 全部正常关闭：{summary['workers_closed']}。详细 provenance、未采用尝试、Slurm 状态见 summary.json。"]
(OUT/'READOUT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(valid_trials=summary['valid_trials'],workers_closed=summary['workers_closed'],paired=paired,replay_count=len(replays))))
