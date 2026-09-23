"""Record a completed quantization delivery and the new, inspected storage pause."""
import datetime,json,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
paper=ROOT.parent/'comem_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md'
text=paper.read_text(encoding='utf-8')
text=text.replace('最终8,000步全五项评测正在运行，尚无完整结果；','最终8,000步全五项尚无完整结果；03:12–03:18共享盘Errno121使剩余五分片退出，当前等待存储恢复后续评；')
text=text.replace('固定第8,000步最终权重的五项benchmark正在评测，不根据中途测试分数挑选权重；','固定第8,000步最终权重的五项benchmark尚未全部完成，剩余评测受共享盘故障影响；不根据中途测试分数挑选权重；')
paper.write_text(text,encoding='utf-8')
failure=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    verified_failed_jobs=['106475','106477','106545','106546','106575'],
    failure='Shared filesystem Remote I/O error121, including unavailable Triton output file',
    recovery_health_attempts=2,recovery_actual_returncodes=[1,1],
    recovery_stage='Initial health check failed; no receipts archived, no GPU submissions, no Judge restart',
    missing_parent_receipts=['106545','106546'],
    judge_saved_decisions=9433,api_failure=False,
    cpu_summary_actual_wait=0,quant_27b_delivery_verified=True,
    recovery_tool='control/recover_storage_20260919_0312.py')
(ROOT/'delivery/storage_failure_20260919/recovery_status.json').write_text(json.dumps(failure,indent=2)+'\n')
config=tomllib.loads(Path('/srv/encbank/client/.codex/automations/qwen-lora-3/automation.toml').read_text(encoding='utf-8'))
prompt=config['prompt']
latest='''【最新状态覆盖下面所有旧快照：2026-09-19 03:25】
Qwen3.8-27B H8/H4全五benchmark已完成并正式交付FIVE_BENCHMARK_RESULTS_zh.md。原rank32/alpha32/4000，j21，H8五项99.39/97.20/50.96/73.38/50.50，Avg74.29；H4为99.40/97.20/51.08/72.81/50.30，Avg74.16；H16 Avg74.37。各7236题、14472新预测、3972 Judge均完整，errors0。03:21 CPU summarize真实wait0，后续本地独立聚合核对及采集脚本exit0。原始包与summary/exit_evidence/judge在delivery/qwen38_quant_five_benchmarks/。两模型原adapter H量化五项均已完成，不重跑/重复交付，不再次执行one-shot update_qwen38_quant_table.py。
27B rank128第4000/8000步LongEval也已完整交付：98/100/96/97/96，均值97.40，500题，四分片真实wait0+sacct0，delivery/qwen38_midpoint_longeval/。两模型8000步训练都已完成，不重训。只剩9B最终最后分片和27B最终四分片及其Judge/汇总。
新故障：03:12–03:18共享BeeGFS Errno121导致large-final-m0-s3/job106475、large-final-m1-s0/106477、s1/106545、s2/106546、s3/106575全部FAILED1:0。106475是临时PTX output无法打开，同一共享盘故障；不是OOM。106545/106546的parent_exit写入也失败，不能补造真实wait回执；Slurm失败和父子日志保留。旧owner548826自然退出；Judge2904295在WAITING_GENERATIONS保存watch_status时Errno121退出，非API故障，9433判分均保留。此时本agentGPU请求0，23/28完成5失败。
03:20三处小文件health通过，但03:22与03:24两次恢复前检查在control/health-recovery-20260919-0312.recovery.tmp重现Errno121，实际exit1。两次均未归档receipt、未提交GPU、未重启Judge；不能称已恢复。系统根盘仍约23GB、共享盘33TB，不是磁盘已满。检查证据delivery/storage_failure_20260919/{inspection.json,recovery_status.json}。
恢复工具已部署control/recover_storage_20260919_0312.py（系统/usr/bin/python3即可，无模型依赖）。每次巡检最多一次小文件健康检查/恢复准备；存储未恢复时安静等待，不盲投GPU。如果maintenance_history/storage-recovery-20260919-0312尚不存在、五旧job确实终止、owner/Judge确实不存活，可执行该工具。它在所有读写/fsync/rename检查通过后核任务锁、有效唯一预测、9433 Judge记录、失败原因，再归档错误receipt并恢复唯一owner。若history出现但恢复未完成，先读intent/recovery/真实进程，不能重跑one-shot工具。科学代码、配置、训练权重、已有答案和判分均保持。
owner恢复ACTIVE且旧Judge launch已归档后才运行launch_judge_watch.py一次。继续原4workers与原重试预算；本次是存储故障，不新增API探测或额外重试额度。存活/新队列确认之后才能通知恢复。若再次失败，保存本次新job/错误，不能套用旧job继续盲投。CPU quant汇总337170/337576已完成，勿再等待或启动它。
独立本地BGE任务仍是已完成状态，无本地GPU预留、不重跑。完整报告见下面说明。

'''
prompt=latest+prompt
args=dict(id=config['id'],mode='update',kind=config['kind'],name=config['name'],prompt=prompt,
    status=config['status'],rrule=config['rrule'],targetThreadId=config['target_thread_id'])
if 'notification_policy' in config:args['notificationPolicy']=config['notification_policy']
(ROOT/'delivery/automation_update_20260919_0325.json').write_text(json.dumps(args,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'delivery/heartbeat_current_prompt.txt').write_text(prompt,encoding='utf-8')
print(json.dumps(dict(delivery=str(paper),automation_args=str(ROOT/'delivery/automation_update_20260919_0325.json'))))
