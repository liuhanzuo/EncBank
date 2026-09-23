"""Publish the inspected recurrence and prevent repeated recovery loops."""
import json,tomllib
from pathlib import Path
r=Path(__file__).resolve().parent
note='''**2026-09-19 03:52再次故障，已暂停新提交。** 27B前三分片106607/106608/106609于03:45–03:46 FAILED1:0，分别在Triton缓存mkdir与torchinductor临时目录写入报Errno121。Judge1926367再次在WAITING_GENERATIONS的watch_status写入时Errno121退出；非API错误，9433条判分保留。新暂停标记PAUSE_SUBMISSIONS.json已实际写入。106606（9B最后分片）与106640（27B最后分片）保持运行；9B03:49已有910条预测，optional progress写入失败时跳过，不因此杀进程。检查记录delivery/storage_failure_20260919_recurrence/inspection_and_pause.json。

03:36的恢复属于历史成功，不能据此称存储目前正常。新恢复须使用这次job及现有有效输出，不能重跑recover_storage_20260919_0312.py。每次巡检最多一次check_storage_stability_20260919.py，需两次相隔至少9分钟的成功检查后再评估恢复（仍需核GPU节点和原任务退出）。该工具只做小文件mkdir/读写/fsync/rename，不提交GPU。本轮未运行它。只读BeeGFS官方health check返回“only root can use this command”，本账号无法检查服务端健康；不尝试sudo或修改集群。现有owner继续跟踪活跃任务，失败三项/Judge暂不重试。

'''
p=r/'README_zh.md';text=p.read_text(encoding='utf-8');text=text.replace('# Qwen 新模型 H 量化及加强训练\n','# Qwen 新模型 H 量化及加强训练\n\n'+note,1);p.write_text(text,encoding='utf-8')
p=r.parent/'comem_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md';text=p.read_text(encoding='utf-8')
text=text.replace('03:12–03:18共享盘Errno121使剩余五分片退出，03:36检查通过后已保留答案恢复调度，四项重新排队、第五项待空位；',
    '03:36存储恢复后曾续评，但03:45–03:46再次出现Errno121：27B前三分片失败、两项继续运行，现暂停新提交并保留全部有效答案；')
text=text.replace('共享盘故障后的剩余评测已于03:36恢复调度；','剩余评测受反复共享盘故障影响，03:50已暂停新提交，仍在推进的任务保持运行；')
p.write_text(text,encoding='utf-8')
cfg=tomllib.loads(Path('/srv/encbank/client/.codex/automations/qwen-lora-3/automation.toml').read_text(encoding='utf-8'))
header='''【最新故障覆盖后面03:37恢复状态：2026-09-19 03:52】
共享BeeGFS又出现Errno121。27B large-final-m1-s0/106607、s1/106608、s2/106609已03:45–03:46 FAILED1:0，Triton/torchinductor临时文件目录写入失败；Judge1926367在WAITING_GENERATIONS写watch_status时再退出，非API服务错误。9433条判分保留，不加API probe/重试额度。新暂停标记PAUSE_SUBMISSIONS.json已于03:50写入，owner1861243仍在跟踪。106606（9B最后分片）与106640（27B s3）仍运行，9B03:49为910条且推进，optional progress Errno121由原wrapper跳过；不要因progress旧或警告就杀任务。
先运行collect_recovery_status.py核实新状态，保存新失败的短日志/实际Slurm退出。证据delivery/storage_failure_20260919_recurrence/inspection_and_pause.json。hold_after_storage_recurrence_20260919.py已执行，不需要重复。前三27B失败receipt仍原位；不要套旧one-shot恢复（recover_storage_20260919_0312.py已执行完）。暂不重启Judge或失败GPU，单次health通过已不能说明稳定。
下次巡检起每次最多一次本地check_storage_stability_20260919.py；它仅对cache/results/large-final/Judge目录做小文件mkdir/读写/fsync/rename，至少9分钟间隔、连续两次成功才将eligible_for_inspected_recovery设true。记录delivery/storage_failure_20260919_recurrence/stability.json及历史check文件。当前本轮尚未运行此工具。连续未恢复且无新变化保持安静；不要每次重报同一个故障。资格不是GPU节点健康证明：恢复前仍需核最新节点/旧job已结束、worker锁和有效唯一预测；有活动任务时保留，不要求取消它们来取全局空队列。
真正稳定后另写针对最新失败job的恢复准备：先暂停/停止CPU owner并确认锁释放（不杀GPU worker），保留原receipt/错误及预测，只归档已经终止的失败任务，保留活跃任务receipt；新owner遵循最多4 RUNNING+PENDING。必要时先从已有GPU allocation做只读或小文件检查，不申请第五卡。新错误变化及时调查，不能不断套脚本重投。如果官方存储诊断必要，当前账号运行/usr/sbin/beegfs health check返回only root can use this command，不尝试sudo或更改服务；可告知需管理员检查BeeGFS，但未经用户明确授权不要发消息给他人。
同一原则恢复Judge：确认旧进程退出、缓存9433或后续值完整、无不确定API请求后归档旧launch/错误，再启动唯一4worker；存储恢复不等于API失败，不增预算。两模型训练/量化全五项/中途LongEval、本地BGE及旧日志报告均已完成，不重跑。最终大LoRA仍未完成。下文03:37属于已告知的历史恢复，不能重复通知为当前恢复。

'''
args=dict(id=cfg['id'],mode='update',kind=cfg['kind'],name=cfg['name'],prompt=header+cfg['prompt'],
          status=cfg['status'],rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'])
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
(r/'delivery/automation_update_20260919_0352.json').write_text(json.dumps(args,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'delivery/heartbeat_current_prompt.txt').write_text(args['prompt'],encoding='utf-8')
print('Recorded storage recurrence and prepared monitoring update.')
