"""Record confirmed admission recovery without claiming pending jobs have run."""
import json,tomllib
from pathlib import Path
r=Path(__file__).resolve().parent
p=r.parent/'comem_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md'
text=p.read_text(encoding='utf-8')
text=text.replace('03:12–03:18共享盘Errno121使剩余五分片退出，当前等待存储恢复后续评；',
    '03:12–03:18共享盘Errno121使剩余五分片退出，03:36检查通过后已保留答案恢复调度，四项重新排队、第五项待空位；')
text=text.replace('剩余评测受共享盘故障影响；','共享盘故障后的剩余评测已于03:36恢复调度；')
p.write_text(text,encoding='utf-8')
cfg=tomllib.loads(Path('/srv/encbank/client/.codex/automations/qwen-lora-3/automation.toml').read_text(encoding='utf-8'))
old=cfg['prompt'];boundary='继续监控并及时恢复本线程已授权的Qwen3.5-9B/Qwen3.8-27B缓存H量化与大LoRA实验。'
assert old.count(boundary)==1
body=old[old.index(boundary):]
latest='''【最新状态覆盖下面所有旧快照：2026-09-19 03:37恢复成功】
03:12–03:18共享BeeGFS Errno121的恢复已完成。03:36 control/recover_storage_20260919_0312.py真实exit0：在write/read/fsync/rename成功、五旧job已终止、锁已释放之后归档错误receipt并恢复唯一owner。maintenance_history/storage-recovery-20260919-0312已存在且recovery.json齐全，本地delivery/storage_failure_20260919/recovered.json。此one-shot工具禁止再次运行！106475/106477/106545/106546/106575均是历史失败；106545/106546旧parent_exit因写盘失败缺失，保留原日志与Slurm失败，不伪造回执。
唯一owner新PID1861243，ACTIVE、23/28完成0失败。新GPU请求4：106606=large-final-m0-s3，106607=m1-s0，106608=m1-s1，106609=m1-s2；03:36均PENDING(Priority)，排队不等于运行。27B m1-s3待空位自动接续。9B最后分片已保存659条有效唯一预测原样保留，恢复评测自动跳过；27B四分片无已生成答案。最多4本agentGPU请求，不碰其他agent。
Judge由已授权launch_judge_watch.py恢复为PID1926367，4workers、gpt-6-astra/low；旧9433条已判完整保留。本次是存储故障，不是API错误，不增加请求重试预算、不做新probe。启动时重新读已完成输入/缓存可能需几分钟；看真实进程/短日志，不能因旧progress就再起watcher。后续若又失败，用新job/新错误检查，绝不套旧恢复脚本盲投。所有训练已完成，不重训。
Qwen3.8-27B原adapter H8/H4全五benchmark已交付FIVE_BENCHMARK_RESULTS_zh.md：rank32/alpha32/4000 j21，H8五项99.39/97.20/50.96/73.38/50.50 Avg74.29，H4为99.40/97.20/51.08/72.81/50.30 Avg74.16，H16 Avg74.37。各7236题，14472新预测、3972 Judge，errors0；CPU汇总真实wait0且本地采集/独立聚合核对exit0。交付delivery/qwen38_quant_five_benchmarks/。两模型原adapter H16/H8/H4全五项均已完成，不重跑、不重复交付，不再执行one-shot update_qwen38_quant_table.py。旧CPU汇总337170/337576也已完成。
27B rank128中途第4000/8000步LongEval也已完整交付：98/100/96/97/96，宏平均97.40，500题四分片wait0+sacct0，delivery/qwen38_midpoint_longeval/。9B中途97.20同样完成。两模型8000步训练均完成。剩余仅两模型最终8000步全五项评测/Judge/本地汇总，完整后补同一FIVE文件并与原rank32量化分列。
独立本地BGE补测/复用更新整理均已交付，不重跑、不预留本地GPU。下文时间/PID快照已旧，以本段及实时检查为准。正常排队、启动、进展无行动需要时静默；新增失败/恢复/完整交付才通知。

'''
args=dict(id=cfg['id'],mode='update',kind=cfg['kind'],name=cfg['name'],prompt=latest+body,
          status=cfg['status'],rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'])
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
(r/'delivery/automation_update_20260919_0337.json').write_text(json.dumps(args,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'delivery/heartbeat_current_prompt.txt').write_text(args['prompt'],encoding='utf-8')
print('Recorded recovery and prepared monitor update.')
