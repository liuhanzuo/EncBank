"""Retain complete saved predictions without fabricating a successful generation exit."""
import collections,datetime,hashlib,json,tomllib
from pathlib import Path
r=Path(__file__).resolve().parent;out=r/'delivery/storage_failure_20260919_recurrence'
p=out/'qwen35_final_shard3_predictions.jsonl';raw=p.read_bytes();rows=[json.loads(l) for l in raw.splitlines()]
assert len(rows)==1809 and len({(x['id'],x['arm']) for x in rows})==1809
assert all(x['status']=='ok' and x['rank']==128 and x['step']==8000 for x in rows)
v=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=1809,unique=1809,statuses=dict(collections.Counter(x['status'] for x in rows)),
    benchmark_counts=dict(collections.Counter(x['benchmark'] for x in rows)),sha256=hashlib.sha256(raw).hexdigest(),
    local_backup=str(p),formal_completion=False,
    reason='Job106606 FAILED1:0 while writing complete.json.tmp; no parent_wait0 exists. Saved answers retained; no fabricated completion receipt.')
(out/'qwen35_final_shard3_backup_verification.json').write_text(json.dumps(v,indent=2)+'\n')
note='''**2026-09-19 04:25恢复未能稳定，已停止新任务调度。** 两轮通用health和节点检查通过后，新106701/106702仍于04:20在torchinductor临时目录mkdir处Errno121失败；Judge3817318又在watch_status写入时Errno121退出。PAUSE_SUBMISSIONS.persistent-storage.tmp也写入失败，暂停文件并未保存；已核实身份后仅停止CPU owner3759947，确认退出，未杀GPU任务。status.json的ACTIVE/完成数将陈旧，需以实时Slurm/进程和原始回执为准。

9B job106606生成全部1809条后，在complete.json.tmp写入时报Errno121，于04:22:54 FAILED1:0，parent_exit文件也未保存。1809条有效唯一答案已备份本地delivery/storage_failure_20260919_recurrence/qwen35_final_shard3_predictions.jsonl并核验rank128/step8000，保留原失败，不伪造正式完成。当前仅27B shard3/job106640仍RUNNING；m1-s2已归档旧106609、尚无新submission，三项失败/未排任务不盲重投。

旧stability成功计数已因真实再次失败清零，不再凭通用probe通过重复恢复。最新证据failed_recovery_0422.json；精确回执/状态需下轮实时读。需要管理员检查BeeGFS；普通账号官方health check被root限制，未尝试sudo或联系他人。监控仍保留，用于读在途106640、备份新增答案、观察存储恢复；若改进控制层，可评估将编译临时缓存移到节点本地、使用唯一临时文件的有限原子写入重试，须在相关进程停止后部署验证、科学代码不变；当前尚未实施，不能称已修复。

'''
p=r/'README_zh.md';text=p.read_text(encoding='utf-8');text=text.replace('# Qwen 新模型 H 量化及加强训练\n','# Qwen 新模型 H 量化及加强训练\n\n'+note,1);p.write_text(text,encoding='utf-8')
p=r.parent/'encbank_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md';text=p.read_text(encoding='utf-8')
text=text.replace('03:45–03:46共享盘故障后三项曾暂停；两轮稳定检查及GPU节点检查通过后，04:18已保留答案恢复四卡调度，仍待完整评测与Judge；',
    '04:18恢复后再次出现共享盘Errno121，04:22已停止CPU调度器以暂停新提交；仅27B最后分片仍在运行，完整结果与Judge尚未齐全；')
text=text.replace('剩余评测受共享盘故障影响后已于04:18恢复调度，尚待完整生成和Judge；',
    '剩余评测因反复共享盘故障暂停新提交，9B最后1809条答案虽已保存但完成记录失败，仍需正式收尾与Judge；')
p.write_text(text,encoding='utf-8')
cfg=tomllib.loads(Path('/srv/encbank/client/.codex/automations/qwen-lora-3/automation.toml').read_text(encoding='utf-8'))
header='''【最新状态，覆盖下文04:19恢复：2026-09-19 04:25，恢复再次失败】
04:18恢复未能稳定：新106701/106702在04:20 FAILED1:0，torchinductor临时目录mkdir Errno121；Judge3817318也在WAITING_GENERATIONS写watch_status时Errno121退出，9433判分保留、非API故障。PAUSE_SUBMISSIONS.persistent-storage.tmp同样Errno121，暂停文件未保存。已准确核身份并仅SIGTERM CPU owner3759947、确认退出，GPU任务未杀；新提交实际停止。owner旧launch和status.json仍在，ACTIVE/完成数是陈旧快照，必须看真实进程与Slurm。禁止把owner不存活当未知故障再启动：这是有意暂停！
9B最后job106606生成1809条后写complete.json.tmp失败，04:22:54 FAILED1:0、parent_exit文件也缺失，不伪造wait0。全部1809有效唯一rank128/step8000答案已scp本机，delivery/storage_failure_20260919_recurrence/qwen35_final_shard3_predictions.jsonl和backup_verification.json（实际文件名qwen35_final_shard3_backup_verification.json）。后续恢复只跳过保存答案或经严格CPU验证收尾，不重新生成这1809题；原job失败必须保留且不能拿手写complete冒充成功。9B其他三个最终分片已正常完成，量化与中途诊断都已交付。
当前仅27B shard3/job106640仍RUNNING；定期看实际预测推进，若完成收集真实退出/记录。27B shard0/1最新失败106701/106702，shard2已归档旧106609后等待名额、目前无新submission。原恢复工具recover_three_shards_20260919_0415.py及pause_storage_again_20260919_0422.py均已执行，禁止再次运行。证据delivery/storage_failure_20260919_recurrence/failed_recovery_0422.json。
真实再次失败已将stability成功计数清零。不要再用通用probe连续成功就重启GPU/Judge的循环，这已被实际失败证伪。继续只读监控在途任务、保留新增有效答案；未有新变化时静默。需要管理员检查BeeGFS，普通账号官方health check仅root可用，未尝试sudo、不未经授权联系他人。可进行有针对性的只读诊断，或在不改科学代码/不影响活跃进程前提下准备控制层改进（节点本地编译临时缓存、唯一临时文件有限原子写入重试），但必须具体验证后才能部署恢复；当前尚未实施，不能称已修复。不要每个heartbeat都重报同一存储故障。存储恢复须验证原失败目录/操作及节点，而非只看新建probe文件成功。
仍保持本agent最多4GPU请求，当前只占1。所有训练/原adapter量化五项/中途LongEval/BGE任务均已完成，不重跑。以下04:19恢复记录是历史，只有明确新证据下恢复才通知。Judge已有9433条保留，未来只恢复剩余既有评分，不新增probe/请求预算。

'''
args=dict(id=cfg['id'],mode='update',kind=cfg['kind'],name=cfg['name'],prompt=header+cfg['prompt'],status=cfg['status'],rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'])
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
(r/'delivery/automation_update_20260919_0425.json').write_text(json.dumps(args,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'delivery/heartbeat_current_prompt.txt').write_text(args['prompt'],encoding='utf-8')
print(json.dumps(v))
