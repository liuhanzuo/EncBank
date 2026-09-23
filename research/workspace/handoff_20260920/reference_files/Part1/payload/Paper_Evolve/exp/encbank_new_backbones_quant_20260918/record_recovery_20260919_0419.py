"""Record the inspected 04:18 recovery while retaining all previous failure evidence."""
import json,tomllib
from pathlib import Path
r=Path(__file__).resolve().parent
p=r/'README_zh.md';text=p.read_text(encoding='utf-8')
note='''**2026-09-19 04:19恢复更新。** 04:03和04:14两轮小文件检查连续通过；现有106640 allocation中的CPU检查也验证GPU节点共享盘读写/fsync/rename成功。新的recover_three_shards_20260919_0415.py实际exit0，仅归档106607–106609三项已终止失败，保留仍运行的106606/106640及其receipt/答案。旧CPU owner已确认停止并释放锁，新唯一owner3759947恢复ACTIVE。106701/106702已RUNNING（27B shard0/1），shard2待空位；原9B shard3于04:18有1716/1809、27B shard3有640/1809。两旧任务近期有optional progress写入警告但答案持续推进，不能称共享盘完全消除抖动。Judge已恢复PID3817318，已有9433判分不变；模型、prompt和重试预算不变。

本地证据delivery/storage_failure_20260919_recurrence/recovered_0418.json，远端maintenance_history/storage-recovery-20260919-0415/。新one-shot恢复已执行，不可重复；03:52暂停为历史状态，PAUSE_SUBMISSIONS已归档。若再故障按最新job核验，先保持有效答案，不重跑已完成训练/量化。

'''
text=text.replace('# Qwen 新模型 H 量化及加强训练\n','# Qwen 新模型 H 量化及加强训练\n\n'+note,1);p.write_text(text,encoding='utf-8')
p=r.parent/'encbank_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md';text=p.read_text(encoding='utf-8')
text=text.replace('03:36存储恢复后曾续评，但03:45–03:46再次出现Errno121：27B前三分片失败、两项继续运行，现暂停新提交并保留全部有效答案；',
    '03:45–03:46共享盘故障后三项曾暂停；两轮稳定检查及GPU节点检查通过后，04:18已保留答案恢复四卡调度，仍待完整评测与Judge；')
text=text.replace('剩余评测受反复共享盘故障影响，03:50已暂停新提交，仍在推进的任务保持运行；',
    '剩余评测受共享盘故障影响后已于04:18恢复调度，尚待完整生成和Judge；')
p.write_text(text,encoding='utf-8')
cfg=tomllib.loads(Path('/srv/encbank/client/.codex/automations/qwen-lora-3/automation.toml').read_text(encoding='utf-8'))
prompt='''继续监控和恢复本线程已授权的Qwen3.5-9B/Qwen3.8-27B实验，每10分钟巡检。正常启动/排队/推进或既有故障未变时保持安静；新增故障、实际恢复成功、完整结果交付或需要用户处理才通知。全部最终评测、Judge及本地交付完成后删除本专属qwen-lora-3，不碰其他agent任务。不改论文、不push。

【当前状态：2026-09-19 04:19】
根目录本地F:/Paper_Evolve/exp/encbank_new_backbones_quant_20260918，远端/srv/encbank/qencbank_align_codex_20260911/encbank_new_backbones_quant_20260918，SSH gpu-node1。先读README_zh.md顶部及monitor_state.json，运行collect_recovery_status.py读取当前Slurm、owner、短日志、Judge。monitor里recovery字段仍含9月18日旧恢复，勿误当最新；以owner.recovery_evidence为准。轻量读状态，不反复读/哈希大checkpoint。
03:45–03:46的Errno121在04:18恢复：04:03/04:14小文件健康检查连续两次成功；从106640已有allocation的CPU-only step验证节点共享盘mkdir/读写/fsync/rename成功。control/recover_three_shards_20260919_0415.py真实exit0，仅归档27B前三分片106607/106608/106609已终止的失败，保留正在运行的106606（9B最后分片）与106640（27B shard3）及其全部receipt/有效答案。旧owner1861243已确认停止释放锁，新唯一owner3759947，ACTIVE、23/28完成0失败。新106701=27B shard0、106702=27B shard1已RUNNING；shard2待名额自动接续，四卡总上限不变。04:18原两项进度1716/1809和640/1809，近期optional progress仍曾警告Errno121但答案推进，不杀进程、不称存储完全无抖动。恢复证据maintenance_history/storage-recovery-20260919-0415/recovery.json及本地delivery/storage_failure_20260919_recurrence/recovered_0418.json。PAUSE_SUBMISSIONS已归档，不再暂停。该one-shot恢复已经执行，禁止再运行。
Judge已由launch_judge_watch.py恢复PID3817318，4workers、gpt-6-astra/low，已有9433判分保留；启动先重读11个已完成generation任务，可能数分钟，不因旧progress起第二watcher。本次不是API故障，不加probe或额外请求预算。CPU汇总337170/337576和历史owner/Judge PID均已结束，不再等待。

【完成与剩余】
两模型rank128/alpha128/8000训练均完成，不重训；27B于00:58:59完成32,768,000tokens并有真实wait0/Slurm0，delivery/qwen38_large_training_complete。4000/8000中途LongEval均完整交付500题：9B为98/99/96/97/96，均值97.20；27B为98/100/96/97/96，97.40。中途只是诊断，固定8000最终权重，不按测试分数选checkpoint。
原rank32/alpha32/4000的H16/H8/H4量化全五benchmark，两模型均已完整交付F:/Paper_Evolve/exp/encbank_new_backbones_formal_20260915/delivery/FIVE_BENCHMARK_RESULTS_zh.md。9B H16/H8/H4 Avg65.89/65.94/66.13；27B为74.37/74.29/74.16。27B H8五项99.39/97.20/50.96/73.38/50.50，H4为99.40/97.20/51.08/72.81/50.30。每模型14472新预测+3972 Judge，各臂7236，0错误，CPU逐题复算、完整分母和本地独立宏平均核验均完成。原始包delivery/qwen35_quant_five_benchmarks/与qwen38_quant_five_benchmarks/。不要重跑或重复交付，也不要重复one-shot update_qwen38_quant_table.py。
剩余仅两模型固定8000步最终大LoRA的五benchmark生成、LoCoMo Judge和本地完整汇总。完整后做原评分CPU复算、唯一记录/分母/实际parent_wait0+Slurm0:0核对、收集原始记录，补同一FIVE文件并与原rank32量化分列。未完成不填0、不外推最终。长CPU汇总用后台父进程保存真实wait回执，避免SSH中断误判GPU失败。

【固定科学配置】
Qwen3.5-9B j6/L32，Qwen3.8-27B j21/L64。原rank32/alpha32/4000只改变H精度，group64 H8/H4，模型权重BF16、sink/query/在线KV不量化；各100题H16 hidden/token精确parity已通过后复用原H16。新rank128/alpha128/8000：PG19 64篇、seed42/window4096/chunk512/top64双向KL(lambda.6)、同teacher、suffix Linear、lr1e-4 AdamW(.9,.95)、warmup50、clip1、cosine8000。4000中途非原cosine4000终点的纯rank对照。每臂7236题：RULER1500、LongEval500、LongBench1150、BABILong2100、LoCoMo1986。LoCoMo C1–4语义1540+C5本地拒答446按题加权，报告分类和全1986。旧prompt-pack可能含部分问题，不能称独立文档预处理；质量elapsed不当TTFT/tps。实际GPU按L20D设备报告，勿改名称来暗示硬件。

【调度与恢复】
effective_plan.json28项，科学plan.json保持。独立最多4本agent RUNNING+PENDING GPU请求，每项1卡，其他agent额度独立，绝不操作其任务。共享qencbank_gpu_admission.lock及唯一owner决定接续；owner活跃时不旁路sbatch/起第二owner。本地任务已完成，local_gpu_reservation=null。
先查真实日志/Slurm，区分存储/节点/API/OOM/科学错误。活跃进程即使慢也先查长输入、启动或I/O，不能凭旧progress杀任务。io_resilience/durable_runner只对相同原子写入有限重试，optional progress可跳过，checkpoint/complete必须真实落盘；JSONL追加禁止盲重试。保存有效唯一预测及原错误，不把失败计0。失败恢复只从真实checkpoint/optimizer/RNG或已保存答案续行，训练完成不再加载checkpoint检查。
如果共享盘再发生新故障，写PAUSE_SUBMISSIONS.json暂停新提交，保留活跃GPU和状态跟踪，不删别人或系统文件。使用check_storage_stability_20260919.py每巡检最多一次，>=9分钟间隔连续两次成功才考虑恢复；旧成功如果发生新故障必须重新计数，不能沿用故障前的stability记录。必要时用现有GPU allocation的CPU-only step检查节点，不申请第五卡。只有准备恢复时停止CPU owner、确认锁释放；只归档已终止失败receipt，保留活跃任务，使用新job与新错误写新恢复工具。恢复后核新owner实际存活和新Slurm受理，才通知成功。提交不确定先查squeue/sacct，不盲投。
所有one-shot历史恢复都已执行，不再运行：recover_storage_1722.py、recover_after_filesystem.py、recover_submission_2219.py、hold_submissions_2234.py、resume_after_root_recovery_2235.py、install_bge_reservation.py、reserve_local_bge_now.py、finish_pending_deferral_remote.py、recover_storage_20260919_0312.py、hold_after_storage_recurrence_20260919.py、recover_three_shards_20260919_0415.py。历史错误在maintenance_history和runs/attempt_history。106545/106546旧parent_exit写失败，保留Slurm/日志，不补造wait回执。beegfs health check需要root，当前账号只读命令也返回only root can use this command；不尝试sudo或修改服务，不未经授权联系管理员。

【Judge约束】
midcache-locomo-gpt6-astra-v1、gpt-6-astra/low、4workers、每请求180s；只在本批generation完成后评分，成功stimulus缓存复用。当前恢复自存储故障，没有新增API预算。真实Codex Responses曾可用，/models403不代表评分不可用，service_checks/20260918T120832Z/report.json一条正式未缓存题3.26s成功且已复用。正常不探测或加并发；实际请求失败才最多每3小时一次正式未缓存题probe_judge_recovery.py --direct-codex，或者用户新授权。
同stimulus跨重启通常最多3次。transport_recovery.json两条历史503额外一次各已用完，不追加。dispatch存在而无result为不确定，禁止盲重发；401/403/quota不重试。过载/超时增加需保存结果、排空旧watcher并确认退出，必要时4降2，绝不双watcher。密钥只内存、不打印落盘。判分数含复用，不等于API调用数。历史主动并发调整KeyboardInterrupt不是新故障，resume_judge_after_probe.py也已执行勿重复。

【独立本地任务已完成，不重启】
F:/Paper_Evolve/exp/encbank_bge_matched_20260918/REVIEWER_RESULTS_zh.md及summary/verification已交付。BGE-Qasper A三轮校准及200题三臂600答案真实wait0。raw k6/k8/Encbank k12 TTFT510.827/633.882/577.956ms，F1 4.54653/5.16746/11.81033，未满足预定±5%，只能称邻近工作点；不追加k7/调样本/预算。配对148context簇CI已交付，600正式答案自然封顶128、无空答错误；旧4题Question片段离线边界已修正，原分数不可混用。文档bank/index准备完，store-ready TTFT含真实query BGE/search/fetch/query Write/prefill，不是冷E2E。controller85660和GPU已退出，reservation已释放。
已有2400请求、40更新的B/C整理已交付：Write均值.893752s，Q10 raw/Encbank18.946839/11.165820s，Q80 153.998186/82.306705s，均10/10获益；观察80问内crossover中位2范围1–3。输出长度不同，非固定输出加速。维护为分项核算，追加.064755/.747659s、插入.362858/.697493s、40/40 H相同；含共同全BM25重建+search，非增量BM25或连续wallclock，未计token化/持久IO/元数据事务，非改事实QA。KV独立LoRA已对齐：LongEval74.67/72.67、Qasper4.25/11.37，不重训、不跨环境算速度。条件LoCoMo200对新语义重评分未启动，不自行加API calls。
用户已缩减：新的j训练扫描、生产并发、更多agent benchmark、lower-Write重训均不跑；44项ctrl-*已撤回未提交，WITHDRAWN_BY_USER.json可核。dots3-note-prev/GLM5.3暂缓。不改论文、不push。
'''
args=dict(id=cfg['id'],mode='update',kind=cfg['kind'],name=cfg['name'],prompt=prompt,status=cfg['status'],rrule=cfg['rrule'],targetThreadId=cfg['target_thread_id'])
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
(r/'delivery/automation_update_20260919_0419.json').write_text(json.dumps(args,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'delivery/heartbeat_current_prompt.txt').write_text(prompt,encoding='utf-8')
print('Recorded the current recovery and prepared the consolidated monitoring instructions.')
