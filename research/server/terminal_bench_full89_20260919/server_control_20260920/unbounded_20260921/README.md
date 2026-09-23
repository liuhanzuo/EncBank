# Encbank 三组服务器续跑：已提交

最新扩容（2026-09-21）：按用户要求从2张扩至最多4张GPU，每卡8题，总并发上限32。新k12A117352、k48A117353已提交；原116712/116713只让已启动题自然完成，待办已互斥移交。CPU调度作业117351会在名额释放后提交已准备的k12B/k48B。当前身份与唯一归属以scale4_20260921/registry.json及各submission.json为准，详见SCALE4_20260921.md；下表是扩容前的初始三组快照。

2026-09-21用户授权关闭旧控制器、取消人为生成token和任务/请求时间上限，重测截断题并完成未完成题。模型、checkpoint、新实验环境和控制器均在服务器。

| 方法 | 唯一Slurm作业 | 新计划题数 | 保留未截断历史结果 | 并发上限 |
|---|---:|---:|---:|---:|
| Dense | 115822 | 18 | 71（49通过、22未通过） | 8 |
| Encbank k12 | 116712 | 75 | 14（全通过） | 8 |
| Encbank k48 | 116713 | 69 | 20（18通过、2未通过） | 8 |

三个作业各申请1张GPU，实际核准时间均UNLIMITED。首次巡检均PENDING/Priority、零模型请求；不能把提交说成开始推理。本线程最多4个运行及排队GPU请求，不统计或干预独立Reader研究线程。禁止重复提交。

远端根目录为 /srv/encbank/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920，子目录分别为dense_unbounded_20260921、k12_unbounded_20260921、k48_unbounded_20260921，以各自submission.json为准。2026-09-21用户要求共享磁盘的任意可用节点均可排队：k12/k48已解除固定节点限制；首次异节点启动遇到用户服务未就绪，零模型调用，修复后现作业116712/116713；Dense继续当前运行。详情见ANY_NODE_20260921.md。

## 只读巡检

在当前电脑运行：

```powershell
python /srv/encbank/workspace/server_control_20260920/unbounded_20260921/monitor.py
```

经SSH执行远端unbounded_20260921/observe_remote.py，更新本地monitoring/latest.json与progress.md，不提交、取消、重跑或修改实验。核查Slurm、请求/回复SHA、评分、异常、逐题父wait、Encbank H不变/会话释放和耗时。最终关闭还需模型/所有者真实父退出、容器与GPU释放核验。控制器唯一身份和GPU占用按实际节点、作业号、worker_ready的GPU UUID定位，不把其他研究GPU当成本工作。

心跳encbank每小时一次，无变化且无可操作问题时安静，新题完成/阶段切换/故障/需用户行动时通知。所有新计划闭合并裁定、汇总耗时后停止。正常结果不自动重跑，不因耗时长终止。

## 协议与验收

取消32768生成上限、12000秒请求保护、任务/评分/准备/模型启动/空闲计时上限，记录耗时。控制链连通及容器挂载故障诊断仍有短保护，不作为题目总时限。

原生位置容量262144和物理显存仍有限。Dense使用原生剩余上下文所需API参数；Encbank按真实读取序列检查位置，不再按已归档H的历史总长度截断。按用户2026-09-21最新口径，原生上下文容量耗尽记失败/0分，单列NATIVE_CONTEXT_CAPACITY原因；保留原始verifier值和异常，不改正在运行的推理。

Dense预留8个完整上下文（2097152 KV tokens），同配置缓存实测2569933 tokens。Encbank最多8会话/8路decode，按权重、H、最坏KV加32GiB余量准入，分配上限248GiB、设备保护256GiB；实际路数可能低于8，不通过缩短回复满足显存。三组共享主机任务预约上限200GiB，分别检查作业内存，保持各题原CPU/内存限制；每作业32CPU、160GiB。

CPU验收115553（unbounded_env_qualification_r3_20260921）实际走89题Harbor/Terminus安装路径，89/89通过、0模型调用，不计benchmark成绩。修复APT缓存权限及镜像工作目录适配；失败的115484、115518保留。三个新作业均通过服务器和模型预检，k12/k48还核验adapter身份。

已用源码不可热改。templates_final.tar.gz及launch_records保存最终代码。build_unbounded_templates.py只是初版生成器，不可覆盖最终版。

## 旧任务及证据

旧机控制器/自动续跑已由用户停止。新端独立核验Encbank包14672文件、Dense补充包3603文件的字节数及SHA，完整证据留服务器。selection.json是每组89题互斥分区（新计划+保留结果），SHA写入提交，本地副本在launch_records/selection.json。

旧112400/112403已COMPLETED，真实模型父wait0及会话释放已保存。114684在k12关闭后取消分配，k12模型wait0、两会话释放；本机batch终止后owner缺父wait，缺口保留、不伪造。最后未提交请求记用户切换中断，Docker容器已关闭，k48未开始。115237已COMPLETED，题目及owner/holder实际wait均保存：保留torch两题，mcmc/rstan因协议切换中止后纳入新Dense。

生成截断重测包括Dense旧6题加Docker regex；k12旧distribution-search、feal-linear-cryptanalysis，加未完成Docker regex；k48旧distribution-search、feal-differential-cryptanalysis、headless-terminal。旧汇总answers不完整，以原生result逐次元数据为准，最终通过也不忽略截断。

原待裁定Dense5题及k12三题均核实AgentTimeoutError，按新授权重测。旧mteb的错误断网8调用结果保留为环境无效，修复后进入新Dense，不计正常0分。旧上下文失败独立标注，新协议仍可能遇到物理容量边界。旧12个Dense共享批次缺独立题级父wait，保留原批次退出130及证据边界，历史保留数不是全89严格完成声明。

2026-09-21统计口径更新：见CONTEXT_FAILURE_POLICY_20260921.json。原生上下文容量失败纳入已核和失败数；其他基础设施异常继续单独核验，不一概计0分。旧历史快照保留当时口径，以最新monitoring/progress.md和latest.json为准。

2026-09-21 21:04阶段完成：Dense115822已于21:03:00 COMPLETED，18/18新尝试闭合（12通过、6失败，其中4原生上下文失败），累计89/89、61通过28失败。936回复SHA、实际父退出、18容器关闭和GPU释放已确认。详见monitoring/heartbeat_20260921_2100_summary.md。k12 116712和k48 116713继续各8题并发；当前使用2GPU，巡检继续。Dense不得重启。

2026-09-22故障更新：k12A117352显存分配失败（8题中断），k12B120001在模型前60秒依赖导入检查失败（32题未启动），k48B120002已运行。详见[故障证据与待恢复项](INCIDENT_K12_20260922.md)。禁止把旧active快照或零模型调用当作完成/正常0分。独立内核原型尚未部署，临时试验GPU已释放。
