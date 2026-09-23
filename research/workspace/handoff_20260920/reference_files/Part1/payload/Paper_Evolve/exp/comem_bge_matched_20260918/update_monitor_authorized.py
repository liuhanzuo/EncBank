"""Prepare the existing heartbeat update; actual mutation goes through app tool."""
import json,tomllib
from pathlib import Path
import config
x=tomllib.loads(Path('/srv/encbank/client/.codex/automations/qwen-lora-3/automation.toml').read_text(encoding='utf-8'))
p=x['prompt']
start=p.index('本地唯一controller已00:19启动')
end=p.index('\n\nA执行：',start)
new='''本地启动更新（2026-09-19 00:39，覆盖此前严格桌面门槛）：用户明确“现在也可以算是空闲了 直接开始测吧”，授权约5.5GiB桌面占用情况下启动。本实验authorized_admission.py仅在自己的进程中把准入阈值设为已有占用<7GiB、nvidia-smi实际free>=24GiB，并继续45秒稳定、排除其他Python GPU、共享exp/results/.gpu.lock和27GiB allocator cap。共享gpu_gate.py文件及其他进程默认策略没改。desktop_admission_authorization.json保存授权，不要恢复旧<5GiB/free26GiB门槛或要求用户再次确认。仍不关闭桌面/其他应用。旧等待controller94264/99584已停止且当时没有GPU子进程；新实际controller85660、wrapper87724，校准worker实际97036、wrapper98748（PID仅为启动快照，读实时身份）。已经启动第一个校准进程，是否已有测量以progress/records和真实进程为准。

总额度仍本agent最多4卡，包括本地。remote local_gpu_reservation.json已预留1卡并由controller接管，远端最多3请求。为了立即启动，仅对仍PENDING的本线程job105748（large-final-m0-s2）使用Slurm服务端PENDING过滤撤回，sacct确认CANCELLED且Start=None，绝未开始；原submission已归档到maintenance_history/local-bge-start-20260919，移出runs后由原唯一owner在本地release后自动重新排队。远端三个RUNNING任务没有中断。不重复执行reserve_local_bge_now.py/finish_pending_deferral_remote.py/install_bge_reservation.py。前者一次因Slurm Start字段为None而在收回receipt时断言，后者已仅完成归档；这是已处理的控制记录问题，不是科学失败。

远端唯一owner control/coordinator.py保持PID548826，以实时身份为准。通过control/bge_reservation.py在共享admission锁内预留/释放；本地GPU子进程全部真实wait后release，远端回到4。如果controller退出，先核GPU子进程与全局锁是否仍活跃，不能提前释放造成第5卡；无法确认时保留并调查。不要另起队列。模型加载与45秒稳定期间尚无records属正常，只有实际测量才报已测数量。'''
p=p[:start]+new+p[end:]
p=p.replace('2026-09-19 00:20 最新用户范围','2026-09-19 00:39 最新用户范围',1)
params=dict(mode='update',id=x['id'],kind=x['kind'],name=x['name'],prompt=p,rrule=x['rrule'],status=x['status'],targetThreadId=x['target_thread_id'])
(config.ROOT/'automation_update.json').write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared current authorization and original monitoring instructions.')
