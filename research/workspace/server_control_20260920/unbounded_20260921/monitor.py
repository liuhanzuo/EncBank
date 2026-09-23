"""Refresh local summaries by running the server's read-only observer."""
import argparse,datetime,json,subprocess
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ssh-host',default='gpu-node1',help='SSH transport host for shared-file reads; actual-node process checks remain separate.')
args=parser.parse_args()
H=Path(__file__).resolve().parent
remote='/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920/unbounded_20260921/observe_remote.py'
r=subprocess.run(['ssh',args.ssh_host,'/srv/encbank/qcomem_runtime_20260911/python312/bin/python',remote],capture_output=True,text=True,encoding='utf-8',check=True)
d=json.loads(r.stdout);d['observer_ssh_host']=args.ssh_host;out=H/'monitoring';out.mkdir(exist_ok=True)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');raw=json.dumps(d,indent=2,ensure_ascii=False)+'\n'
(out/(stamp+'.json')).write_text(raw,encoding='utf-8');(out/'latest.json').write_text(raw,encoding='utf-8')
lines=['COMem 三组服务器续跑（'+stamp+'）','',d['queue'].strip(),'','方法 | 新计划 | 本轮通过/失败 | 其中上下文失败 | 累计已核 | 累计通过/失败 | 请求/回复','--- | ---: | ---: | ---: | ---: | ---: | ---:']
for a,x in d['arms'].items():lines.append(f"{a} | {x['planned']} | {x['new_passed']}/{x['new_failed']} | {x['new_context_failures']} | {x['cumulative_verified']}/89 | {x['cumulative_passed']}/{x['cumulative_failed']} | {x['requests']}/{x['responses']}")
lines+=['','按2026-09-21用户最新口径，原生上下文容量耗尽计失败/0分，同时保留NATIVE_CONTEXT_CAPACITY标记、原始异常及回复SHA，不改原始verifier结果。']
for a,x in d['arms'].items():
    for task in x['tasks'].values():
        if task['state']=='verified_context_failure':lines.append(f"- {a} / {task['task']}：失败（原生上下文容量），耗时{task['trial_seconds']/60:.1f}分钟。")
if d.get('runs'):
    lines+=['','分片 | 作业 | 已核/分配题数 | 当前活跃题','--- | --- | ---: | ---:']
    for rid,r in d['runs'].items():lines.append(f"{rid} | {r.get('job_id') or '待名额，尚未提交'} | {r['new_verified']}/{r['planned']} | {len(r['active'])}")
    lines+=['','总并发上限32，每卡上限8；实际并发以各分片活跃题之和为准。原作业正在结束已启动题，后续分片由服务器CPU调度器按4GPU上限自动提交。']
(out/'progress.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(queue=d['queue'],dispatcher_failure=d.get('dispatcher_failure'),arms={a:{k:x[k] for k in ['job_id','planned','retained','new_verified','new_context_failures','new_passed','new_failed','cumulative_verified','active','requests','responses','worker_failure','owner_failure','integrity_errors','ownership_errors']} for a,x in d['arms'].items()}),ensure_ascii=False))
