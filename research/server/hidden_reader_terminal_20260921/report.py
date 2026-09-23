"""Read-only partial/final summary; infrastructure failures never become quality zero."""
import datetime,json,statistics,subprocess
from common import ROOT,PLAN as P,save,sha
def load(p):return json.loads(p.read_text()) if p.exists() else None
def median(xs):return statistics.median(xs) if xs else None
rows=[];paired=[]
for task in P['tasks']:
    taskrows={}
    for arm in P['arms']:
        directory=ROOT/'results'/(task+'--'+arm)
        result=load(directory/'actual_result.json');receipt=load(directory/'execution_receipt.json');wait=load(ROOT/'jobs'/('wait-'+task+'--'+arm+'.json'))
        requests=[]
        for path in sorted((ROOT/'mailbox'/task).glob(arm+'_*.request.json')):
            response=load(path.with_name(path.name.replace('.request.','.response.')))
            if response:
                assert response['request_sha256']==sha(path)
                requests.append(response)
        valid=bool(result and receipt and wait and wait['exit_code']==0 and wait['actual_parent_wait'] and receipt['closure'] and receipt['closure']['cgroup_empty'] and not result['exception_info'] and result['verifier_result'] and requests and all(r['status']=='ok' for r in requests))
        rewards=result['verifier_result']['rewards'] if valid else None
        row=dict(task=task,arm=arm,valid=valid,rewards=rewards,
            exception=result.get('exception_info') if result else None,wall_seconds=receipt['elapsed_seconds'] if receipt else None,
            calls=len(requests),generated_tokens=sum(r.get('generated_tokens',0) for r in requests),
            model_seconds=sum(r.get('request_seconds',0) for r in requests),
            prefill_seconds=sum(r.get('prefill_seconds',0) for r in requests),
            write_seconds=sum(r.get('write_seconds',0) for r in requests),
            selection_seconds=sum(r.get('selection_seconds',0) for r in requests),
            decode_seconds=sum(r.get('decode_seconds',0) for r in requests),
            history_calls=sum(r.get('selected_chunks',0)>0 for r in requests),
            full12_history_calls=sum(r.get('selected_chunks',0)==12 for r in requests),
            selected_chunk_counts=[r.get('selected_chunks',0) for r in requests],
            prefill_share_of_model_time=(sum(r.get('prefill_seconds',0) for r in requests)/max(1e-12,sum(r.get('request_seconds',0) for r in requests))) if requests else None)
        rows.append(row);taskrows[arm]=row
    if all(r['valid'] for r in taskrows.values()):
        a,b=taskrows['native'],taskrows['n24']
        paired.append(dict(task=task,wall_ratio=b['wall_seconds']/a['wall_seconds'],model_time_ratio=b['model_seconds']/a['model_seconds'],
            same_generated_token_count=a['generated_tokens']==b['generated_tokens'],native_rewards=a['rewards'],n24_rewards=b['rewards']))
replays=[]
for task in P['tasks']:
    data=load(ROOT/'pairs'/task/'replay.json') or []
    for rid in sorted({r['source_request'] for r in data}):
        group={arm:[r for r in data if r['source_request']==rid and r['arm']==arm] for arm in ['native','n36','n24']}
        if not all(len(x)==2 for x in group.values()):continue
        times={arm:{key:median([r[key] for r in group[arm] if r.get(key) is not None]) for key in ['prefill_seconds','memory_build_seconds','decode_seconds','request_seconds']} for arm in group}
        replays.append(dict(task=task,request=rid,selected_chunks=group['native'][0]['selected_chunks'],query_tokens=group['native'][0]['query_tokens'],times=times,
            n24_to_native_prefill=times['n24']['prefill_seconds']/times['native']['prefill_seconds'],
            n24_to_n36_build=times['n24']['memory_build_seconds']/times['n36']['memory_build_seconds']))
all_closed=all(load(ROOT/'jobs'/('wait-'+task+'--'+arm+'.json')) for task in P['tasks'] for arm in P['arms'])
workers_closed=all((load(ROOT/'pairs'/task/'parent_exit.json') or {}).get('exit_code')==0 for task in P['tasks'])
summary=dict(at=datetime.datetime.now().astimezone().isoformat(),all_live_closed=all_closed,workers_closed=workers_closed,valid_trials=sum(r['valid'] for r in rows),expected_trials=12,rows=rows,paired=paired,replays=replays)
save(ROOT/'summary.json',summary)
lines=['# 真实Terminal-Bench：Encbank联合深度实验','',f"更新时间：{summary['at']}。有效完成{summary['valid_trials']}/12组。",'',
    '6个任务、Qwen3-8B同一Encbank LoRA、贪心非thinking生成。native是原Encbank联合prefill；n24只让历史第13～24层联合，25～36层按chunk计算。每个任务两组用同一GPU及独立新容器。与另一个27B全量补测分开。','',
    '| 任务 | 配置 | 有效 | 奖励 | 总秒数 | 模型秒数 | prefill秒数 | decode秒数 | 模型调用 | 生成token | 有历史/12块调用 |',
    '|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    wall=f"{r['wall_seconds']:.1f}" if r['wall_seconds'] is not None else '—'
    lines.append(f"| {r['task']} | {r['arm']} | {r['valid']} | {r['rewards']} | {wall} | {r['model_seconds']:.1f} | {r['prefill_seconds']:.1f} | {r['decode_seconds']:.1f} | {r['calls']} | {r['generated_tokens']} | {r['history_calls']}/{r['full12_history_calls']} |")
lines += ['','## 同会话回放（不计任务得分）','',
    '| 任务/请求 | chunk数 | query tokens | n24/native prefill时间 | n24/n36重建时间 |',
    '|---|---:|---:|---:|---:|']
for r in replays:lines.append(f"| {r['task']}/{r['request']} | {r['selected_chunks']} | {r['query_tokens']} | {r['n24_to_native_prefill']:.3f}× | {r['n24_to_n36_build']:.3f}× |")
lines += ['','总秒数包含任务准备、模型交互、终端命令和判分。模型秒数不含排队与回放；不同自由运行轨迹的token数和命令工作量可能不同，不能把墙钟时间比直接解释为内核加速。回放固定同一请求历史和原输出token，用于归因。','',
    '没有人为输出/任务时间上限；原生位置与物理显存边界仍有限。环境或容量失败不记正常质量0。任务均分/成功率只能在有效完成数说明下报告。少于12个检索chunk或历史尚未归档时，n24未必带来加速。', '',
    '检索每次模型调用一次，生成期间不重新检索；这一轮未引入hot KV buffer或蒸馏投影头。6题仅用于探索，不是全89题Terminal-Bench成绩。']
(ROOT/'READOUT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(valid_trials=summary['valid_trials'],all_live_closed=all_closed,workers_closed=workers_closed,paired=paired,replay_count=len(replays)),indent=2))
