"""Read official scalar rewards and request measurements, never verifier source."""
import json,statistics
from pathlib import Path
from io_utils import save
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text())
def pct(values,p):
    if not values:return None
    v=sorted(values);return v[min(len(v)-1,int((len(v)-1)*p))]
def fmt(v):return '-' if v is None else f'{v:.2f}'
def main():
    summary=[];trials=[]
    for arm in P['arms']:
        rows=[]
        for path in (ROOT/'results'/('qcm8b_'+arm)).glob('*/result.json'):
            r=json.loads(path.read_text());reward=(r.get('verifier_result') or {}).get('rewards')
            row=dict(arm=arm,trial=path.parent.name,task=r.get('task_name'),rewards=reward,exception=r.get('exception_info'),agent_execution=r.get('agent_execution'),started_at=r.get('started_at'),finished_at=r.get('finished_at'))
            rows.append(row);trials.append(row)
        requests=[];failed=[]
        for path in (ROOT/'rpc').glob('*.response.json'):
            r=json.loads(path.read_text())
            if r.get('arm')!=arm:continue
            if r['status']=='ok':requests.append(r)
            else:failed.append(r)
        scored=[r for r in rows if r['rewards'] is not None]
        wins=sum(float(r['rewards'].get('reward',0)) for r in scored)
        s=dict(arm=arm,expected_tasks=len(P['tasks']),finished_trials=len(rows),scored_trials=len(scored),reward_sum=wins,
            success_percent_all_tasks=100*wins/len(P['tasks']) if len(rows)==len(P['tasks']) else None,
            unscored_trials=len(rows)-len(scored),requests=len(requests),failed_requests=len(failed),
            max_logical_history_tokens=max((r['logical_history_tokens'] for r in requests),default=0),
            requests_over_32k=sum(r['logical_history_tokens']>=32768 for r in requests),
            retrieval_active_requests=sum(r.get('selected_chunks',0)>0 for r in requests),
            history_pruned_requests=sum(r.get('omitted_history_chunks',0)>0 for r in requests),
            mean_selected_chunks=statistics.mean([r.get('selected_chunks',0) for r in requests]) if requests else None,
            ttft_p50_seconds=pct([r['ttft_seconds'] for r in requests],.5),ttft_p95_seconds=pct([r['ttft_seconds'] for r in requests],.95),
            model_seconds=sum(r['request_seconds'] for r in requests),write_seconds=sum(r.get('write_seconds',0) for r in requests),
            peak_allocated_gib=max((r['peak_allocated_bytes']/2**30 for r in requests),default=None),
            failed_request_details=failed)
        summary.append(s)
    save(ROOT/'summary.json',dict(scope=P['scope'],arms=summary,trials=trials))
    lines=['# Encbank Terminal-Bench 2.1 首轮闭环试验','',P['scope'],'',
        'Qwen3-8B；Encbank j=12，原 adapter；任务环境与 verifier 未修改；每组 6 个预先选定任务、一次尝试。此表不是完整 89 任务榜单。',
        '', '| 方法 | 已结束/6 | 有评分 | 成功率（全6项） | 最大历史 tokens | 检索请求/总请求 | p50 TTFT(s) | 峰值 allocated GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['arm']} | {s['finished_trials']}/6 | {s['scored_trials']} | {fmt(s['success_percent_all_tasks'])} | {s['max_logical_history_tokens']} | {s['retrieval_active_requests']}/{s['requests']} | {fmt(s['ttft_p50_seconds'])} | {fmt(s['peak_allocated_gib'])} |")
    lines+=['','尚未完成的组不报成功率；无判分的任务须与模型/超时/环境故障分类后解释，不能当成功或从分母静默删除。',
        'TTFT 包含本次新增历史 Write、query Write、选块和 Read，不包含本地—GPU 传输；端到端任务时间见原始 Harbor 时间戳，往返耗时见 RPC timing。',
        'Dense 为无 adapter 原模型并保留精确前缀 KV；四个 Encbank 分支使用同一个 adapter。该对照不能单独归因于缓存表示。',
        'Attention top-p 为候选48块上的条件注意力质量启发式，不能套用 Twilight 的原生误差保证。历史工具输出为当时观察，不代表文件当前状态。',
        '只有实际历史长度、检索启动与历史裁剪统计支持时，才能讨论长上下文收益。峰值 allocated 为 PyTorch 进程分配口径。','']
    (ROOT/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps([dict(arm=s['arm'],finished=s['finished_trials'],requests=s['requests']) for s in summary]))
if __name__=='__main__':main()
