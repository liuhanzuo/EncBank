"""Read official scalar rewards and request measurements, never verifier source."""
import json,statistics
from collections import Counter
from pathlib import Path
from io_utils import save
ROOT=Path(__file__).resolve().parent;P=json.loads((ROOT/'plan.json').read_text())
def pct(values,p):
    if not values:return None
    v=sorted(values);position=(len(v)-1)*p;lower=int(position);upper=min(lower+1,len(v)-1)
    return v[lower]+(v[upper]-v[lower])*(position-lower)
def fmt(v):return '-' if v is None else f'{v:.2f}'
def main():
    summary=[];trials=[]
    for arm in P['arms']:
        rows=[]
        config=json.loads((ROOT/'harbor_configs'/(arm+'.json')).read_text())
        for path in (ROOT/P.get('results_subdir','results')/config['job_name']).glob('*/result.json'):
            r=json.loads(path.read_text());reward=(r.get('verifier_result') or {}).get('rewards')
            row=dict(arm=arm,trial=path.parent.name,task=r.get('task_name'),rewards=reward,exception=r.get('exception_info'),agent_execution=r.get('agent_execution'),started_at=r.get('started_at'),finished_at=r.get('finished_at'))
            rows.append(row);trials.append(row)
        requests=[];failed=[];roundtrips=[]
        for path in (ROOT/P.get('rpc_subdir','rpc')).glob('*.response.json'):
            r=json.loads(path.read_text())
            if r.get('arm')!=arm or r.get('request_id') in P.get('excluded_request_ids',[]):continue
            if r['status']=='ok':
                requests.append(r)
                timing=path.with_name(path.name.replace('.response.json','.timing.json'))
                if timing.exists():roundtrips.append(json.loads(timing.read_text())['client_roundtrip_seconds'])
            else:failed.append(r)
        scored=[r for r in rows if r['rewards'] is not None]
        wins=sum(float(r['rewards'].get('reward',0)) for r in scored)
        all_records=requests+failed
        measured=[r for r in all_records if 'logical_history_tokens' in r]
        queue_all=[r['queue_seconds'] for r in all_records if 'queue_seconds' in r]
        cohorts={r['batch_number']:r['batch_initial_size'] for r in all_records if 'batch_number' in r}
        s=dict(arm=arm,expected_tasks=len(P['tasks']),finished_trials=len(rows),scored_trials=len(scored),reward_sum=wins,
            success_percent_all_tasks=100*wins/len(P['tasks']) if len(rows)==len(scored)==len(P['tasks']) else None,
            unscored_trials=len(rows)-len(scored),requests=len(requests),failed_requests=len(failed),
            max_logical_history_tokens=max((r['logical_history_tokens'] for r in requests),default=0),
            requests_over_32k=sum(r['logical_history_tokens']>=32768 for r in requests),
            retrieval_active_requests=sum(r.get('selected_chunks',0)>0 for r in requests),
            history_pruned_requests=sum(r.get('omitted_history_chunks',0)>0 for r in requests),
            mean_selected_chunks=statistics.mean([r.get('selected_chunks',0) for r in requests]) if requests else None,
            ttft_p50_seconds=pct([r['full_worker_ttft_seconds'] for r in requests],.5),ttft_p95_seconds=pct([r['full_worker_ttft_seconds'] for r in requests],.95),
            queue_p50_seconds=pct([r['queue_seconds'] for r in requests],.5),queue_p95_seconds=pct([r['queue_seconds'] for r in requests],.95),
            service_ttft_p50_seconds=pct([r['queue_seconds']+r['full_worker_ttft_seconds'] for r in requests],.5),
            service_ttft_p95_seconds=pct([r['queue_seconds']+r['full_worker_ttft_seconds'] for r in requests],.95),
            client_roundtrip_p50_seconds=pct(roundtrips,.5),client_roundtrip_p95_seconds=pct(roundtrips,.95),
            sum_request_seconds=sum(r['request_seconds'] for r in requests),write_seconds=sum(r.get('write_seconds',0) for r in requests),
            initial_batch_size_histogram=dict(Counter(cohorts.values())),generation_cap_requests=sum(bool(r.get('hit_generation_cap')) for r in requests),
            attention_probe_seconds=sum(r.get('attention_probe_seconds',0) for r in requests),
            selection_zero_mass_requests=sum(bool(r.get('selection_detail',{}).get('zero_mass')) for r in requests),
            selection_cap_limited_requests=sum(bool(r.get('selection_detail',{}).get('cap_limited')) for r in requests),
            peak_allocated_gib=max((r['peak_allocated_bytes']/2**30 for r in requests),default=None),
            failed_request_details=failed)
        s['all_response_statistics']=dict(
            total_requests=len(all_records),status_counts=dict(Counter(r['status'] for r in all_records)),
            measured_history_requests=len(measured),max_measured_history_tokens=max((r['logical_history_tokens'] for r in measured),default=0),
            measured_history_over_32k=sum(r['logical_history_tokens']>=32768 for r in measured),
            max_selected_chunks=max((r.get('selected_chunks',0) for r in measured),default=0),
            retrieval_active_requests=sum(r.get('selected_chunks',0)>0 for r in measured),
            queue_records=len(queue_all),queue_record_p50_seconds=pct(queue_all,.5),queue_record_p95_seconds=pct(queue_all,.95),
            queue_record_max_seconds=max(queue_all,default=None),
            peak_allocated_gib=max((r['peak_allocated_bytes']/2**30 for r in all_records if 'peak_allocated_bytes' in r),default=None),
            attention_probe_requests=sum(r.get('attention_probe_seconds',0)>0 for r in measured),
            attention_probe_seconds=sum(r.get('attention_probe_seconds',0) for r in measured),
            max_attention_candidates=max((len(r.get('selection_detail',{}).get('candidates',[])) for r in measured),default=0),
            selection_zero_mass_requests=sum(bool(r.get('selection_detail',{}).get('zero_mass')) for r in measured),
            selection_cap_limited_requests=sum(bool(r.get('selection_detail',{}).get('cap_limited')) for r in measured),
            generation_cap_requests=sum(bool(r.get('hit_generation_cap')) for r in all_records),
            immutable_checks_recorded=sum('state_hashes_unchanged' in r for r in all_records),
            all_recorded_states_unchanged=all(r.get('state_hashes_unchanged',True) for r in all_records),
            task_exception_counts=dict(Counter((r['exception'] or {}).get('exception_type','none') for r in rows)))
        summary.append(s)
    recovery=ROOT/'infrastructure_recovery.json'
    deadline_recovery=ROOT/'deadline_bridge_recovery.json'
    selector_recovery=ROOT/'selector_empty_history_recovery.json'
    save(ROOT/'summary.json',dict(scope=P['scope'],arms=summary,trials=trials,
        metric_scopes=dict(percentiles='linear interpolation (type 7); final reporting replaces earlier lower-order-statistic estimates',
            legacy_arm_metrics='status=ok responses only, except initial_batch_size_histogram and failed_requests/details',
            all_response_statistics='All valid responses including deadline/cancelled; each measurement reports its available sample scope',
            failed_queue_records='Server dequeue delay; may extend past client cancellation and is not client-observed latency',
            first_token='Internal GPU service measurement; RPC returns a complete response and does not stream tokens'),
        infrastructure_recovery=json.loads(recovery.read_text()) if recovery.exists() else None,
        deadline_bridge_recovery=json.loads(deadline_recovery.read_text()) if deadline_recovery.exists() else None,
        selector_empty_history_recovery=json.loads(selector_recovery.read_text()) if selector_recovery.exists() else None))
    lines=['# Encbank Terminal-Bench 2.1 首轮闭环试验','',P['scope'],'',
        'Qwen3.8-27B；Encbank j=21，原 adapter，xhigh/T1/top-p.95/top-k20/32768maxnew；任务环境与 verifier 未修改；每组 6 个预先选定任务、一次尝试。此表不是完整 89 任务榜单。',
        '', '| 方法 | 已结束/6 | 有评分 | 成功率（全6项，%） | 正常回复最大历史 tokens | 检索/正常回复 | 正常回复p50排队(s) | 正常回复p50服务TTFT(s) | 全记录峰值 allocated GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    if all(s['finished_trials']==len(P['tasks']) for s in summary):
        lines[4:4]=[f"四组共{sum(s['finished_trials'] for s in summary)}次任务运行已结束，{sum(s['scored_trials'] for s in summary)}项获得官方评分，成功{sum(s['reward_sum'] for s in summary):g}项，另有{sum(s['unscored_trials'] for s in summary)}项未评分。未评分条目不补零。本批主要暴露当前服务的调度限制，不能作为完整四组检索质量比较。",'']
    for s in summary:
        lines.append(f"| {s['arm']} | {s['finished_trials']}/6 | {s['scored_trials']} | {fmt(s['success_percent_all_tasks'])} | {s['max_logical_history_tokens']} | {s['retrieval_active_requests']}/{s['requests']} | {fmt(s['queue_p50_seconds'])} | {fmt(s['service_ttft_p50_seconds'])} | {fmt(s['all_response_statistics']['peak_allocated_gib'])} |")
    lines+=['','尚未完成或含无判分任务的组不报全组成功率；无判分的任务须与模型/超时/环境故障分类后解释，不能当成功、科学零分或从分母静默删除。',
        'iter_k12 / iter_k48沿用迭代BM25选块，上限分别为12 / 48；bm25_p90以累计覆盖90%归一化BM25分数为目标；iter48_qk_p90先取最多48个BM25候选，再以累计注意力权重0.90为目标选块。两种top-p均保留原min=4、max=48与零分回退规则。这里的选块p=0.90，与生成采样p=0.95不同。',
        '表中服务TTFT包含GPU侧队列等待、本次新增历史Write、query Write、选块和Read，不含本地—GPU传输。summary同时保留不含队列的worker TTFT、队列p50/p95、含队列服务TTFT及成功回复的完整客户端往返p50/p95；仅看worker TTFT会掩盖队首阻塞。端到端任务时间见原始Harbor时间戳，往返耗时见RPC timing。',
        '当前服务按整批decode结束后才接收下一批请求；长生成会阻塞已完成请求的下一轮工具调用。网络正常时触及官方agent时限属于当前部署的有效超时结果，不能归为基础设施无效或据此单独判断检索质量；不放宽时限或重跑低分。',
        '本次四个 Encbank 分支使用同一个 adapter。原27B Dense只作另列参考，运行设置/服务和计时边界不能混合；未重跑已完成Dense题。',
        'Attention top-p 为候选48块上的条件注意力质量启发式，不能套用 Twilight 的原生误差保证。历史工具输出为当时观察，不代表文件当前状态。',
        '上述延迟和检索比例只统计status=ok的正常回复；这不等于任务成功。峰值显存纳入有测量的超时/取消回复，是共享cohort的PyTorch allocated峰值，不是单个会话独占量，也不是硬件总容量。未进入模型的请求没有历史长度/显存测量，不能补零。',
        'TTFT为GPU内部首token时刻；当前RPC整条回复返回，不提供流式首token。因此这里的TTFT不是客户端可见首token延迟。最终分位数采用线性插值（type 7），取代早期报告的下取整顺序统计量，原始测量不变。',
        'summary.json 同时记录去重后的实际初始batch尺寸分布、生成长度封顶、选块零分/额度不足、QK探针开销。sum_request_seconds会在合批期间重叠，不能当系统墙钟时间或用于直接计算总吞吐。','']
    lines+=['所有有效RPC的完成状态与服务端等待记录如下。deadline/cancelled的出队记录可能晚于客户端已超时或取消的时刻，因此这里只用于解释后台队列，不当作用户实际等待分位数。',
        '', '| 方法 | 正常/全部RPC | deadline / cancelled | 历史可测/全部RPC | 可测最大历史 tokens | 最多选中块 | 出队等待p95 / 最大(s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        a=s['all_response_statistics'];c=a['status_counts']
        lines.append(f"| {s['arm']} | {c.get('ok',0)}/{a['total_requests']} | {c.get('deadline',0)} / {c.get('cancelled',0)} | {a['measured_history_requests']}/{a['total_requests']} | {a['max_measured_history_tokens']} | {a['max_selected_chunks']} | {fmt(a['queue_record_p95_seconds'])} / {fmt(a['queue_record_max_seconds'])} |")
    qk=next(s['all_response_statistics'] for s in summary if s['arm']=='iter48_qk_p90')
    lines+=['',f"Attention top-p实际执行{qk['attention_probe_requests']}次探针，总探针耗时{qk['attention_probe_seconds']:.3f}s（含有测量的超时回复），候选最多{qk['max_attention_candidates']}块。该耗时已包含在选块和服务计时中，不额外重复相加。",
        '本批任务历史未形成32k以上的实测样本，不能据此宣称长上下文收益、k=48饱和时的行为或生产并发能力。少量任务且大量超时，也不支持稳定的尾延迟优劣排序。','']
    if recovery.exists():
        lines+=['首组的两次基础设施无效尝试保留在 infrastructure_attempts/20260919_first_arm 和 infrastructure_attempts/20260920_second_arm_transport。第二次的SSH连接错误使共享控制器停止转发全部任务，不能据此判断模型质量；两次均整组排除，不按任务成绩挑选。恢复沿用同一GPU模型、采样、任务内容和原始超时。RPC改为单路批量SSH交换，断线后按原请求ID取回回复，不重复生成；任务文件使用Linux持久目录。','']
    if deadline_recovery.exists():
        lines+=['首个k12组部分任务在部署超时后，被旧接口转换为通用异常，导致官方verifier未执行；这些任务保留未评分，不能补零或将已有子集当作全组准确率。后续组已修复超时异常类型的传递，使官方verifier正常处理任务终态；修复不新增生成、不延长时限、不改变模型或检索。首组日志不重跑，具体修复记录见deadline_bridge_recovery.json。全组官方评分不齐时，这批结果主要用于调度诊断，不能作为完整四组检索质量比较。','']
    if selector_recovery.exists():
        lines+=['BM25 top-p最初在无历史缓存的首轮触发空输入异常，GPU作业109054随即退出，尚无该组模型回复。此轮未启动模型的等待超时属于服务故障，原始记录单独保留，不计模型得分。修复仅为空候选池返回空选择，非空候选的评分和top-p规则未改；k12/k48结果保留，仅另启单卡恢复余下两组。跨作业保存各自环境记录，不能声称四组使用同一次模型加载。','']
    (ROOT/'RESULTS_zh.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps([dict(arm=s['arm'],finished=s['finished_trials'],requests=s['requests']) for s in summary]))
if __name__=='__main__':main()
