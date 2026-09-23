"""Build local monitoring summaries from completed read-only audits; never touches remote jobs."""
import datetime
import json
import sys
from pathlib import Path

M = Path(__file__).resolve().parent
stamp = sys.argv[1]
assert len(stamp) == 13 and stamp[:8].isdigit() and stamp[8] == '_' and stamp[9:].isdigit()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

before = read(M / f'before_heartbeat_{stamp}.json')
now = read(M / 'latest.json')
assert now['epoch'] > before['epoch'], 'Observer has not refreshed latest.json'
hosts = {path.stem.rsplit('_', 1)[1]: read(path) for path in M.glob(f'heartbeat_{stamp}_lj-gpu*.json')}
assert hosts, 'Actual-node audits are required'
report = dict(stamp=stamp, baseline_epoch=before['epoch'], observer_epoch=now['epoch'],
    queue=now['queue'], accounting=now['accounting'], scale4_registry_sha256=now['scale4_registry_sha256'],
    arms={}, runs={}, host_epochs={node: h['epoch'] for node, h in hosts.items()},
    issues=[], new_verified_tasks=[], task_state_changes=[], new_faults=[],
    scientific_source_changes=False, submissions=0, cancellations=0,
    known_unresolved=[
        'k12A117352: 8 infrastructure-interrupted tasks and 15 never-started tasks; not scored or replayed.',
        'k12B120001: pre-model dependency import failure; 32 tasks zero model calls, no resubmission.',
        'k48 extract-moves-from-video: prior UnicodeDecodeError, not scored or replayed.'],
    snapshot_boundary='Sequential observer and host reads have distinct timestamps while generation continues.')
for arm, current in now['arms'].items():
    previous = before['arms'][arm]
    entry = {k: current[k] for k in ['new_verified', 'new_passed', 'new_failed', 'new_context_failures',
        'cumulative_verified', 'cumulative_passed', 'cumulative_failed', 'counts', 'requests', 'responses',
        'generated_tokens', 'integrity_errors', 'ownership_errors']}
    entry['delta'] = {k: current[k] - previous[k] for k in ['new_verified', 'requests', 'responses', 'generated_tokens']}
    entry['H_check_failures'] = sum(t['H_check_failures'] for t in current['tasks'].values())
    entry['tasks'] = {}
    for task, row in current['tasks'].items():
        old = previous['tasks'][task]
        entry['tasks'][task] = {k: row.get(k) for k in ['state', 'owner_run', 'started_epoch', 'ended_epoch',
            'ended_epoch_upper_bound', 'elapsed_seconds', 'elapsed_seconds_upper_bound', 'phase_seconds',
            'trial_seconds', 'requests', 'responses', 'generated_tokens', 'model_seconds', 'exception_type',
            'reply_proofs_ok', 'sessions_released', 'containers_closed', 'reported_reward']}
        entry['tasks'][task]['completed_reply_tokens_per_model_second'] = (
            row['generated_tokens'] / row['model_seconds'] if row['model_seconds'] else None)
        if row['state'] != old['state']:
            report['task_state_changes'].append([arm, task, old['state'], row['state']])
        if row['state'].startswith('verified') and not old['state'].startswith('verified'):
            report['new_verified_tasks'].append(dict(arm=arm, **row))
        if row['state'] in ['incomplete', 'infrastructure_interrupted', 'closed_needs_audit'] and row['state'] != old['state']:
            report['new_faults'].append(dict(arm=arm, task=task, state=row['state']))
    for key in ['worker_failure', 'owner_failure', 'memory_cap_failure']:
        if current.get(key) != previous.get(key):
            report['new_faults'].append(dict(arm=arm, field=key, value=current.get(key)))
    if entry['integrity_errors'] or entry['ownership_errors'] or entry['H_check_failures']:
        report['issues'].append(arm + ': reply/ownership/H validation')
    report['arms'][arm] = entry
for node, host in hosts.items():
    report['issues'].extend(node + ': ' + error for error in host['errors'])
    for run_id, run in host['runs'].items():
        report['runs'][run_id] = dict(host=node, **run)
        for container in run['containers']:
            if 'oom_kill 0' not in (container.get('memory.events') or ''):
                report['issues'].append(run_id + ': container OOM/missing memory evidence: ' + container['task'])
    report['dispatcher'] = host['dispatcher']
    report['dispatcher_accounting'] = host['dispatcher_accounting']
expected_running = {line.split('|')[0] for line in now['queue'].splitlines() if '|RUNNING|' in line}
observed_running = {run['job_id'] for run in report['runs'].values()}
if expected_running != observed_running:
    report['issues'].append('Slurm/node snapshot running-job mismatch; inspect timestamps and any stage transition')
report['new_total_verified'] = sum(a['new_verified'] for a in report['arms'].values())
report['running_gpu_jobs'] = len(expected_running)
report['active_tasks'] = sum(len(r['active']) for r in now['runs'].values())
report['audited_active_tasks'] = sum(len(r['active']) for r in report['runs'].values())
report['active_containers'] = sum(len(r['containers']) for r in report['runs'].values())
report['decision_candidate'] = 'NOTIFY' if report['new_verified_tasks'] or report['new_faults'] or report['issues'] else 'DONT_NOTIFY'
audit_path = M / f'heartbeat_{stamp}_new_tasks_audit.json'
if audit_path.exists():
    audit = read(audit_path)
    report['new_task_audit_file'] = audit_path.name
    report['parser_notes'] = {name: dict(agent_steps=r.get('trajectory_agent_steps'),
        warnings=r.get('parser_warnings'), errors=r.get('parser_errors')) for name, r in audit['tasks'].items()}
tz = datetime.timezone(datetime.timedelta(hours=8))
date = datetime.datetime.fromtimestamp(now['epoch'], tz).isoformat()
lines = [f'COMem 心跳 {stamp}；逐题观察结束 {date}', '',
    '方法 | 累计已核/89 | 累计通过/失败 | 新计划已核 | 其中上下文失败',
    '--- | ---: | ---: | ---: | ---:']
for arm, row in report['arms'].items():
    lines.append(f"{arm} | {row['cumulative_verified']}/89 | {row['cumulative_passed']}/{row['cumulative_failed']} | {row['new_verified']} | {row['new_context_failures']}")
lines += ['', f"新计划合计{report['new_total_verified']}/162已核。Slurm正式运行GPU {report['running_gpu_jobs']}张，控制器报告活跃题{report['active_tasks']}；已在实际节点核验{report['audited_active_tasks']}活跃题、{report['active_containers']}容器。"]
for row in report['new_verified_tasks']:
    seconds = row.get('model_seconds') or 0
    tps = row['generated_tokens'] / seconds if seconds else 0
    lines += ['', f"新增完成 {row['arm']} / {row['task']}：{row['reported_reward']}分，{row['state']}。",
        f"请求/完整回复 {row['requests']}/{row['responses']}，生成{row['generated_tokens']} token，累计模型请求{seconds:.2f}秒（{tps:.2f} token/s）。",
        f"控制器启动至父wait {row.get('elapsed_seconds', 0)/60:.2f}分钟；trial {row.get('trial_seconds', 0)/60:.2f}分钟。阶段耗时：{json.dumps(row.get('phase_seconds'), ensure_ascii=False)}",
        f"实际父wait：{(row.get('parent_receipt') or {}).get('actual_parent_wait')}；回复证明：{row.get('reply_proofs_ok')}；H异常：{row['H_check_failures']}；会话释放：{row.get('sessions_released')}；容器关闭：{row.get('containers_closed')}。"]
lines += ['', 'GPU瞬时快照：']
for rid, run in report['runs'].items():
    gpu = run['gpu']
    lines.append(f"- {rid} / {run['job_id']} / {run['host']} / {run['gpu_uuid']}: {gpu['memory_mib']}/{gpu['total_mib']} MiB，{gpu['utilization_percent']}%，{len(run['active'])}活跃题。")
lines += ['', '各组新增完整回复与生成token：']
for arm, row in report['arms'].items():
    lines.append(f"- {arm}: +{row['delta']['responses']} 回复，+{row['delta']['generated_tokens']} token；SHA错误{len(row['integrity_errors'])}，归属错误{len(row['ownership_errors'])}，H异常{row['H_check_failures']}。")
lines += ['', 'CPU调度器实际状态：' + report['dispatcher_accounting'].strip(),
    '已完成调度器的dispatcher_status.pending仅为历史快照，以dispatcher_complete和各submission为准。',
    '', 'k12A OOM中断与未启动题、k12B模型前启动失败、k48既有编码异常继续单列待处理；不计正常0分或自动重跑。',
    '未修改正式科学源码、提交或取消任务。保留UNLIMITED与原生容量政策。各快照并非同一瞬间；时间、速度和阶段耗时见JSON。',
    f"此次候选通知：{report['decision_candidate']}；新故障：{len(report['new_faults'])}；核验待处理项：{json.dumps(report['issues'], ensure_ascii=False)}。全部162次尚未闭合，继续每小时巡检。"]
(M / f'heartbeat_{stamp}_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
(M / f'heartbeat_{stamp}_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({key: report[key] for key in ['decision_candidate', 'new_total_verified', 'active_tasks',
    'active_containers', 'issues', 'new_faults']}, ensure_ascii=False))
print(json.dumps([dict(arm=r['arm'], task=r['task'], reward=r['reported_reward']) for r in report['new_verified_tasks']], ensure_ascii=False))
