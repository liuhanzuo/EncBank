"""Final cross-job accounting and reporting checks; no model calls."""
import collections, datetime, json, statistics
from pathlib import Path

root=Path(__file__).resolve().parent
read=lambda p: json.loads(p.read_text(encoding='utf8'))
p=read(root/'plan.json');summary=read(root/'summary.json')
checks=read(root/'completed_arm_validation.json')
assert {v['arm'] for v in checks}==set(p['arms']) and all(v['status']=='PASS' for v in checks)
assert read(root/'owner_status.json')['phase']=='COMPLETE'
monitor=read(root/'monitor.json')
assert '109631|COMPLETED|0:0' in monitor['sacct'] and not monitor['queue']
new=read(root/'remote_records'/'environment.json')
parent=read(root/'remote_records'/'parent_exit.json')
worker=read(root/'remote_records'/'worker_complete.json')
old=read(root/'infrastructure_attempts'/'20260920_bm25_empty_history'/'remote_job_evidence.json')['files']
assert parent['actual_wait'] and parent['returncode']==0 and worker['actual_shutdown']
assert old['parent_exit.json']['actual_wait'] and old['parent_exit.json']['returncode']==1
for name in ('torch','transformers','adapter_sha256'):
    assert old['environment.json'][name]==new[name]
assert new['adapter_sha256']==p['adapter_sha256']
probe=read(root/'remote_records'/'probe_correctness.json')['reference_check']
assert probe['max_abs']<.01 and probe['rms']<.005
all_q=[read(f) for f in (root/p['rpc_subdir']).glob('*.request.json')]
all_q=[q for q in all_q if q['request_id'] not in p.get('excluded_request_ids',[])]
assert len(all_q)==sum(v['request_count'] for v in checks)==68
new_ids={q['request_id'] for q in all_q if q['arm'] in ('bm25_p90','iter48_qk_p90')}
events=[json.loads(line) for line in (root/'remote_records'/'events_109631.jsonl').read_text().splitlines()]
starts=[e['request_id'] for e in events if e['event']=='request_start']
assert len(starts)==len(set(starts))==len(new_ids)==worker['requests']==34
assert set(starts)==new_ids
assert sum(e['event']=='session_release' for e in events)==worker['released_sessions']==12
assert sum(e['event']=='batch_complete' for e in events)==worker['batches']==17
for s in summary['arms']:
    check=next(v for v in checks if v['arm']==s['arm'])
    a=s['all_response_statistics']
    assert s['finished_trials']==6 and a['total_requests']==check['request_count']
    assert a['status_counts']==check['response_statuses']
    rows=[read(root/p['rpc_subdir']/(q['request_id']+'.response.json')) for q in all_q if q['arm']==s['arm']]
    measured=[r for r in rows if 'logical_history_tokens' in r]
    ok=[r for r in rows if r['status']=='ok']
    assert a['measured_history_requests']==len(measured)
    assert a['max_measured_history_tokens']==max(r['logical_history_tokens'] for r in measured)
    assert a['peak_allocated_gib']==max(r['peak_allocated_bytes']/2**30 for r in rows if 'peak_allocated_bytes' in r)
    assert s['queue_p50_seconds']==statistics.median(r['queue_seconds'] for r in ok)
    assert a['all_recorded_states_unchanged']
assert sum(s['scored_trials'] for s in summary['arms'])==19
assert sum(s['reward_sum'] for s in summary['arms'])==0
assert sum(s['unscored_trials'] for s in summary['arms'])==5
result=dict(status='PASS',checked_at=datetime.datetime.now().astimezone().isoformat(),
    completed_job='109631',slurm='COMPLETED 0:0',actual_worker_wait=0,
    earlier_job='109054',earlier_job_exit=1,earlier_failure_preserved=True,
    tasks_finished=24,officially_scored=19,official_successes=0,unscored=5,
    valid_requests=68,new_job_requests=34,new_job_session_releases=12,new_job_batches=17,
    same_adapter_and_dependency_versions=True,probe=probe,
    reporting_scopes_checked=True,full_four_way_quality_comparison=False,
    max_measured_history_tokens=max(s['all_response_statistics']['max_measured_history_tokens'] for s in summary['arms']),
    gpu_experiments_rerun_for_this_validation=False,
    limitations=['Five k12 trials unscored after old deadline-bridge error.',
                 'Later 18 tasks all reached original task timeouts; cohort scheduling confounds selector quality.',
                 'No measured 32k history or 48-candidate saturation.',
                 'TTFT is internal, not streamed client-visible latency; failed dequeue delay may outlast client cancellation.'])
(root/'delivery_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
