"""Fetch authorized experiment results and keep a local, explicit progress report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from summarize_runs import read_json
from sync_final_checkpoint import sync_final_checkpoint

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
REMOTE = '/data/liuhanzuo/encbank_v2_20260908'


def summarize_qa_cost(state, plan):
    """Read small runner receipts only; never rescore or tokenize timing inputs."""
    phases = {}
    jobs = state.get('jobs', {})
    active_id = state.get('active_job')
    active_status = {}
    if state.get('active_attempt') and active_id not in jobs:
        status_path = Path(state['active_attempt']) / 'status.json'
        if status_path.exists():
            active_status = read_json(status_path)
    for phase in ('smoke', 'full'):
        planned = [job for job in plan.get('jobs', []) if job.get('phase') == phase]
        complete_jobs = complete_calls = observed_calls = expected_calls = 0
        for job in planned:
            expected = len(job['indices']) * job['timed_repetitions']
            expected_calls += expected
            saved = jobs.get(job['id'], {})
            valid = (saved.get('status') == 'complete'
                     and saved.get('timed_generations') == expected
                     and saved.get('unique_method_examples') == len(job['indices']))
            if valid:
                complete_jobs += 1
                complete_calls += expected
                observed_calls += expected
            elif job['id'] == active_id and active_status.get('job') == active_id:
                # Warmups and extra smoke reference generations never enter this count.
                observed_calls += min(expected, max(0, active_status.get('completed_repetitions', 0)))
        phases[phase] = {'planned_jobs': len(planned), 'completed_jobs': complete_jobs,
                         'expected_timed_generations': expected_calls,
                         'completed_timed_generations': complete_calls,
                         'observed_timed_generations': observed_calls}
    full, smoke = phases['full'], phases['smoke']
    full_ready = (full['planned_jobs'] == full['completed_jobs'] == 4
                  and full['expected_timed_generations'] == full['completed_timed_generations'] == 2400)
    smoke_ready = (smoke['planned_jobs'] == smoke['completed_jobs'] == 4
                   and smoke['expected_timed_generations'] == smoke['completed_timed_generations'] == 8)
    complete = (state.get('status') == 'completed' and full_ready and smoke_ready
                and state.get('protocol') == 'native-explicit-qa-cost-v1'
                and state.get('required_gpu') == 'NVIDIA GeForce RTX 5090'
                and state.get('remote_timing_allowed') is False
                and not state.get('active_job') and not state.get('child_pid') and not state.get('error'))
    return {'status': state.get('status', 'not_started'), 'active_job': active_id,
            'active_status': active_status.get('status'), 'full': full, 'smoke': smoke,
            'all_complete': bool(complete)}


def summarize_cacheblend_serving(state, plans):
    """Trust bootstrap-validated jobs only after small completion markers agree."""
    phases, invalid_jobs = {}, []
    for phase in ('smoke', 'full'):
        plan = plans.get(phase, {})
        planned = plan.get('jobs', [])
        expected_queries = (max(plan.get('query_counts', [0]))
                            * len(plan.get('fixed_generation_lengths', []))
                            * len(plan.get('tiers', [])))
        counts = {'planned_jobs': len(planned), 'completed_jobs': 0,
                  'planned_cells': sum(job['cells'] for job in planned),
                  'completed_cells': 0, 'planned_query_records': expected_queries * len(planned),
                  'completed_query_records': 0, 'fresh_reference_sequences': 0}
        for job in planned:
            key = phase + '/' + job['id']
            saved = state.get('jobs', {}).get(key, {})
            if saved.get('status') != 'complete':
                continue
            result = Path(saved['result']) if saved.get('result') else None
            marker_path = result / 'COMPLETED.json' if result else None
            marker = read_json(marker_path) if marker_path and marker_path.exists() else {}
            hardware = marker.get('hardware', {})
            valid = (saved.get('phase') == phase and saved.get('context_tokens') == job['context_tokens']
                     and saved.get('cells') == job['cells'] and saved.get('query_records') == expected_queries
                     and saved.get('timing_eligible') is (phase == 'full')
                     and marker.get('status') == 'complete' and marker.get('cells') == job['cells']
                     and hardware.get('platform') == 'Windows' and hardware.get('device') == 'cuda:0'
                     and hardware.get('device_name') == 'NVIDIA GeForce RTX 5090'
                     and hardware.get('timing_eligible') is True)
            if phase == 'smoke':
                fresh_path = result / 'FRESH_REFERENCE_CHECK.json' if result else None
                fresh = read_json(fresh_path) if fresh_path and fresh_path.exists() else {}
                checks = fresh.get('checks', [])
                valid = (valid and saved.get('fresh_reference_sequences') == expected_queries
                         and fresh.get('complete') is True and fresh.get('timing_eligible') is False
                         and fresh.get('verified_queries') == fresh.get('extra_reference_sequences') == expected_queries
                         and len(checks) == expected_queries
                         and all(check.get('equal_ids') is True
                                 and check.get('equal_selected_positions') is True
                                 and check.get('persistent_cpu_versions_and_disk_metadata_unchanged') is True
                                 and check.get('reuse_capture_calls') == 0 for check in checks))
            else:
                valid = valid and saved.get('fresh_reference_sequences') == 0
            if not valid:
                invalid_jobs.append(key)
                continue
            counts['completed_jobs'] += 1
            counts['completed_cells'] += job['cells']
            counts['completed_query_records'] += expected_queries
            counts['fresh_reference_sequences'] += saved['fresh_reference_sequences']
        phases[phase] = counts
    smoke, full = phases['smoke'], phases['full']
    full_ready = (full['planned_jobs'] == full['completed_jobs'] == 2
                  and full['planned_cells'] == full['completed_cells'] == 24
                  and full['planned_query_records'] == full['completed_query_records'] == 800)
    smoke_ready = (smoke['planned_jobs'] == smoke['completed_jobs'] == 1
                   and smoke['planned_cells'] == smoke['completed_cells'] == 8
                   and smoke['planned_query_records'] == smoke['completed_query_records'] == 8
                   and smoke['fresh_reference_sequences'] == 8)
    complete = (state.get('status') == 'completed' and full_ready and smoke_ready and not invalid_jobs
                and state.get('protocol') == 'cacheblend-serving-reuse-v1'
                and state.get('required_gpu') == 'NVIDIA GeForce RTX 5090'
                and state.get('remote_timing_allowed') is False
                and not state.get('active_job') and not state.get('child_pid') and not state.get('error'))
    return {'status': state.get('status', 'not_started'), 'phase': state.get('phase'),
            'active_job': state.get('active_job'), 'full': full, 'smoke': smoke,
            'invalid_completed_jobs': invalid_jobs, 'all_complete': bool(complete)}


def summarize_online_kv(state, plan):
    """Count completed diagnostic generations and both actual cache boundaries.

    Read only small receipts and scalar tensor metadata; never import the model
    runner or treat instrumented generations as timing/accuracy measurements.
    """
    methods = ('fix_all', 'pub', 'pub_sink', 'j0', 'pub_lora', 'cacheblend16')
    phases = {'prefill_complete', 'decode_complete'}
    protocol = 'online-kv-inventory-v1'
    valid_plan = (plan.get('protocol') == protocol and tuple(plan.get('methods', [])) == methods
        and plan.get('document_lengths') == [32768, 131072]
        and plan.get('generation_lengths') == [16, 128]
        and set(plan.get('boundaries', [])) == phases
        and plan.get('planned_generations') == 48 and plan.get('planned_cache_boundaries') == 96
        and plan.get('timing_eligible') is False)
    completed_jobs = completed_generations = completed_boundaries = disagreements = 0
    invalid_jobs = []
    for arm in methods:
        saved = state.get('jobs', {}).get(arm, {})
        if saved.get('status') != 'complete':
            continue
        try:
            folder = Path(saved['result'])
            marker = read_json(folder / 'COMPLETED.json')
            cases = read_json(folder / 'cases.json')
            rows = [json.loads(line) for line in (folder / 'inventories.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            hardware = marker.get('hardware', {})
            admission = hardware.get('gpu_admission', {})
            valid = (saved.get('diagnostic_generations') == marker.get('diagnostic_generations') == len(rows) == len(cases) == 8
                and saved.get('observed_cache_boundaries') == marker.get('observed_cache_boundaries') == 16
                and saved.get('timing_eligible') is marker.get('timing_eligible') is False
                and marker.get('status') == 'complete' and marker.get('protocol') == protocol and marker.get('arm') == arm
                and hardware.get('platform') == 'Windows' and hardware.get('device') == 'cuda:0'
                and hardware.get('device_name') == 'NVIDIA GeForce RTX 5090' and hardware.get('timing_eligible') is True
                and hardware.get('torch_cpu_threads') == 2 and hardware.get('torch_interop_threads') == 16
                and hardware.get('omp_num_threads') == hardware.get('mkl_num_threads') == '2'
                and hardware.get('tokenizers_parallelism') == 'false'
                and admission.get('effective_idle_slack_gib') == 5.0 and admission.get('comparison') == 'strictly_less_than'
                and admission.get('source') == 'nvidia-smi MiB / 1024'
                and 0 <= admission.get('initial_used_gib', 5.0) < 5.0
                and 0 <= admission.get('recheck_used_gib', 5.0) < 5.0
                and admission.get('other_python_compute_processes') == [])
            case_ids = {case['case_id'] for case in cases}
            rows_by_id = {row['case_id']: row for row in rows}
            valid = (valid and len(case_ids) == len(rows_by_id) == 8 and set(rows_by_id) == case_ids
                and {(case['context_tokens'], case['G']) for case in cases} == {(n, g) for n in (32768, 131072) for g in (16, 128)})
            for length in (32768, 131072):
                query_sets = [{case['query_id'] for case in cases if case['context_tokens'] == length and case['G'] == generation}
                              for generation in (16, 128)]
                valid = valid and query_sets[0] == query_sets[1] and len(query_sets[0]) == 2 and 0 in query_sets[0]
            for case in cases:
                row = rows_by_id.get(case['case_id'], {})
                valid = (valid and case.get('arm') == arm and row.get('arm') == arm and row.get('protocol') == protocol
                    and row.get('hardware') == hardware and row.get('timing_eligible') is False
                    and row.get('context_tokens') == case['context_tokens'] and row.get('query_id') == case['query_id']
                    and row.get('persistent_store_metadata_unchanged') is True
                    and row.get('generated_tokens') == len(row.get('generated_ids', [])) == case['G']
                    and row.get('actual_decode_forward_calls') == case['G'] - 1
                    and row.get('fixed_generation_length') is True and row.get('document_or_query_capture_calls') == 0
                    and row.get('state_reset_verified') is True
                    and set(row.get('inventory', {})) == phases)
                for inventory in row.get('inventory', {}).values():
                    valid = (valid and inventory.get('populated_layer_entries') == row.get('num_layers') == 36
                        and inventory.get('tensor_count') == 72 and len(inventory.get('layers', [])) == 36
                        and inventory.get('logical_tensor_bytes', 0) > 0 and inventory.get('unique_backing_storage_bytes', 0) > 0
                        and all(tensor.get('device') == 'cuda:0' for layer in inventory['layers'] for tensor in layer.get('tensors', [])))
            disagreement_count = sum(row.get('historical_generated_ids_equal') is False for row in rows)
            valid = (valid and all(isinstance(row.get('historical_generated_ids_equal'), bool) for row in rows)
                and saved.get('historical_output_disagreement_cases') == marker.get('historical_output_disagreement_cases') == disagreement_count)
            if not valid:
                raise ValueError('Diagnostic receipt, case counts or phase boundaries disagree')
        except (OSError, ValueError, KeyError, TypeError):
            invalid_jobs.append(arm)
            continue
        completed_jobs += 1
        completed_generations += 8
        completed_boundaries += sum(len(row['inventory']) for row in rows)
        disagreements += disagreement_count
    active_id = state.get('active_job')
    active_observed = 0
    if active_id in methods and state.get('jobs', {}).get(active_id, {}).get('status') != 'complete' and state.get('active_attempt'):
        try:
            active = read_json(Path(state['active_attempt']) / 'status.json')
            if active.get('protocol') == protocol and active.get('arm') == active_id:
                active_observed = min(8, max(0, int(active.get('completed_diagnostic_generations', 0))))
        except (OSError, ValueError, TypeError):
            pass
    complete = (valid_plan and not invalid_jobs and completed_jobs == 6 and completed_generations == 48 and completed_boundaries == 96
        and state.get('status') == 'completed' and state.get('protocol') == protocol
        and state.get('required_gpu') == 'NVIDIA GeForce RTX 5090' and state.get('remote_timing_allowed') is False
        and state.get('timing_eligible') is False and not active_id and not state.get('child_pid') and not state.get('error'))
    return {'status': state.get('status', 'not_started'), 'protocol': protocol, 'timing_eligible': False,
        'valid_plan': bool(valid_plan), 'planned_jobs': 6, 'completed_jobs': completed_jobs,
        'planned_diagnostic_generations': 48, 'completed_diagnostic_generations': completed_generations,
        'observed_diagnostic_generations': completed_generations + active_observed,
        'planned_cache_boundaries': 96, 'completed_cache_boundaries': completed_boundaries,
        'observed_cache_boundaries': completed_boundaries + 2*active_observed,
        'active_job': active_id, 'invalid_completed_jobs': invalid_jobs,
        'historical_output_disagreement_cases': disagreements, 'all_complete': bool(complete)}


def scheduled_workloads_complete(quality_complete, serving_status, qa_complete, cb_complete,
                                 online_kv_complete, evidence_complete, diagnostic_errors):
    return bool(quality_complete and serving_status == 'completed' and qa_complete and cb_complete
                and online_kv_complete and evidence_complete and not diagnostic_errors)


def sync_once():
    command = f'{REMOTE}/venv/bin/python {REMOTE}/workspace/exp/encbank_v2_benchmarks_20260908/collect_report_payload.py'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'longjing-1', command],
                   check=True, stdout=subprocess.DEVNULL, timeout=600)
    payload = HERE / 'results/remote_report_payload.tar.gz'
    payload.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                    f'longjing-1:{REMOTE}/outputs/report_payload.tar.gz', str(payload)], check=True, timeout=180)
    destination = (HERE / 'results/remote').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(payload, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not member.isfile():
                raise ValueError(f'Unexpected archive member: {member.name}')
        archive.extractall(destination, members=members, filter='data')
    now = datetime.now(timezone.utc).isoformat()
    reports = {}
    for name, plan_name, output in [('main', 'full_plan.json', 'benchmarks_v2'),
                                   ('trained_pub', 'trained_pub_full_plan.json', 'trained_pub'),
                                   ('cacheblend16', 'cacheblend16_full_plan.json', 'cacheblend16'),
                                   ('ruler_matched16k', 'ruler_matched16k_full_plan.json', 'ruler_matched16k'),
                                   ('oracle_support', 'oracle_support_full_plan.json', 'oracle_support')]:
        report = read_json(destination / 'outputs/report_progress' / (name + '_progress.json'))
        (HERE / 'results' / (name + '_progress.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        state_path = destination / 'outputs' / ('bootstrap_' + name) / 'status.json'
        state = read_json(state_path) if state_path.exists() else {'status': 'not_started'}
        reports[name] = {'state': state, 'summary': report}
    train_path = destination / 'outputs/8b_j12_pub_4k/status.json'
    bootstrap_path = destination / 'logs/train_8b_bootstrap_status.json'
    training = read_json(train_path) if train_path.exists() else {}
    training_bootstrap = read_json(bootstrap_path) if bootstrap_path.exists() else {}
    live_training = {}
    train_log = destination / 'outputs/8b_j12_pub_4k/train.jsonl'
    if train_log.exists():
        for line in reversed(train_log.read_text(encoding='utf-8').splitlines()):
            try:
                row = json.loads(line)
                if 'step' in row:
                    live_training = row
                    break
            except ValueError:
                continue
    live_step = max(training.get('step', 0), live_training.get('step', 0))
    evidence = read_json(destination / 'outputs/diagnostics/evidence_collection_status.json')
    diagnostic_errors = list(evidence.get('errors', []))
    evidence_complete = evidence.get('status') == 'completed'
    if evidence.get('status') not in {'completed', 'pending'} and not diagnostic_errors:
        diagnostic_errors.append({'error': 'Remote evidence collection failed or has an unknown status'})
    checkpoint_transfer = sync_final_checkpoint(training)
    # Remote 3090 timings are ineligible; the local 5090 queue is authoritative.
    serving_path = HERE / 'results/local/bootstrap_serving/status.json'
    serving = read_json(serving_path) if serving_path.exists() else {'status': 'not_started', 'required_gpu': 'NVIDIA GeForce RTX 5090'}
    qa_path = HERE / 'results/local/bootstrap_qa_cost/status.json'
    qa_plan_path = HERE / 'results/protocol/qa_cost/task_plan.json'
    qa_state = read_json(qa_path) if qa_path.exists() else {}
    qa_plan = read_json(qa_plan_path) if qa_plan_path.exists() else {}
    qa_cost = summarize_qa_cost(qa_state, qa_plan)
    (HERE / 'results/qa_cost_progress.json').write_text(json.dumps(qa_cost, indent=2) + '\n', encoding='utf-8')
    cb_path = HERE / 'results/local/bootstrap_cacheblend_serving/status.json'
    cb_state = read_json(cb_path) if cb_path.exists() else {}
    cb_plans = {}
    for phase in ('smoke', 'full'):
        path = HERE / ('cacheblend_serving_' + phase + '_plan.json')
        cb_plans[phase] = read_json(path) if path.exists() else {}
    cb_serving = summarize_cacheblend_serving(cb_state, cb_plans)
    (HERE / 'results/cacheblend_serving_progress.json').write_text(json.dumps(cb_serving, indent=2) + '\n', encoding='utf-8')
    online_path = HERE / 'results/local/bootstrap_online_kv/status.json'
    online_plan_path = HERE / 'online_kv_plan.json'
    online_state = read_json(online_path) if online_path.exists() else {}
    online_plan = read_json(online_plan_path) if online_plan_path.exists() else {}
    online_kv = summarize_online_kv(online_state, online_plan)
    (HERE / 'results/online_kv_progress.json').write_text(json.dumps(online_kv, indent=2) + '\n', encoding='utf-8')
    pairing_path = destination / 'outputs/diagnostics/oracle_support_paired.json'
    pairing_status_path = destination / 'outputs/diagnostics/oracle_pairing_status.json'
    pairing = read_json(pairing_path) if pairing_path.exists() else {}
    pairing_status = read_json(pairing_status_path) if pairing_status_path.exists() else {}
    paired_complete = pairing_status.get('status') == 'completed' and pairing.get('all_planned_pairs_complete') is True
    document = [f'# Encbank experiment execution status\n\nUpdated UTC: {now}\n',
        'The target manuscript is Encbank/paper_v2_revision_20260907/main.pdf. '
        'Smoke samples are separate from full benchmark scores. A completed shard is not a completed task.\n',
        f'Training: {live_step}/4000 optimizer steps; saved checkpoint step '
        f'{training.get("step", 0)}; bootstrap phase: '
        f'{training_bootstrap.get("phase", "unknown")}.\n',
        '| Run | State | Phase | Completed jobs | Observed predictions | Complete? |',
        '|---|---|---|---:|---:|---|']
    for name, result in reports.items():
        state, report = result['state'], result['summary']
        document.append(f'| {name} | {state.get("status")} | {state.get("phase", "-")} | '
            f'{report["completed_jobs"]}/{report["planned_jobs"]} | {report["observed_predictions"]} | {report["all_complete"]} |')
    document += ['', 'Main observed predictions include 2,500 reused RULER predictions. '
        'The full main target is 39,480 including the 400-example visibility ablation. '
        'Each strong baseline adds 1,300 predictions. The independent ruler_matched16k follow-up adds '
        '300 V2/j0 predictions on explicit inputs shared with the completed strong baselines; '
        'legacy draws and results are retained. Oracle-support jobs are additional accuracy-only '
        'interventions on eligible annotated-support subsets, not full benchmark scores. '
        'The oracle queue has 3,104 new GPU predictions; 1,495 identical-pack conditions are derived '
        'from main natural results only after exact generation-cache and protocol checks. '
        'Full oracle interpretation still waits for the paired natural controls. '
        'LoCoMo answerable F1 and adversarial accuracy remain separate.',
        '', 'Details: `results/main_progress.json`, `results/trained_pub_progress.json`, '
        '`results/cacheblend16_progress.json`, `results/ruler_matched16k_progress.json`, '
        '`results/oracle_support_progress.json`; '
        'remote logs and predictions are under `results/remote/`.',
        '', f'Local RTX 5090 serving workload: {serving.get("status")}; phase {serving.get("phase", "-")}. '
        'Target: 32k/128k documents, four stock methods plus the completed trained original method, '
        'CPU/file stores, Q=1/10/100 and fixed G=16/128. Remote RTX 3090s are for accuracy and training only.',
        '', f'Local RTX 5090 matched QA cost: {qa_cost["status"]}; '
        f'full jobs {qa_cost["full"]["completed_jobs"]}/4; '
        f'full timed outputs {qa_cost["full"]["observed_timed_generations"]}/2400 '
        '(800 method-example pairs, three timed repetitions each). '
        f'Smoke is separate: {qa_cost["smoke"]["completed_jobs"]}/4 jobs, '
        f'{qa_cost["smoke"]["observed_timed_generations"]}/8 timed outputs. '
        'Unmeasured warmups and extra smoke reference generations are excluded. '
        f'Active job: {qa_cost.get("active_job") or "none"}; complete: {qa_cost["all_complete"]}. '
        'Details: `results/qa_cost_progress.json` and `results/local/bootstrap_qa_cost/status.json`.',
        '', f'Local RTX 5090 CacheBlend-style serving cost: {cb_serving["status"]}; '
        f'full jobs {cb_serving["full"]["completed_jobs"]}/2, '
        f'completed cells {cb_serving["full"]["completed_cells"]}/24, '
        f'completed query records {cb_serving["full"]["completed_query_records"]}/800. '
        f'Smoke is separate and timing-ineligible: {cb_serving["smoke"]["completed_cells"]}/8 cells, '
        f'{cb_serving["smoke"]["completed_query_records"]}/8 reuse queries plus '
        f'{cb_serving["smoke"]["fresh_reference_sequences"]}/8 fresh references. '
        f'Active job: {cb_serving.get("active_job") or "none"}; complete: {cb_serving["all_complete"]}. '
        f'Invalid completed-job markers: {cb_serving["invalid_completed_jobs"]}. '
        'Only bootstrap-validated jobs with matching completion markers count; active-job partial queries are excluded. '
        'Details: `results/cacheblend_serving_progress.json` and `results/local/bootstrap_cacheblend_serving/status.json`.',
        '', f'Local RTX 5090 direct online KV inventory: {online_kv["status"]}; '
        f'completed methods {online_kv["completed_jobs"]}/6; '
        f'completed diagnostic generations {online_kv["completed_diagnostic_generations"]}/48 '
        f'(observed including active progress: {online_kv["observed_diagnostic_generations"]}/48); '
        f'completed cache boundaries {online_kv["completed_cache_boundaries"]}/96 '
        f'(observed including active progress: {online_kv["observed_cache_boundaries"]}/96). '
        'Each generation has prefill_complete and decode_complete inventories; these are direct KV tensor/storage observations, '
        'not timed outputs, accuracy samples, persistent storage measurements or transient memory peaks. '
        f'Active job: {online_kv.get("active_job") or "none"}; complete: {online_kv["all_complete"]}; '
        f'invalid completed markers: {online_kv["invalid_completed_jobs"]}; '
        f'historical output disagreements: {online_kv["historical_output_disagreement_cases"]}. '
        'Disagreements are reported for interpretation, not silently dropped. '
        'Details: `results/online_kv_progress.json` and `results/local/bootstrap_online_kv/status.json`.',
        '', f'Final checkpoint transfer: {checkpoint_transfer.get("status")}; '
        'target: copy the completed 4000-step PEFT adapter locally while retaining the remote copy.',
        '', 'Supporting-fact diagnostics are generated beside completed LoCoMo/HotpotQA shards. '
        'HotpotQA question alignment is 200/200, but strict full supporting-sentence alignment holds for only '
        '8/200 updated LongBench contexts (131/463 individual facts). Unmatched evidence remains unknown; '
        'document touch and full sentence retrieval are separate. Only completed and checked '
        'oracle model outputs count as scores; input preparation and waiting jobs do not.',
        '', f'Oracle pairing CPU check: {pairing_status.get("status", "not_started")}; '
        f'all planned pairs complete: {paired_complete}. '
        'Details: `results/remote/outputs/diagnostics/oracle_support_paired.json`.']
    if diagnostic_errors:
        document.append('\nEvidence diagnostic errors: ' + json.dumps(diagnostic_errors, ensure_ascii=False))
    document.append(f'\nRemote CPU evidence collection: {evidence.get("status", "unknown")}; '
        f'pending missing diagnostics: {evidence.get("pending_missing_diagnostics", 0)}. '
        'Each collection attempts at most two missing diagnostics; existing results are reused. '
        'Pending diagnostics are unfinished work, not failures.')
    quality_complete = training.get('complete', False) and paired_complete and all(value['summary']['all_complete'] for value in reports.values())
    complete = scheduled_workloads_complete(quality_complete, serving.get('status'), qa_cost['all_complete'],
        cb_serving['all_complete'], online_kv['all_complete'], evidence_complete, diagnostic_errors)
    document.append(f'\nAll currently scheduled workloads complete: {bool(complete)}. This requires all quality and oracle '
        'checks, the completed serving workload, all four full QA cost jobs (2400 timed outputs), '
        'the separate eight QA smoke timed outputs, the CacheBlend-style full cost workload '
        '(24 cells / 800 query records) and its timing-ineligible correctness smoke '
        '(8 reuse queries plus 8 fresh references), the six-method online KV diagnostic '
        '(48 generations / 96 prefill-and-decode cache boundaries), and completed evidence collection without errors or pending diagnostics. '
        'This is not completion of the paper-wide experiment checklist: unscheduled work such as '
        'additional architectures remains outside this flag. '
        'Quality aggregation and missing evidence diagnostics run on the remote CPU.')
    (HERE / 'RUN_STATUS.md').write_text('\n'.join(document) + '\n', encoding='utf-8')
    print(json.dumps({'updated_at': now, 'training_step': live_step,
                     'completed_jobs': {name: value['summary']['completed_jobs'] for name, value in reports.items()},
                     'all_quality_complete': quality_complete, 'serving_status': serving.get('status'),
                     'qa_cost_status': qa_cost['status'], 'qa_cost_full_jobs': qa_cost['full']['completed_jobs'],
                     'qa_cost_timed_outputs': qa_cost['full']['observed_timed_generations'],
                     'qa_cost_complete': qa_cost['all_complete'],
                     'cacheblend_serving_status': cb_serving['status'],
                     'cacheblend_serving_full_cells': cb_serving['full']['completed_cells'],
                     'cacheblend_serving_query_records': cb_serving['full']['completed_query_records'],
                     'cacheblend_serving_complete': cb_serving['all_complete'],
                     'online_kv_status': online_kv['status'],
                     'online_kv_completed_generations': online_kv['completed_diagnostic_generations'],
                     'online_kv_observed_generations': online_kv['observed_diagnostic_generations'],
                     'online_kv_completed_boundaries': online_kv['completed_cache_boundaries'],
                     'online_kv_complete': online_kv['all_complete'],
                     'evidence_status': evidence.get('status'),
                     'pending_evidence_diagnostics': evidence.get('pending_missing_diagnostics', 0),
                     'all_workloads_complete': complete}), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow', action='store_true', help='Follow these existing one-shot processes until complete')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    while True:
        try:
            complete = sync_once()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(json.dumps({'sync_error': str(error)}), flush=True)
            if not args.follow:
                raise
            complete = False
        if complete or not args.follow:
            return
        time.sleep(max(60, args.interval))


if __name__ == '__main__':
    main()
