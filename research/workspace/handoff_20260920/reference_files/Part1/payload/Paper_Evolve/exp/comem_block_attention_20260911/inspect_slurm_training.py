"""Read-only, stdlib SSH snapshot of this task's single-L20D Slurm training.

Run locally: python inspect_slurm_training.py [--job 24023|latest].
Only task metadata and bounded text tails are read. No GPU/model imports, remote
file writes, process probes, submissions, cancellations, or retries are made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def collect_remote(requested_job, tail_lines):
    """Self-contained function sent over SSH stdin; never consults login /proc."""
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    import re
    import socket
    import subprocess
    import time

    root = Path('/srv/encbank/comem_sparse_slurm_20260912')
    output = root / 'outputs/sparse_comem_20260911'
    now = time.time()

    def confined(path):
        resolved = path.resolve()
        if root.resolve() not in resolved.parents:
            raise ValueError('Refusing a read outside the fixed task root: '+str(path))
        return resolved

    def read_file(path, *, tail=False, limit=262144):
        row = {'path': str(path)}
        try:
            path = confined(path)
            if not path.is_file():
                return {**row, 'exists': False}
            with path.open('rb') as stream:
                import os
                stat = os.fstat(stream.fileno())
                row.update(exists=True, bytes=stat.st_size, mtime_unix=stat.st_mtime,
                           age_seconds=max(0., now-stat.st_mtime))
                if tail:
                    stream.seek(max(0, stat.st_size-limit))
                    raw = stream.read(limit)
                    lines = raw.decode('utf-8', errors='replace').splitlines()
                    if stat.st_size > limit:
                        lines = lines[1:]  # The first line may start mid-record.
                    row.update(text='\n'.join(lines[-tail_lines:]),
                               truncated=stat.st_size > limit or len(lines) > tail_lines)
                elif stat.st_size > limit:
                    row.update(error='File exceeds bounded JSON read limit')
                else:
                    row['data'] = json.loads(stream.read(limit+1).decode('utf-8-sig'))
        except Exception as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
        return row

    selection = {'mode': requested_job}
    job_id = '24023'
    if requested_job == 'latest':
        candidates = []
        for path in sorted((root/'logs').glob('submission-*.json')):
            receipt = read_file(path).get('data', {})
            command = receipt.get('command', [])
            if (receipt.get('job_name') != 'comem-sparse-1gpu'
                    or receipt.get('destination_backend') != 'slurm-l20d'
                    or not re.fullmatch(r'[0-9]+', str(receipt.get('job_id', '')))
                    or not isinstance(command, list) or '--chdir' not in command
                    or command.index('--chdir')+1 >= len(command)
                    or command[command.index('--chdir')+1] != str(root)):
                raise ValueError('Invalid task submission receipt: '+str(path))
            stamp = datetime.fromisoformat(receipt['submitted_utc'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError('Submission timestamp must identify a timezone')
            candidates.append((stamp, str(receipt['job_id']), str(path)))
        if not candidates:
            raise ValueError('No valid submission receipt exists for this task')
        latest = max(row[0] for row in candidates)
        selected = [row for row in candidates if row[0] == latest]
        if len(selected) != 1:
            raise ValueError('Latest submission receipt is ambiguous')
        _, job_id, selection['receipt_path'] = selected[0]
    elif requested_job != '24023':
        raise ValueError('Only job 24023 or the latest task-owned receipt is allowed')

    slurm = {'command': ['scontrol', 'show', 'job', '-o', job_id]}
    try:
        proc = subprocess.run(slurm['command'], capture_output=True, text=True, timeout=20)
        slurm.update(returncode=proc.returncode, stdout=proc.stdout[:65536], stderr=proc.stderr[:8192])
        fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)', proc.stdout))
        slurm['fields'] = fields
        if proc.returncode == 0:
            if (fields.get('JobId') != job_id or fields.get('JobName') != 'comem-sparse-1gpu'
                    or fields.get('WorkDir') != str(root)):
                raise ValueError('Slurm job identity does not match the authorized task')
            slurm['task_identity_verified'] = True
        else:
            slurm['task_identity_verified'] = False
            slurm['state_unavailable'] = True
    except subprocess.TimeoutExpired:
        slurm.update(error='scontrol timed out', state_unavailable=True,
                     task_identity_verified=False)

    queue_file = read_file(output/'queue.json', limit=1048576)
    queue = queue_file.get('data', {})
    queue_matches = (queue.get('config', {}).get('task_root') == str(root)
                     and str(queue.get('allocation', {}).get('job_id')) == job_id)
    snapshots = {}
    for stage in ('smoke', 'train'):
        for arm in ('D0', 'A', 'B', 'D1'):
            key = stage+'/'+arm
            leaf = 'D0_retry1' if key == 'smoke/D0' else arm
            destination = output/stage/leaf
            job = queue.get('jobs', {}).get(key, {}) if queue_matches else {}
            if (key == 'train/A' and job.get('out') == str(output/'train/A_retry1')
                    and job.get('retry_of') == str(output/'train/A')
                    and job.get('recovery_receipt') == str(root/'logs/recovery-24023/recovery.json')):
                destination = output/'train/A_retry1'
            if job and job.get('out') != str(destination):
                raise ValueError('Unexpected output mapping in queue for '+key)
            status = read_file(destination/'status.json')
            log = read_file(destination/'worker.log', tail=True, limit=32768)
            lease_file = read_file(destination/'gpu_lease.json')
            admission_file = read_file(destination/'gpu_admission.json')
            lease = lease_file.get('data', {})
            updated = lease.get('updated_unix_s')
            anonymous = lease.get('heartbeat', {}).get('format') == 'anonymous-memfd-monotonic-v1'
            lease_observation = dict(updated_unix_s=updated,
                age_seconds=(time.time()-updated) if not anonymous and type(updated) in (int, float) else None,
                metadata_age_seconds=(time.time()-updated) if type(updated) in (int, float) else None,
                heartbeat_transport='anonymous-memfd' if anonymous else 'shared-file',
                controller_record=lease.get('controller'),
                matches_queue_identity=bool(queue_matches and job and lease
                    and lease.get('controller') == queue.get('controller')
                    and str(lease.get('allocation', {}).get('job_id')) == job_id
                    and lease.get('run_id') == job.get('run_id')),
                liveness_claim=('unverified; anonymous heartbeat is visible only on the compute node'
                    if anonymous else 'unverified; file heartbeat observation only'))
            last_event = None
            for line in log.get('text', '').splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and type(event.get('step')) is int:
                        last_event = event
                except ValueError:
                    pass
            snapshots[key] = dict(out=str(destination), exists=confined(destination).is_dir(),
                                  queue_job=job, status=status, worker_log=log,
                                  last_step_event=last_event, gpu_lease=lease_file,
                                  lease_observation=lease_observation, gpu_admission=admission_file)
    files = {name: read_file(output/name) for name in
             ('slurm_cpu_validation.json', 'slurm_device.json')}
    logs = {name: read_file(path, tail=True, limit=32768) for name, path in (
        ('slurm_stdout', root/'logs'/('slurm-'+job_id+'.out')),
        ('slurm_stderr', root/'logs'/('slurm-'+job_id+'.err')),
        ('cpu_validation_0', output/'slurm_cpu_validation_0.log'),
        ('cpu_validation_1', output/'slurm_cpu_validation_1.log'),
        ('device_probe', output/'slurm_device_probe.log'))}
    return dict(schema='sparse-slurm-inspection-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
                task_root=str(root), job_id=job_id, selection=selection, slurm=slurm,
                inspection_host=socket.gethostname(), queue_file=queue_file,
                queue_matches_selected_job=queue_matches, outputs=snapshots, files=files, logs=logs,
                process_liveness=dict(status='unverified', queue_host=queue.get('host'),
                    controller_record=queue.get('controller'),
                    slurm_node=slurm.get('fields', {}).get('NodeList'),
                    note='No login-node /proc probe. Slurm state and task-file timestamps are separate observations; recorded PIDs do not establish current compute-node liveness.'),
                formal_inference_timing=False, read_only=True)


def compact(snapshot):
    queue = snapshot.get('queue_file', {}).get('data', {})
    rows = {}
    for key, item in snapshot.get('outputs', {}).items():
        status = item.get('status', {}).get('data', {})
        event = item.get('last_step_event') or {}
        lease = item.get('lease_observation', {})
        checks = item.get('gpu_admission', {}).get('data', {}).get('checks', [])
        steps = [n for n in (status.get('step'), event.get('step')) if type(n) is int]
        rows[key] = dict(queue_phase=item.get('queue_job', {}).get('phase', 'unverified'),
                         artifact_phase=status.get('phase'), observed_step=max(steps) if steps else None,
                         error=(status.get('error') or item.get('queue_job', {}).get('error')),
                         heartbeat_age_seconds=lease.get('age_seconds'),
                         heartbeat_matches_queue_identity=lease.get('matches_queue_identity'),
                         last_admission_stage=checks[-1].get('stage') if checks else None,
                         status_age_seconds=item.get('status', {}).get('age_seconds'),
                         log_age_seconds=item.get('worker_log', {}).get('age_seconds'))
    return dict(job_id=snapshot.get('job_id'),
                slurm_state=snapshot.get('slurm', {}).get('fields', {}).get('JobState', 'unavailable'),
                node=snapshot.get('slurm', {}).get('fields', {}).get('NodeList'),
                queue_reason=queue.get('reason'), queue_updated_utc=queue.get('updated_utc'),
                queue_matches_selected_job=snapshot.get('queue_matches_selected_job'),
                process_liveness='unverified', jobs=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job', choices=('24023', 'latest'), default='24023')
    ap.add_argument('--tail-lines', type=int, default=24)
    ap.add_argument('--out', type=Path, help='New local directory strictly under this experiment/results')
    args = ap.parse_args()
    if not 1 <= args.tail_lines <= 80:
        ap.error('--tail-lines must be between 1 and 80')
    destination = (args.out or RESULTS/('slurm_inspect_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))).resolve()
    if RESULTS.resolve() not in destination.parents:
        ap.error('--out must be strictly under this experiment/results')
    destination.mkdir(parents=True, exist_ok=False)
    program = 'import json\n'+inspect.getsource(collect_remote)
    program += '\nprint(json.dumps(collect_remote('+repr(args.job)+', '+str(args.tail_lines)+'), ensure_ascii=False))\n'
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'gpu-node1',
               'env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /srv/encbank/qcomem_runtime_20260911/python312/bin/python -B -']
    try:
        proc = subprocess.run(command, input=program, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=55)
        if proc.returncode:
            raise RuntimeError('SSH inspection failed: '+proc.stderr[-8192:])
        snapshot = json.loads(proc.stdout)
        snapshot['ssh_stderr'] = proc.stderr[-8192:]
        summary = compact(snapshot)
        (destination/'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (destination/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('Saved '+str(destination/'snapshot.json'))
        print(f"Job {summary['job_id']}: {summary['slurm_state']} @ {summary['node']}; queue={summary['queue_reason']}; process liveness=unverified")
        for key, row in summary['jobs'].items():
            print(f"{key}: queue={row['queue_phase']} artifacts={row['artifact_phase']} step={row['observed_step']} admission={row['last_admission_stage']} heartbeat_age_s={row['heartbeat_age_seconds']}"+(('; error='+str(row['error'])[:300]) if row['error'] else ''))
        return 0
    except Exception as exc:
        failure = dict(status='inspection_failed', error=type(exc).__name__+': '+str(exc),
                       observed_utc=datetime.now(timezone.utc).isoformat(), command=command)
        (destination/'snapshot.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(failure, ensure_ascii=False))
        print('Saved '+str(destination/'snapshot.json'))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
