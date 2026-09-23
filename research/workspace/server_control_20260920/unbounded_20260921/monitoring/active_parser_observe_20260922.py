"""Read-only cumulative parser-feedback snapshot; tolerate concurrent trajectory writes."""
import hashlib
import json
import time
from pathlib import Path

S = Path('/srv/encbank/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
B = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

report = dict(epoch=time.time(), scope='Cumulative feedback in active trajectories, not a count of new failures.',
    runs={}, read_errors=[], transient_reads=[])
registry = read(S / 'unbounded_20260921/scale4_20260921/registry.json')
for spec in registry['runs']:
    home = Path(spec['root'])
    state = read(home / 'execution/status.json') or {}
    if (home / 'server_job_receipt.json').exists() or not state.get('active'):
        continue
    rows = {}
    report['runs'][spec['id']] = rows
    for task in state['active']:
        if task not in spec['tasks']:
            continue
        paths = list((B / home.name / 'results' / task).glob('*/agent/trajectory.json'))
        row = dict(paths=len(paths))
        rows[task] = row
        if len(paths) != 1:
            continue
        path = paths[0]
        for attempt in range(3):
            try:
                before = path.stat()
                raw = path.read_bytes()
                trajectory = json.loads(raw)
                after = path.stat()
                break
            except (OSError, ValueError) as exc:
                evidence = dict(run=spec['id'], task=task, attempt=attempt + 1, error=str(exc)[:500])
                if attempt == 2:
                    report['read_errors'].append(evidence)
                    trajectory = None
                else:
                    report['transient_reads'].append(evidence)
                    time.sleep(.25)
        if trajectory is None:
            continue
        agents = [s for s in trajectory.get('steps', []) if s.get('source') == 'agent']
        warnings, errors = [], []
        for step in agents:
            for observation in (step.get('observation') or {}).get('results', []):
                content = str(observation.get('content', ''))
                item = dict(step=step['step_id'], timestamp=step.get('timestamp'),
                    message=content.split('New Terminal Output:', 1)[0][:500])
                if content.startswith('Previous response had warnings:'):
                    warnings.append(item)
                if content.startswith(('Previous response had parsing errors:', 'Previous response had errors:')):
                    errors.append(item)
        row.update(path=str(path), bytes=len(raw), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
            mtime=before.st_mtime,
            file_stable_during_read=(before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size),
            agent_steps=len(agents), last_agent_step=agents[-1]['step_id'] if agents else None,
            last_agent_timestamp=agents[-1].get('timestamp') if agents else None,
            parser_warning_count=len(warnings), parser_error_count=len(errors),
            recent_parser_warnings=warnings[-3:], recent_parser_errors=errors[-3:])
print(json.dumps(report))
