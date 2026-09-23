"""Clean only orphan Apptainer runtimes attributable to a completed diagnostic Slurm job."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('job_id')
args = parser.parse_args()
assert args.job_id.isdigit()
root = Path('/srv/encbank/qcomem_runtime_20260911/server_control_20260920')
assert (root / ('candidate-' + args.job_id + '.log')).is_file()
query = subprocess.run(['squeue', '-h', '-j', args.job_id, '-o', '%i'], capture_output=True, text=True)
assert query.returncode == 0 and not query.stdout.strip(), 'Diagnostic job is still active'
def identity(pid):
    directory = Path('/proc') / str(pid)
    stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), ppid=int(stat[1]), pgrp=int(stat[2]), start_ticks=stat[19], state=stat[0],
                uid=directory.stat().st_uid)


owned = {}
snapshot = {}
roots = []
observed_path = root / ('candidate-' + args.job_id + '-owned_processes.json')
observed = json.loads(observed_path.read_text()) if observed_path.exists() else []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        current = identity(directory.name)
        if current['uid'] != os.getuid():
            continue
        snapshot[current['pid']] = current
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if not command.startswith('Apptainer runtime parent: alexgshaw_'):
            continue
        match = next((row for row in observed if row['pid'] == current['pid']
                      and row['start_ticks'] == current['start_ticks']
                      and row['pgrp'] == current['pgrp'] and row['cmdline'] == command.strip()), None)
        if match:
            roots.append(current['pid'])
            continue
        env = dict(entry.split('=', 1) for entry in (directory / 'environ').read_bytes().decode().split('\0') if '=' in entry)
        if env.get('SLURM_JOB_ID') != args.job_id:
            continue
        assert env.get('APPTAINER_CACHEDIR') == str(root / 'apptainer_cache')
        roots.append(current['pid'])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
pending = roots[:]
while pending:
    pid = pending.pop()
    if pid in owned:
        continue
    owned[pid] = snapshot[pid]
    pending.extend(item['pid'] for item in snapshot.values() if item['ppid'] == pid)


def still_owned(record):
    try:
        current = identity(record['pid'])
        return current['uid'] == os.getuid() and current['start_ticks'] == record['start_ticks'] and current['state'] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


for sig in [signal.SIGTERM, signal.SIGKILL]:
    for record in reversed(list(owned.values())):
        if still_owned(record):
            try:
                os.kill(record['pid'], sig)
            except ProcessLookupError:
                pass
    time.sleep(2)
running = [record['pid'] for record in owned.values() if still_owned(record)]
report = dict(job_id=args.job_id, epoch=time.time(), tracked=list(owned.values()),
              remaining_running_pids=running, status='PASS' if not running else 'FAIL')
(root / ('candidate-' + args.job_id + '-cleanup.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert not running
