"""Release held score-only shards as slots open, never above four main GPUs."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

S = Path('/cluster/home/liuhanzuo/qcomem_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
U = S / 'unbounded_20260921'
R = U / 'score_resume_20260925'
LOCK = S.parent.parent / 'terminal_bench_20260918' / 'admission.lock'
CAP = 4
ORDER = [('k12', 3), ('k48', 3), ('k12', 4), ('k12', 5), ('k12', 6)]


def read(path):
    return json.loads(Path(path).read_text())


def queue():
    output = subprocess.check_output(
        ['squeue', '-r', '-u', 'liuhanzuo', '-h', '-o', '%i|%j|%T|%b|%r'],
        text=True, timeout=20)
    rows = []
    for line in output.splitlines():
        job, name, state, gres, reason = line.split('|', 4)
        if (name.startswith('encbank-tb-') or name.startswith('qcomem-tb-')) and 'gpu' in gres.lower():
            rows.append(dict(job_id=job, name=name, state=state, gres=gres, reason=reason))
    return rows


def main(release):
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipts = {}
        for arm, n in ORDER:
            root = S / f'{arm}_score_resume_{n}_20260925'
            receipt = read(root / 'submission.json')
            assert receipt['status'] == 'submitted' and receipt['exit_code'] == 0
            receipts[receipt['job_id']] = (arm, n)
        rows = queue()
        eligible = [r for r in rows if r['reason'] not in ('JobHeldUser', 'JobHeldAdmin')]
        assert len(eligible) <= CAP, eligible
        available = CAP - len(eligible)
        held = [r for r in rows if r['job_id'] in receipts and r['state'] == 'PENDING'
                and r['reason'] == 'JobHeldUser']
        rank = {str(read(S / f'{arm}_score_resume_{n}_20260925' / 'submission.json')['job_id']): i
                for i, (arm, n) in enumerate(ORDER)}
        held.sort(key=lambda r: rank[r['job_id']])
        chosen = held[:available]
        print(json.dumps(dict(mode='release' if release else 'dry_run', eligible=len(eligible),
                              held=len(held), slots=available,
                              selected=[r['job_id'] for r in chosen])), flush=True)
        if not release:
            return
        for row in chosen:
            child = subprocess.run(['scontrol', 'release', row['job_id']],
                                   capture_output=True, text=True, timeout=20)
            assert child.returncode == 0, dict(job_id=row['job_id'], stderr=child.stderr)
            current = queue()
            eligible_now = [r for r in current if r['reason'] not in ('JobHeldUser', 'JobHeldAdmin')]
            assert len(eligible_now) <= CAP, eligible_now
            event = dict(epoch=time.time(), job_id=row['job_id'], source=str(Path(__file__)),
                         eligible_after=len(eligible_now), actual_parent_wait=True)
            with (R / 'release_events.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', action='store_true')
    main(parser.parse_args().release)
