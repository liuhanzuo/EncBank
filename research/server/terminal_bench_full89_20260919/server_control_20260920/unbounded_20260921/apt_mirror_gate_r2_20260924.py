"""Release paused new trials only after Ubuntu security metadata is coherent.

This service never submits or repeats a benchmark trial.  Already active trials
continue while its read-only mirror checks run.  It removes only known admission
pause flags after three complete, SHA-verified checks.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = 'http://security.ubuntu.com/ubuntu/dists/noble-security/'
PROXY = 'http://PROXY_HOST:8888'
ROOT = Path('/cluster/home/USER/qencbank_align_codex_20260911/terminal_bench_full89_20260919/server_control_20260920')
FLAGS = [ROOT / f'{arm}_proxy_recovery_r2_{i}_20260924' / 'execution' / 'pause_new_tasks.json'
         for arm, count in [('k12', 5), ('k48', 3)] for i in range(1, count + 1)]
OUT = Path('/tmp/encbank_apt_mirror_gate_20260924.json')


def save(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def fetch(url):
    return subprocess.check_output(['curl', '-fsSL', '--max-time', '45', '-x', PROXY, url], timeout=55)


def check():
    release = fetch(BASE + 'InRelease').decode('utf-8')
    assert 'SHA256:' in release
    section = release.split('SHA256:\n', 1)[1].split('\nSHA512:', 1)[0]
    records = {}
    for line in section.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit():
            records[parts[2]] = (parts[0], int(parts[1]))
    verified = {}
    for component in ('main', 'restricted', 'universe', 'multiverse'):
        relative = f'{component}/binary-amd64/Packages.gz'
        expected_sha, expected_size = records[relative]
        content = fetch(BASE + relative)
        actual_sha = hashlib.sha256(content).hexdigest()
        assert len(content) == expected_size and actual_sha == expected_sha, (
            relative, expected_size, len(content), expected_sha, actual_sha)
        verified[relative] = {'bytes': len(content), 'sha256': actual_sha}
    # A recent 404 for libexpat accompanied the inconsistent package index.
    package = fetch('http://security.ubuntu.com/ubuntu/pool/main/e/expat/libexpat1_2.6.1-2ubuntu0.6_amd64.deb')
    assert package and len(package) > 10000
    return {'epoch': time.time(), 'release_sha256': hashlib.sha256(release.encode()).hexdigest(),
            'packages': verified, 'libexpat_bytes': len(package)}


def main():
    assert all(path.exists() for path in FLAGS), 'Admission flag is missing; review current controller state'
    successes = []
    attempts = 0
    while True:
        attempts += 1
        try:
            evidence = check()
            successes.append(evidence)
            successes = successes[-3:]
            save(OUT, {'state': 'CHECKING', 'attempts': attempts, 'consecutive_passes': len(successes),
                       'latest': evidence, 'epoch': time.time()})
            if len(successes) == 3:
                for path in FLAGS:
                    pause = json.loads(path.read_text())
                    assert pause['scope'] == 'new_task_admission_only'
                    save(path.parent / 'apt_mirror_gate_release.json',
                         {'epoch': time.time(), 'checks': successes, 'pause': pause})
                    path.unlink()
                save(OUT, {'state': 'RELEASED', 'attempts': attempts,
                           'consecutive_passes': 3, 'checks': successes, 'epoch': time.time()})
                print(json.dumps({'state': 'RELEASED', 'attempts': attempts}), flush=True)
                return
        except Exception as exc:
            successes = []
            save(OUT, {'state': 'WAITING_MIRROR', 'attempts': attempts,
                       'error': repr(exc), 'epoch': time.time()})
        time.sleep(30)


if __name__ == '__main__':
    main()
