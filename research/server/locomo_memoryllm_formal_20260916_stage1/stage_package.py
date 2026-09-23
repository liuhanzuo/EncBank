"""Native SSH/SCP package promotion and CPU validation only; never submits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

HOME = Path('/srv/encbank')
CONTROL = HOME / 'qencbank_align_codex_20260911/locomo_memoryllm_formal_20260916_stage1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    a = parser.parse_args()
    CONTROL.resolve().relative_to(HOME)
    archive = CONTROL / 'package.tar'
    assert sha(archive) == a.archive_sha256
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(m.isfile() and not m.issym() and not m.islnk() for m in members)
        manifest_bytes = tar.extractfile('package_manifest.json').read()
        assert hashlib.sha256(manifest_bytes).hexdigest() == a.manifest_sha256
        manifest = json.loads(manifest_bytes)
        here = Path(manifest['remote_package']).resolve()
        here.relative_to(HOME)
        expected = {x['path']: x for x in manifest['files']}
        expected['package_manifest.json'] = dict(sha256=a.manifest_sha256, bytes=len(manifest_bytes))
        assert {m.name for m in members} == set(expected)
        for member in members:
            path = (here / member.name).resolve()
            path.relative_to(here)
            blob = tar.extractfile(member).read()
            assert hashlib.sha256(blob).hexdigest() == expected[member.name]['sha256']
            assert len(blob) == expected[member.name]['bytes']
            if path.exists():
                assert sha(path) == expected[member.name]['sha256'], 'No overwrite of differing package'
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
    plan = json.loads((here / 'plan.json').read_text())
    assert sha(here / 'plan.json') == manifest['plan_sha256']
    cache = Path(plan['task_cache_root']).resolve()
    cache.relative_to(HOME)
    for leaf in ['tmp', 'xdg', 'hf', 'torch', 'triton']:
        (cache / leaf).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(cache / 'tmp'),
               HF_HOME=str(cache / 'hf'), TORCH_HOME=str(cache / 'torch'),
               XDG_CACHE_HOME=str(cache / 'xdg'), TRITON_CACHE_DIR=str(cache / 'triton'))
    checks = []
    for name, command in [('full_preflight', [plan['runtime'], '-B', str(here / 'run_formal.py'),
                            '--expected-plan-sha256', manifest['plan_sha256'], '--check-only']),
                          ('bash_syntax', ['bash', '-n', str(here / 'batch.sbatch')])]:
        child = subprocess.run(command, capture_output=True, text=True, env=env)
        record = dict(name=name, argv=command, actual_exit_code=child.returncode,
                      stdout=child.stdout, stderr=child.stderr, parent_wait_observed=True)
        checks.append(record)
        (CONTROL / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        assert child.returncode == 0, record
    result = dict(status='PASS', plan_sha256=manifest['plan_sha256'],
                  package_manifest_sha256=a.manifest_sha256, checks=checks,
                  files=len(expected), observed_at_epoch=time.time(),
                  GPU_or_submission_actions=0)
    (here / 'staged_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    (CONTROL / 'staged_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
