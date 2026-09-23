"""Read only this sweep's jobs and package completed non-checkpoint results."""
import argparse
import datetime
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--collect', action='store_true')
a = p.parse_args()
plan = json.loads((ROOT / 'plan.json').read_text())
launch = json.loads((ROOT / 'launch.json').read_text())


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def inspect(path):
    out = {'run': str(path)}
    for name in ('progress.json', 'failure.json', 'correctness.json', 'correctness_adapted.json', 'base_reuse_checks.json'):
        if (path / name).exists():
            out[name] = read(path / name)
    verified = read(path / 'verified_summary.json')
    out['verified'] = bool(verified and verified.get('verified'))
    if verified:
        out.update(samples=verified['samples'], predictions=verified['predictions'])
    return out


report = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **launch,
          'planned_depths': sum(len(x['depths']) for x in plan['models'].values()), 'models': {}}
for command, field in [(['squeue', '-j', launch['array_job'], '-h', '-o', '%i %T %M %R'], 'queue'),
                       (['sacct', '-j', launch['array_job'], '--noheader', '--parsable2', '--format=JobID,JobName,State,ExitCode,Elapsed,NodeList'], 'accounting')]:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    report[field] = result.stdout.strip().splitlines()
    if result.returncode:
        report[field + '_error'] = result.stderr.strip()
selected = []
for name, info in plan['models'].items():
    entries = {}
    for j in info['depths']:
        if j == info['anchor_j']:
            runs = [Path(info['anchor'])]
        else:
            runs = sorted((ROOT / 'results' / name / f'j{j:02}').glob('job_*'), key=lambda x: x.stat().st_mtime)
        attempts = [inspect(run) for run in runs]
        valid = [item for item in attempts if item['verified'] and 'failure.json' not in item]
        current = valid[-1] if valid else (attempts[-1] if attempts else {'verified': False, 'progress.json': {'phase': 'not_started'}})
        entries[str(j)] = {**current, 'anchor_reused': j == info['anchor_j'], 'attempts': attempts}
        if valid:
            selected.append((name, j, Path(valid[-1]['run'])))
    report['models'][name] = entries
report['verified_depths'] = len(selected)
report['all_verified'] = len(selected) == report['planned_depths']
if a.collect:
    archive = ROOT / 'sweep_results.tar.gz'
    allowed = ['protocol.json', 'training.jsonl', 'predictions.jsonl', 'samples.jsonl.gz',
               'correctness.json', 'correctness_adapted.json', 'base_reuse_checks.json',
               'summary.json', 'verified_summary.json', 'progress.json']
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(ROOT / 'plan.json', arcname='plan.json')
        for name, j, path in selected:
            for filename in allowed:
                if (path / filename).exists():
                    tar.add(path / filename, arcname=f'results/{name}/j{j:02}/{filename}')
    report['archive'] = str(archive)
print(json.dumps(report, indent=2))
